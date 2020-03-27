"""
Code for various models and components.
"""


from typing import Any, Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Interface types
RNNOutput = Tuple[torch.Tensor, Any]  # (output sequence, state)


# Utils
def _repeat_and_flag(inputs: torch.Tensor, times: int) -> torch.Tensor:
    # Add a binary start-of-step flag (shape: [S, B, D + 1])
    padded = F.pad(inputs, [0, 1], mode="constant", value=0.0)

    # Repeat the inputs a fixed number of times (shape: [S * times, B, D + 1])
    repeated = padded.repeat_interleave(times, dim=0)

    # Mark the first copy as such (other flags remain 0)
    repeated[::times, :, -1] = 1.0
    return repeated


# Allocators
class FixedLSTMAllocator(nn.Module):
    """
    An allocator which initializes a fixed amount of scratch space, initializing
    using an LSTM on the repeated input.

    Parameters
    ----------
    input_size: int
        The number of expected features in the input.
    hidden_size
        The number of features in the hidden state.
    space: int
        How many scratch values to allocate.
    """

    input_size: int
    hidden_size: int
    space: int
    lstm: nn.LSTM

    def __init__(self, input_size: int, hidden_size: int, space: int):
        if input_size < 1:
            raise ValueError("input_size must be positive.")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        if space < 1:
            raise ValueError("space must be positive.")

        super(FixedLSTMAllocator, self).__init__()
        self.space = space
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Reserve a slot for marking the first occurrence of each input
        self.lstm = nn.LSTM(input_size + 1, hidden_size)

    # pylint: disable=arguments-differ
    def forward(self, inputs: torch.Tensor, state=None):  # type: ignore
        repeated = _repeat_and_flag(inputs, self.space)
        initial_values, (_, cell_state) = self.lstm(repeated, state)
        return initial_values, cell_state


# TODO: clean up into one class
class FixedGRUAllocator(nn.Module):
    """
    An allocator which initializes a fixed amount of scratch space, initializing
    using a GRU on the repeated input.

    Parameters
    ----------
    input_size: int
        The number of expected features in the input.
    hidden_size
        The number of features in the hidden state.
    space: int
        How many scratch values to allocate.
    """

    input_size: int
    hidden_size: int
    space: int
    gru: nn.GRU

    def __init__(self, input_size: int, hidden_size: int, space: int):
        if input_size < 1:
            raise ValueError("input_size must be positive.")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        if space < 1:
            raise ValueError("space must be positive.")

        super(FixedGRUAllocator, self).__init__()
        self.space = space
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Reserve a slot for marking the first occurrence of each input
        self.gru = nn.GRU(input_size + 1, hidden_size)

    # pylint: disable=arguments-differ
    def forward(self, inputs: torch.Tensor, state=None):  # type: ignore
        repeated = _repeat_and_flag(inputs, self.space)
        allocated_values, _ = self.gru(repeated)
        return allocated_values


# TODO: refactor
class FixedGRUExpander(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        space: int,
        translator_dim: int,
        num_layers: int,
        heads: int = 1,
    ):
        super(FixedGRUExpander, self).__init__()
        self.space = space
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.translator_dim = translator_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size + 1, hidden_size)

        norm = nn.LayerNorm(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(  # type: ignore
            hidden_size, nhead=heads, dim_feedforward=translator_dim, dropout=0
        )
        self.translator = nn.TransformerEncoder(  # type: ignore
            encoder_layer, num_layers=num_layers, norm=norm
        )

    # pylint: disable=arguments-differ
    def forward(self, inputs: torch.Tensor, state=None):  # type: ignore
        outputs = []
        for input_ in inputs:
            if state is not None:
                state = state.unsqueeze(0)

            repeated = _repeat_and_flag(input_.unsqueeze(0), self.space)
            initial_values, _ = self.gru(repeated, state)
            state = self.translator(initial_values).mean(0)
            outputs.append(state)

        return torch.stack(outputs), state


class FixedLSTMExpander(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        space: int,
        translator_dim: int,
        num_layers: int,
        heads: int = 1,
    ):
        super(FixedLSTMExpander, self).__init__()
        self.space = space
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.translator_dim = translator_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size + 1, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(  # type: ignore
            hidden_size, nhead=heads, dim_feedforward=translator_dim, dropout=0
        )
        self.translator = nn.TransformerEncoder(  # type: ignore
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_size)
        )

    # pylint: disable=arguments-differ
    def forward(self, inputs: torch.Tensor, state=None):  # type: ignore
        outputs = []
        for input_ in inputs:
            repeated = _repeat_and_flag(input_.unsqueeze(0), self.space)
            initial_values, (_, cell_state) = self.lstm(repeated, state)
            hidden_state = self.translator(initial_values).mean(0)
            state = (hidden_state.unsqueeze(0), cell_state)
            outputs.append(hidden_state)

        return torch.stack(outputs), state


# Other models
RNNConstructor = Callable[[int, int], nn.RNNBase]  # input, hidden -> RNN


class RepeatRNN(nn.Module):
    """
    An implementation of RepeatRNN from Fojo et al, which modifies an existing
    RNN by repeating its action a fixed number of times at each time step.

    Parameters
    ----------
    rnn: RNNConstructor
        The base RNN type to modify through repeating.
    input_size: int
        Number of expected features in the input.
    hidden_size: int
        Number of features in the hidden state.
    repeats: int
        How many steps to repeat per input.
    """

    rnn: nn.RNNBase
    repeats: int
    input_size: int
    hidden_size: int

    def __init__(
        self,
        rnn_func: RNNConstructor,
        input_size: int,
        hidden_size: int,
        repeats: int,
    ):
        if repeats < 1:
            raise ValueError("repeats must be positive.")

        super(RepeatRNN, self).__init__()
        self.repeats = repeats
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Reserve one spot for the start-of-sequence flag
        self.rnn = rnn_func(input_size + 1, hidden_size)

    # pylint: disable=arguments-differ
    def forward(self, inputs: torch.Tensor, state=None) -> RNNOutput:  # type: ignore
        repeated = _repeat_and_flag(inputs, self.repeats)
        outputs, state = self.rnn(repeated)
        # Select only the last output for each input
        return outputs[self.repeats - 1 :: self.repeats, :, :], state
