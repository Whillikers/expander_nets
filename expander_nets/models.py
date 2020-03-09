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
