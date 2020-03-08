"""
Code for various models and components.
"""


import torch
from torch import nn
from torch.nn import functional as F  # NOQA


# Utils
def _repeat_and_flag(inputs: torch.Tensor, times: int) -> torch.Tensor:
    # Add a binary start-of-step flag (shape: [B, D + 1])
    padded = F.pad(inputs, [0, 1], mode="constant", value=0.0)

    # Repeat the inputs a fixed number of times (shape: [times, B, D + 1])
    repeated = padded.unsqueeze(0).repeat([times, 1, 1])

    # Mark the first copy as such (other flags remain 0)
    repeated[0, :, -1] = 1.0
    return repeated


# Task heads
class BinaryClassifier(nn.Module):
    """
    Use a sequential encoder to make a single binary classification by running
    it on a unary sequence. Outputs a single logit.

    Parameters
    ----------
    encoder: nn.Module
        Any sequence-to-sequence model.
    """

    def __init__(self, encoder: nn.Module):
        super(BinaryClassifier, self).__init__()
        self.encoder = encoder
        self.linear = nn.Linear(encoder.hidden_size, 1)  # type: ignore

    # pylint: disable=arguments-differ
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore
        encoded, _ = self.encoder(inputs)
        logit = self.linear(encoded).squeeze()
        return logit


# Other models
class RepeatRNN(nn.Module):
    """
    An implementation of RepeatRNN from Fojo et al, which modifies an existing
    RNN by repeating its action a fixed number of times at each time step.
    Operates on a single input at a time.

    NOTE: if mark_first is True (default), has an input size of one less
    than the wrapped rnn.

    Parameters
    ----------
    rnn: nn.RNNBase
        The base RNN to modify through repeating.
    repeats: int
        How many steps to repeat per input.
    """

    rnn: nn.RNNBase
    repeats: int
    input_size: int
    hidden_size: int

    def __init__(self, rnn: nn.RNNBase, repeats: int):
        if repeats < 1:
            raise ValueError("repeats must be positive.")

        super(RepeatRNN, self).__init__()
        self.rnn = rnn
        self.repeats = repeats
        self.hidden_size = rnn.hidden_size
        self.input_size = rnn.input_size - 1

    # pylint: disable=arguments-differ
    def forward(self, inputs: torch.Tensor, state=None):  # type: ignore
        repeated = _repeat_and_flag(inputs, self.repeats)
        _, state = self.rnn(repeated)
        hidden = state[0]
        return hidden, state
