"""
Implements the "parity" task from Graves 2016: determining the parity of a
statically-presented binary vector.
"""

import random
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F  # NOQA

from expander_nets import models, utils

TASK_NAME = "parity"


class ParityClassifier(nn.Module):
    """
    Us a sequential encoder to make a single binary classification by running
    it on a unary sequence. Outputs a single logit.

    Parameters
    ----------
    rnn: nn.Module
        Any sequence-to-sequence model.
    """

    rnn: nn.Module

    def __init__(self, rnn: nn.Module):
        super(ParityClassifier, self).__init__()
        self.rnn = rnn
        self.linear = nn.Linear(rnn.hidden_size, 1)  # type: ignore

    # pylint: disable=arguments-differ
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore
        encoded, _ = self.rnn(inputs)
        logit = self.linear(encoded).squeeze()
        return logit


def train(
    rnn: nn.Module,
    bits: int,
    batch_size: int,
    learning_rate: float,
    use_gpu: bool = True,
    data_workers: int = 4,
    run_name: Optional[str] = None,
):
    """
    Train a model on the binary parity task.

    Parameters
    ----------
    rnn: nn.Module
        A model mapping `bits`-vectors into some hidden vector.
    """
    summary_writer = utils.get_summary_writer(TASK_NAME, run_name)
    use_gpu = use_gpu and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    dataset = BinaryParityDataset(bits)
    data_loader = torch.utils.data.DataLoader(  # type: ignore
        dataset,
        batch_size=batch_size,
        pin_memory=use_gpu,
        num_workers=data_workers,
    )

    model = ParityClassifier(rnn).train().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    running_loss = 0.0
    running_correct = 0

    def summary_step(step):
        nonlocal running_loss
        nonlocal running_correct

        avg_loss = running_loss / utils.SUMMARY_PERIOD
        avg_accuracy = running_correct / (utils.SUMMARY_PERIOD * batch_size)
        summary_writer.add_scalar("loss", avg_loss, step)
        summary_writer.add_scalar("accuracy", avg_accuracy, step)
        running_loss = 0.0
        running_correct = 0

        print(f"[{step}] {avg_loss=:.3f} {avg_accuracy=:.3f}")

    for step, (vector, parity) in enumerate(data_loader):
        if use_gpu:
            vector = vector.to(device, non_blocking=True)
            parity = parity.to(device, non_blocking=True)

        optimizer.zero_grad()
        logit = model(vector)
        loss = F.binary_cross_entropy_with_logits(logit, parity)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_correct += ((logit.detach() > 0) == parity).sum().item()
        if step and step % utils.SUMMARY_PERIOD == 0:
            summary_step(step)


# pylint: disable=too-few-public-methods
# pylint: disable=abstract-method
class BinaryParityDataset(torch.utils.data.IterableDataset):  # type: ignore
    """
    A torch IterableDataset for binary parity problems.

    Parameters
    ----------
    bits: int
        The length of each binary vector.
    """

    bits: int

    def __init__(self, bits: int):
        if bits <= 0:
            raise ValueError("bits must be at least one.")

        self.bits = bits

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self) -> utils.Example:
        vec = torch.randint(2, (self.bits,), dtype=torch.float32) * 2 - 1
        num_bits = random.randint(1, self.bits)
        vec[num_bits:] = 0
        parity = (vec == 1).sum(dtype=torch.float32) % 2
        return vec, parity


# TODO: remove
if __name__ == "__main__":
    repeat_rnn = models.RepeatRNN(
        nn.LSTM, input_size=64, hidden_size=128, repeats=2
    )
    train(repeat_rnn, 64, 128, 1e-3, run_name="test")
