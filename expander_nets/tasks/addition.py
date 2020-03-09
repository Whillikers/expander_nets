"""
Implements the "addition" task from Graves 2016: adding decimal
numbers of random sizes.
"""

import math
import random
import string
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from expander_nets import utils

EMPTY = "-"
TOKENS = string.digits + EMPTY

NUM_DIGITS = 10
NUM_CLASSES = 11
EMPTY_CLASS = 10  # Class index of "empty" digits (beyond the number's end)

TASK_NAME = "addition"


class AdditionClassifier(nn.Module):
    """
    A model which uses a base RNN to classify digits for the addition task.

    Parameters
    ----------
    rnn: nn.Module
        Base RNN which generates features for a digit sequence.
    num_output_digits: int
        Number of digits in each target.
    """

    rnn: nn.Module
    num_output_digits: int
    num_logits: int

    def __init__(self, rnn: nn.Module, num_output_digits: int):
        super(AdditionClassifier, self).__init__()
        self.num_output_digits = num_output_digits
        self.num_logits = num_output_digits * NUM_CLASSES

        self.rnn = rnn
        self.linear = nn.Linear(rnn.hidden_size, self.num_logits)  # type: ignore

    # pylint: disable=arguments-differ
    def forward(  # type: ignore
        self, inputs: torch.Tensor, state=None
    ) -> Tuple[torch.Tensor, Any]:
        encoded, state = self.rnn(inputs, state)
        logits = self.linear(encoded)  # Shape: [seq_length, batch, num_logits]
        logits = logits.view(
            [
                logits.size(0),
                logits.size(1),
                self.num_output_digits,
                NUM_CLASSES,
            ]
        )
        return logits, state


def train(
    rnn: nn.Module,
    batch_size: int,
    learning_rate: float,
    sequence_length: int,
    max_digits: int,
    use_gpu: bool = True,
    data_workers: int = 4,
    run_name: Optional[str] = None,
):

    summary_writer = utils.get_summary_writer(TASK_NAME, run_name)
    use_gpu = use_gpu and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    dataset = AdditionDataset(sequence_length, max_digits)
    data_loader = torch.utils.data.DataLoader(  # type: ignore
        dataset,
        batch_size=batch_size,
        pin_memory=use_gpu,
        num_workers=data_workers,
        collate_fn=utils.collate_sequences,
    )

    model = AdditionClassifier(rnn, dataset.target_size).train().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    running_loss = 0.0
    running_correct = 0
    running_sequence_correct = 0
    step_count = 0

    def summary_step(step):
        nonlocal running_loss
        nonlocal running_correct
        nonlocal running_sequence_correct
        nonlocal step_count

        avg_loss = running_loss / utils.SUMMARY_PERIOD
        avg_accuracy = running_correct / step_count
        avg_sequence_accuracy = running_sequence_correct / (
            utils.SUMMARY_PERIOD * batch_size
        )
        summary_writer.add_scalar("Loss", avg_loss, step)
        summary_writer.add_scalar("Accuracy", avg_accuracy, step)
        summary_writer.add_scalar(
            "Sequence Accuracy", avg_sequence_accuracy, step
        )
        running_loss = 0.0
        running_correct = 0
        running_sequence_correct = 0
        step_count = 0

        print(
            f"[{step}] {avg_loss=:.3f} {avg_accuracy=:.3f} "
            f"{avg_sequence_accuracy=:.3f}"
        )

    for step, (features, targets) in enumerate(data_loader):
        if use_gpu:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits, _ = model(features)
        log_prob = F.log_softmax(logits, dim=-1)

        loss = F.nll_loss(
            log_prob.view(-1, NUM_CLASSES),
            targets.view(-1),
            ignore_index=utils.MASK_VALUE,
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        matches = (logits.detach().argmax(-1) == targets)[1:]
        running_correct += matches.sum().item()
        running_sequence_correct += matches.all(0).all(-1).sum().item()
        step_count += matches.nelement()

        if step and step % utils.SUMMARY_PERIOD == 0:
            summary_step(step)


# pylint: disable=too-few-public-methods
# pylint: disable=abstract-method
class AdditionDataset(torch.utils.data.IterableDataset):  # type: ignore
    """
    A torch IterableDataset for addition problems.

    Parameters
    ----------
    sequence_length: int (default: 5)
        The length of sequence to add.
    max_digits: int (default: 5)
        The maximum number of digits each number can have.
    """

    sequence_length: int
    max_digits: int
    feature_size: int
    target_size: int

    def __init__(self, sequence_length: int, max_digits: int = 5):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be at least 1.")
        if max_digits <= 0:
            raise ValueError("max_digits must be at least 1.")

        self.sequence_length = sequence_length
        self.max_digits = max_digits
        self.feature_size = NUM_DIGITS * max_digits

        # Find the number of digits we need to allocate to the target.
        # Sum is bounded by sequence_length * (10^max_digits); requires
        # log10(sequence_length * (10^max_digits))
        # = max_digits + log10(sequence_length) digits to represent.
        self.target_size = max_digits + math.ceil(math.log10(sequence_length))

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self) -> utils.Example:
        cumsum = 0
        features = torch.empty(
            [self.sequence_length, self.feature_size], dtype=torch.float32
        )
        targets = torch.empty(
            [self.sequence_length, self.target_size], dtype=torch.long
        )

        for idx in range(self.sequence_length):
            number, digits = self._get_number_and_digits()
            cumsum += number
            digits_onehot = list(map(_onehot, digits))
            features[idx] = torch.cat(digits_onehot)

            if idx > 0:
                sum_digits = [
                    string.digits.index(digit) for digit in str(cumsum)
                ]
                missing_digits = self.target_size - len(sum_digits)
                sum_digits.extend([EMPTY_CLASS] * missing_digits)
                targets[idx] = torch.as_tensor(sum_digits, dtype=torch.long)
            else:  # Mask out first targets
                targets[idx, :] = utils.MASK_VALUE

        return features, targets

    def _get_number_and_digits(self) -> Tuple[int, List[str]]:
        num_digits = random.randint(1, self.max_digits)
        number = 0
        digits = []
        for _ in range(num_digits):
            digit = random.randint(0, 9)
            number = number * 10 + digit
            digits.append(str(digit))

        digits.extend(["-"] * (self.max_digits - num_digits))
        return number, digits


def _onehot(token: str) -> torch.Tensor:
    if len(token) != 1 or token not in TOKENS:
        raise ValueError(f"token must be one of [{TOKENS}]")

    onehot = torch.zeros(NUM_DIGITS, dtype=torch.float32)
    if token != EMPTY:
        onehot[int(token)] = 1.0
    return onehot
