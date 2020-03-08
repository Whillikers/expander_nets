"""
Implements the "addition" task from Graves 2016: adding decimal
numbers of random sizes.
"""

import math
import random
import string
from typing import List, Tuple

import torch

from expander_nets.tasks import utils

EMPTY = "-"
TOKENS = string.digits + EMPTY

N_DIGITS = 10
N_TOKENS = 11
EMPTY_CLASS = 10  # Class index of "empty" digits (beyond the number's end)


# pylint: disable=too-few-public-methods
# pylint: disable=abstract-method
class AdditionDataset(torch.utils.data.IterableDataset):  # type: ignore
    """
    A torch IterableDataset for addition problems.

    Parameters
    ----------
    min_numbers: int (default: 1)
        Minimum sequence length.
    max_numbers: int (default: 5)
        Maximum sequence length.
    max_digits: int (default: 5)
        The maximum number of digits each number can have.
    """

    min_numbers: int
    max_numbers: int
    max_digits: int

    def __init__(
        self, min_numbers: int = 1, max_numbers: int = 5, max_digits: int = 5
    ):
        if min_numbers <= 0:
            raise ValueError("min_numbers must be at least 1.")
        if max_numbers <= 0:
            raise ValueError("min_numbers must be at least 1.")
        if max_digits <= 0:
            raise ValueError("max_digits must be at least 1.")

        self.min_numbers = min_numbers
        self.max_numbers = max_numbers
        self.max_digits = max_digits
        self.feature_size = N_DIGITS * max_digits

        # Find the number of digits we need to allocate to the target
        max_sum = max_numbers * math.pow(10, self.max_digits)
        self.target_size = math.ceil(math.log10(max_sum))

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self) -> utils.Example:
        num_numbers = random.randint(self.min_numbers, self.max_numbers)
        cumsum = 0
        features = torch.empty(
            [num_numbers, self.feature_size], dtype=torch.int8
        )
        targets = torch.empty([num_numbers, self.target_size], dtype=torch.int8)

        for idx in range(num_numbers):
            number, digits = self._get_number_and_digits()
            cumsum += number
            digits_onehot = list(map(_onehot, digits))
            features[idx] = torch.cat(digits_onehot)

            sum_digits = [string.digits.index(digit) for digit in str(cumsum)]
            missing_digits = self.target_size - len(sum_digits)
            sum_digits.extend([EMPTY_CLASS] * missing_digits)
            targets[idx] = torch.as_tensor(sum_digits, dtype=torch.int8)

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

    onehot = torch.zeros(N_DIGITS, dtype=torch.int8)
    if token != EMPTY:
        onehot[int(token)] = True
    return onehot
