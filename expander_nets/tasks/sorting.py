"""
Implements the "sorting" task from Graves 2016: sorting randomly-sized lists
of random numbers.
"""

import random

import numpy as np  # type: ignore
import torch

from expander_nets.tasks import utils

SEQUENCE_END = 1
MASK_VALUE = -2


# pylint: disable=too-few-public-methods
# pylint: disable=abstract-method
class SortingDataset(torch.utils.data.IterableDataset):  # type: ignore
    """
    A torch IterableDataset for sorting problems.

    Features are (number, flag) pairs, where flag is 0 for continuing the
    sequence and 1 at and beyond the end of the sequence and numbers
    are generated from the normal distribution. Targets are MASK_VALUE during
    the input sequence and argsort indices otherwise.

    Parameters
    ----------
    min_length: int (default: 1)
        Minimum count of numbers to sort.
    max_length: int (default: 15)
        Maximum count of numbers to sort.
    stdev: float (default: 1.0)
        Standard deviation of the distribution to generate numbers from.
    min_separation: float (default: 0.0)
        Minimum separation between numbers. If 0, any separation is acceptable.
        NOTE: long sequences with large minimum separation may take a long time
        to generate.
    """

    min_length: int
    max_length: int
    stdev: float
    min_separation: float

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 15,
        stdev: float = 1.0,
        min_separation: float = 0.0,
    ):
        if min_length <= 0:
            raise ValueError("min_length must be at least 1.")
        if max_length < min_length:
            raise ValueError("max_length must be at least min_length.")
        if stdev <= 0.0:
            raise ValueError("stdev must be positive.")
        if min_separation < 0.0:
            raise ValueError("min_separation must be nonnegative.")

        self.min_length = min_length
        self.max_length = max_length
        self.stdev = stdev
        self.min_separation = min_separation

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self) -> utils.Example:
        num_numbers = random.randint(self.min_length, self.max_length)
        # Sequence is separated into (input, target) sections
        length = 2 * num_numbers

        numbers = self._get_numbers(num_numbers)

        features = torch.zeros([length, 2], dtype=torch.float32)
        features[:num_numbers, 0] = numbers
        features[num_numbers - 1 :, 1] = 1

        # pylint: disable=not-callable
        mask_val = torch.tensor(MASK_VALUE, dtype=torch.int8)
        targets = torch.repeat_interleave(mask_val, length)
        targets[num_numbers:] = numbers.argsort()

        return features, targets

    # NOTE: there may be a more efficient way to do this than repeated sampling?
    def _get_numbers(self, length: int) -> torch.Tensor:
        while True:
            sample = np.random.normal(scale=self.stdev, size=length)
            if _separation(sample) >= self.min_separation:
                return torch.tensor(sample)  # pylint: disable=not-callable


def _separation(sample: np.ndarray) -> float:
    sample_sorted = np.sort(sample)
    # Compute pairwise differences by subtracting the sorted array
    # from a shifted copy of itself
    differences = sample_sorted[1:] - sample_sorted[:-1]
    return differences.min()
