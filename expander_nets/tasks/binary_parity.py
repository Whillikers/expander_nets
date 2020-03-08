"""
Implements the "parity" task from Graves 2016: determining the parity of a
statically-presented binary vector.
"""

import random

import torch

from expander_nets.tasks import utils


# pylint: disable=too-few-public-methods
# pylint: disable=abstract-method
class BinaryParityDataset(torch.utils.data.IterableDataset):  # type: ignore
    """
    A torch IterableDataset for binary parity problems.

    Parameters
    ----------
    size: int (default: 64)
        The length of each binary vector.
    difficulty: float (default: 0.5)
        The average fraction of each binary vector that is nonzero.
        Must be in the range [0, 1].
    """

    size: int

    def __init__(self, size: int = 64):
        if size <= 0:
            raise ValueError("size must be at least one.")

        self.size = size

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self) -> utils.Example:
        vec = torch.randint(2, (self.size,), dtype=torch.float32) * 2 - 1
        num_bits = random.randint(1, self.size)
        vec[num_bits:] = 0
        parity = (vec == 1).sum() % 2
        return vec, parity
