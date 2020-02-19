"""
Implements the "parity" task from Graves 2016: determining the parity of a
statically-presented binary vector.
"""

from typing import Tuple

import torch

ParityExample = Tuple[torch.Tensor, torch.Tensor]


class BinaryParityDataset(torch.utils.data.IterableDataset):
    """
    A torch IterableDataset for binary parity problems.

    Parameters
    ----------
    size : int
        The length of each binary vector.
    difficulty : float
        The average fraction of each binary vector that is nonzero.
    """

    def __init__(self, size: int, difficulty: float):
        if size <= 0:
            raise ValueError("size must be at least one.")

        if not 0 <= difficulty <= 1:
            raise ValueError("difficulty must be in range [0, 1]")

        self._difficulty = difficulty
        self._size = size

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self) -> ParityExample:
        vec = torch.randint(2, (self._size,), dtype=torch.int8) * 2 - 1
        mask = torch.rand_like(vec, dtype=float) < self._difficulty
        feature = vec * mask
        return (feature, (feature == 0).sum() % 2)
