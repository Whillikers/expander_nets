"""
Implements the "parity" task from Graves 2016: determining the parity of a
statically-presented binary vector.
"""

from typing import Tuple

import torch

ParityExample = Tuple[torch.Tensor, torch.Tensor]


# pylint: disable=too-few-public-methods
# pylint: disable=abstract-method
class BinaryParityDataset(torch.utils.data.IterableDataset):  # type: ignore
    """
    A torch IterableDataset for binary parity problems.
    Yields examples as "unary sequences" of shape (1, `size`).

    Parameters
    ----------
    size: int (default: 64)
        The length of each binary vector.
    difficulty: float (default: 0.5)
        The average fraction of each binary vector that is nonzero.
        Must be in the range [0, 1].
    """

    size: int
    difficulty: float

    def __init__(self, size: int = 64, difficulty: float = 0.5):
        if size <= 0:
            raise ValueError("size must be at least one.")

        if not 0 <= difficulty <= 1:
            raise ValueError("difficulty must be in range [0, 1]")

        self.difficulty = difficulty
        self.size = size

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self) -> ParityExample:
        vec = torch.randint(2, (1, self.size,), dtype=torch.int8) * 2 - 1
        mask = torch.rand_like(vec, dtype=float) < self.difficulty  # type: ignore
        feature = vec * mask
        parity = ((feature == 1).sum() % 2).to(dtype=bool)
        return feature, parity
