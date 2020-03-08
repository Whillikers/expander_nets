"""
Utilities for tasks and data loading.
"""

from typing import List, Tuple

import torch

# All examples are (x, y) pairs as Torch tensors
Example = Tuple[torch.Tensor, torch.Tensor]

PAD_VALUE = -1


def collate_sequences(batch: List[Example]) -> Example:
    """
    Collating function for batching variable-length sequences, padding each
    sequence to the maximum length.
    NOTE: inputs must not be boolean, and must not use -1 as a value.

    Parameters
    ----------
    """
    features, targets = zip(*batch)
    features_padded = torch.nn.utils.rnn.pad_sequence(
        features, padding_value=PAD_VALUE, batch_first=True
    )
    targets_padded = torch.nn.utils.rnn.pad_sequence(
        targets, padding_value=PAD_VALUE, batch_first=True
    )
    return features_padded, targets_padded
