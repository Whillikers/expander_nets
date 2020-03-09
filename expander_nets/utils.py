"""
Project-wide utilities.
"""

import datetime
import os
from typing import List, Optional, Tuple

import torch
from torch.utils import tensorboard

# All examples are (x, y) pairs as Torch tensors
Example = Tuple[torch.Tensor, torch.Tensor]

PAD_VALUE = -1
LOG_DIR = "./logs"
SUMMARY_PERIOD = 1000  # Steps between writing summaries


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
        features, padding_value=PAD_VALUE
    )
    targets_padded = torch.nn.utils.rnn.pad_sequence(
        targets, padding_value=PAD_VALUE
    )
    return features_padded, targets_padded


def get_summary_writer(
    task_name: str, run_name: Optional[str]
) -> tensorboard.SummaryWriter:
    """
    Get the SummaryWriter for a given task.

    Parameters
    ----------
    task_name: str
        Name of the task.
    run_name: Optional[str]
        Name of the run. If not provided, use the current date/time.
    """
    if not run_name:
        run_name = datetime.datetime.now().strftime("%m.%d.%y_%H:%M")

    log_dir = os.path.join(LOG_DIR, task_name, run_name)
    return tensorboard.SummaryWriter(log_dir)
