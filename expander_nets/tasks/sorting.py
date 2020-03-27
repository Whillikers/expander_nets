"""
Implements the "sorting" task from Graves 2016: sorting randomly-sized lists
of random numbers.
"""
import os
from typing import Optional

import numpy as np  # type: ignore
import torch
import torch.nn.functional as F
from torch import nn

from expander_nets import models, utils

SEQUENCE_END = 1
TASK_NAME = "sort"


class SortingClassifier(nn.Module):
    def __init__(self, rnn: nn.Module, sequence_length: int):
        super(SortingClassifier, self).__init__()
        self.rnn = rnn
        self.sequence_length = sequence_length
        self.linear = nn.Linear(rnn.hidden_size, self.sequence_length)  # type: ignore

    # pylint: disable=arguments-differ
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore
        encoded, _ = self.rnn(inputs)
        logits = self.linear(encoded)  # Shape: [seq length, batch, seq length]
        return logits


def train(
    rnn: nn.Module,
    batch_size: int,
    learning_rate: float,
    sequence_length: int,
    min_separation: float,
    use_gpu: bool = True,
    data_workers: int = 4,
    run_name: Optional[str] = None,
):

    summary_writer = utils.get_summary_writer(TASK_NAME, run_name)
    use_gpu = use_gpu and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    dataset = SortingDataset(sequence_length, min_separation=min_separation)
    data_loader = torch.utils.data.DataLoader(  # type: ignore
        dataset,
        batch_size=batch_size,
        pin_memory=use_gpu,
        num_workers=data_workers,
        collate_fn=utils.collate_sequences,
    )

    #  ckpt = torch.load("./logs/sort/15/expander_fixed_lr/ckpt.tar")
    #  step = ckpt["step"]

    step = 0
    model = SortingClassifier(rnn, sequence_length).train().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 1e-3, 1e-4, cycle_momentum=False, mode="triangular"
    )

    #  model.load_state_dict(ckpt["model_state_dict"])
    #  optimizer.load_state_dict(ckpt["optimizer_state_dict"])

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
            f"{avg_sequence_accuracy=:.3f} "
            f"lr={optimizer.param_groups[0]['lr']}"
        )

    for step_, (features, targets) in enumerate(data_loader):
        step += 1
        if use_gpu:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(features)
        log_prob = F.log_softmax(logits, dim=-1)

        loss = F.nll_loss(
            log_prob.view(-1, sequence_length),
            targets.view(-1),
            ignore_index=utils.MASK_VALUE,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()  # type: ignore

        running_loss += loss.item()
        matches = (logits.detach().argmax(-1) == targets)[sequence_length:]
        running_correct += matches.sum().item()
        running_sequence_correct += matches.all(0).sum().item()
        step_count += matches.nelement()

        if step and step % utils.SUMMARY_PERIOD == 0:
            summary_step(step)

        # TODO: improve
        if step and step % utils.CHECKPOINT_PERIOD == 0:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                os.path.join(
                    utils.LOG_DIR, TASK_NAME, run_name or "", "ckpt.tar"
                ),
            )


# pylint: disable=too-few-public-methods
# pylint: disable=abstract-method
class SortingDataset(torch.utils.data.IterableDataset):  # type: ignore
    """
    A torch IterableDataset for sorting problems.

    Features are (number, flag) pairs, where flag is 0 for continuing the
    sequence and 1 at and beyond the end of the sequence and numbers
    are generated from the normal distribution. Targets are masked during
    the input sequence and argsort indices otherwise.

    Parameters
    ----------
    sequence_length: int (default: 15)
        How many numbers to sort.
    stdev: float (default: 1.0)
        Standard deviation of the distribution to generate numbers from.
    min_separation: float (default: 0.0)
        Minimum separation between numbers. If 0, any separation is acceptable.
        NOTE: long sequences with large minimum separation may take a long time
        to generate.
    """

    sequence_length: int
    stdev: float
    min_separation: float

    def __init__(
        self,
        sequence_length: int = 15,
        stdev: float = 1.0,
        min_separation: float = 0.0,
    ):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be at least 1.")
        if stdev <= 0.0:
            raise ValueError("stdev must be positive.")
        if min_separation < 0.0:
            raise ValueError("min_separation must be nonnegative.")

        self.sequence_length = sequence_length
        self.stdev = stdev
        self.min_separation = min_separation

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self) -> utils.Example:
        # Sequence is separated into (input, target) sections
        length = 2 * self.sequence_length

        numbers = self._get_numbers(self.sequence_length)

        features = torch.zeros([length, 2], dtype=torch.float32)
        features[: self.sequence_length, 0] = numbers
        features[self.sequence_length - 1 :, 1] = 1

        # pylint: disable=not-callable
        mask_val = torch.tensor(utils.MASK_VALUE, dtype=torch.long)
        targets = torch.repeat_interleave(mask_val, length)
        targets[self.sequence_length :] = numbers.argsort()

        return features, targets

    # NOTE: there may be a more efficient way to do this than repeated sampling?
    def _get_numbers(self, length: int) -> torch.Tensor:
        while True:
            sample = np.random.normal(scale=self.stdev, size=length)
            if _separation(sample) >= self.min_separation:
                return torch.tensor(  # pylint: disable=not-callable
                    sample, dtype=torch.long
                )


def _separation(sample: np.ndarray) -> float:
    sample_sorted = np.sort(sample)
    # Compute pairwise differences by subtracting the sorted array
    # from a shifted copy of itself
    differences = sample_sorted[1:] - sample_sorted[:-1]
    return differences.min()


# TODO: remove
if __name__ == "__main__":
    #  repeat_rnn = models.RepeatRNN(
    #      nn.LSTM, input_size=2, hidden_size=512, repeats=3
    #  )
    expander = models.FixedLSTMExpander(
        input_size=2,
        hidden_size=512,
        translator_dim=512,
        space=3,
        num_layers=3,
        heads=4,
    )
    #  print(sum([p.numel() for p in repeat_rnn.parameters()]))
    #  print(sum([p.numel() for p in expander.parameters()]))
    train(
        expander,
        sequence_length=15,
        min_separation=0,
        batch_size=512,
        learning_rate=3e-4,
        run_name="15/expander_cyclic_again",
    )
