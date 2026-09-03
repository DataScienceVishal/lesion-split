"""Training one model, and the experiment that trains several.

Everything is held fixed across arms except the split. Same architecture, same
seed, same epochs, same batch size, same optimiser, same augmentation, same class
weights recomputed from whichever training set the arm produced. If two runs
differ, the split is the reason, because nothing else was allowed to vary.

Runs on CPU by design. This is not a claim that CPU is the right way to train
vision models; it is that a result nobody can rerun is a weaker result, and the
whole experiment finishes here in about the time it takes to read the README.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lesionsplit import model as models
from lesionsplit import splits
from lesionsplit.data import CLASSES, Decoded, Lesions, class_weights

EPOCHS = 10
BATCH = 128
LEARNING_RATE = 3e-3


@dataclass
class Run:
    """One arm at one seed, with everything needed to compare it to another."""

    split: str
    seed: int
    train_images: int
    test_images: int
    leaked_rows: int
    leaked_fraction: float
    epochs: int
    seconds: float
    scores: dict[str, float]
    per_class_recall: dict[str, float]
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_record(self) -> dict:
        return asdict(self)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_once(source: Decoded, split: splits.Split, seed: int, epochs: int = EPOCHS) -> Run:
    """Fit the network on one side of `split` and score it on the other."""
    _seed_everything(seed)
    started = time.perf_counter()

    train_rows = source.rows_for(split.train)
    test_rows = source.rows_for(split.test)
    train_labels = source.labels[train_rows]
    test_labels = source.labels[test_rows]

    loader = DataLoader(
        Lesions(source, train_rows, augment=True, seed=seed),
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
    )
    network = models.SmallCNN()
    loss_function = nn.CrossEntropyLoss(weight=class_weights(train_labels))
    optimiser = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=LEARNING_RATE, epochs=epochs, steps_per_epoch=len(loader)
    )

    network.train()
    for _ in range(epochs):
        for batch, labels in loader:
            optimiser.zero_grad()
            loss = loss_function(network(batch), labels)
            loss.backward()
            optimiser.step()
            schedule.step()

    network.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        evaluation = DataLoader(
            Lesions(source, test_rows, augment=False), batch_size=BATCH, shuffle=False
        )
        for batch, _ in evaluation:
            predictions.append(network(batch).argmax(1).numpy())
    predicted = np.concatenate(predictions)

    return Run(
        split=split.name,
        seed=seed,
        train_images=len(split.train),
        test_images=len(split.test),
        leaked_rows=split.leaked_rows,
        leaked_fraction=split.leaked_fraction,
        epochs=epochs,
        seconds=time.perf_counter() - started,
        scores=models.scores(test_labels, predicted),
        per_class_recall=_recall_by_class(test_labels, predicted),
        baselines={
            "majority_class": models.majority_class(train_labels, test_labels),
            "stratified_chance": models.stratified_chance(train_labels, test_labels, seed),
        },
    )


def _recall_by_class(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Recall per class, including the classes the model never emits.

    A macro average hides which class was abandoned. On this corpus the rare
    classes are the ones a reader should care about, so they get named.
    """
    recall = {}
    for index, name in enumerate(CLASSES):
        present = truth == index
        if present.any():
            recall[name] = float((predicted[present] == index).mean())
        else:
            recall[name] = float("nan")
    return recall


def experiment(source: Decoded, frame, seeds: range, epochs: int = EPOCHS) -> list[Run]:
    """Both arms at every seed, in an order that interleaves them.

    Interleaving matters on a laptop. Running all the image-split arms first and
    all the lesion-split arms second would let thermal throttling load onto one
    arm, and the timing column would then say something about the machine rather
    than about the work.
    """
    runs: list[Run] = []
    for seed in seeds:
        for make in (splits.by_image, splits.by_lesion):
            runs.append(train_once(source, make(frame, seed), seed, epochs))
    return runs
