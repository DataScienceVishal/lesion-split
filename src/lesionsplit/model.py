"""A small convolutional network, and the two baselines it has to beat.

The architecture is deliberately unremarkable: four convolutional blocks with a strided
first layer, batch norm, global average pooling, one linear layer, 390,695
parameters. Nothing here is
a contribution and the README says so. The contribution is the pair of numbers it
produces under two different splits, and a network anyone can reproduce in under two
minutes on a laptop makes that comparison easier to check rather than harder.

Transfer learning from ImageNet would score higher. It would also make the
headline harder to trust, because a pretrained backbone has seen a hundred times
more data than this corpus holds and the question here is about the corpus.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from lesionsplit.data import CLASSES


class SmallCNN(nn.Module):
    """Four blocks, doubling width, global pool, linear. No skip connections.

    Global average pooling rather than a flatten keeps the parameter count in
    the convolutions instead of in one enormous dense layer, which is what lets
    this train on a CPU. The dropout sits before the classifier because that is
    where overfitting shows up first on 8,000 images.
    """

    def __init__(self, classes: int = len(CLASSES), width: int = 32, dropout: float = 0.3):
        super().__init__()
        blocks: list[nn.Module] = []
        channels = 3
        for depth in range(4):
            out = width * (2**depth)
            # The first block strides. Measured on this laptop, running block one
            # at the full 64 by 64 costs 456 ms per step against 155 ms with the
            # stride, and the six-run experiment goes from 34 minutes to 12. The
            # input stays at 64 so the decoder still sees that detail; only the
            # first convolution skips half of it.
            blocks += [
                nn.Conv2d(
                    channels, out, kernel_size=3, stride=2 if depth == 0 else 1,
                    padding=1, bias=False,
                ),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            channels = out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, classes))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(batch)).flatten(1))

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def majority_class(train_labels: np.ndarray, test_labels: np.ndarray) -> dict[str, float]:
    """Answer the commonest training class every time.

    This is the number any accuracy figure has to clear before it means
    anything. On HAM10000 it is 66.9%, which is high enough that a network
    reporting 70% accuracy has learned almost nothing and a reader who sees only
    accuracy cannot tell.
    """
    guess = int(np.bincount(train_labels, minlength=len(CLASSES)).argmax())
    predictions = np.full_like(test_labels, guess)
    return scores(test_labels, predictions)


def stratified_chance(
    train_labels: np.ndarray, test_labels: np.ndarray, seed: int
) -> dict[str, float]:
    """Sample predictions from the training class distribution, ignoring the image.

    The second floor, and the more informative one for balanced accuracy: a
    model that guesses in proportion gets balanced accuracy near 1/7 rather than
    the near-zero that the majority baseline scores.
    """
    counts = np.bincount(train_labels, minlength=len(CLASSES)).astype(np.float64)
    rng = np.random.default_rng(seed)
    predictions = rng.choice(len(CLASSES), size=len(test_labels), p=counts / counts.sum())
    return scores(test_labels, predictions)


def scores(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Accuracy, balanced accuracy, and macro F1, always together.

    Accuracy alone is misleading on a corpus that is 66.9% one class, and
    balanced accuracy alone hides that a model may be trading majority
    performance for it. Reporting one without the other is how a skin lesion
    classifier gets published looking better than it is.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    return {
        "accuracy": float((truth == predicted).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
    }
