"""The two ways to divide HAM10000, and the measurement that separates them.

HAM10000 photographs some lesions more than once. 10,015 images cover 7,470
distinct lesions, so 2,545 images are a second, third or sixth look at something
already in the set. Splitting on the image column puts those repeats on both
sides of the divide and the model is scored partly on pictures of lesions it
trained on.

Splitting on `lesion_id` instead keeps every photograph of one lesion on one
side. That is the only difference between the two functions below, and the gap
between the accuracies they produce is what this project reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TEST_FRACTION = 0.2


@dataclass(frozen=True)
class Split:
    """One division of the corpus, and how much of the test set it gave away.

    `leaked_rows` counts test images whose lesion also appears in train. It is
    zero by construction for a grouped split, and reporting it anyway is the
    point: a number that is always zero on one arm and never zero on the other
    is the cheapest possible proof that the two arms differ in the way claimed.
    """

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    seed: int

    @property
    def leaked_rows(self) -> int:
        shared = set(self.train.lesion_id) & set(self.test.lesion_id)
        return int(self.test.lesion_id.isin(shared).sum())

    @property
    def leaked_fraction(self) -> float:
        return self.leaked_rows / len(self.test)


def by_image(frame: pd.DataFrame, seed: int, test_fraction: float = TEST_FRACTION) -> Split:
    """Shuffle the rows and cut. What almost every published notebook does.

    Nothing here is wrong as code. It is wrong as an experiment, and only
    because of a property of this particular corpus that the file itself
    announces in its second column.
    """
    order = np.random.default_rng(seed).permutation(len(frame))
    cut = int(round(len(frame) * (1 - test_fraction)))
    return Split(
        name="by_image",
        train=frame.iloc[order[:cut]].reset_index(drop=True),
        test=frame.iloc[order[cut:]].reset_index(drop=True),
        seed=seed,
    )


def by_lesion(frame: pd.DataFrame, seed: int, test_fraction: float = TEST_FRACTION) -> Split:
    """Shuffle the lesions and cut, then take whatever images come with them.

    The test fraction is approximate here and cannot be otherwise. Lesions carry
    between one and six images, so choosing 20 percent of lesions does not
    choose 20 percent of images. The alternative, trimming the test set back to
    an exact fraction, would mean dropping images from lesions already assigned,
    and a lesion split that splits lesions is not a lesion split.
    """
    lesions = frame.lesion_id.unique()
    order = np.random.default_rng(seed).permutation(len(lesions))
    cut = int(round(len(lesions) * (1 - test_fraction)))
    train_ids = set(lesions[order[:cut]])
    held = frame.lesion_id.isin(train_ids)
    return Split(
        name="by_lesion",
        train=frame[held].reset_index(drop=True),
        test=frame[~held].reset_index(drop=True),
        seed=seed,
    )


def repeat_counts(frame: pd.DataFrame) -> pd.Series:
    """Images per lesion, descending. The distribution the whole project rests on."""
    return frame.groupby("lesion_id").size().sort_values(ascending=False)


def leakage_over_seeds(frame: pd.DataFrame, seeds: range) -> pd.DataFrame:
    """Both splits over the same seeds, so the zero column is earned rather than asserted.

    Running the grouped arm here looks redundant, since `by_lesion` cannot leak
    and the column is zero every time. That is exactly why it runs: if a future
    edit breaks the grouping, this table stops being zero and says so, whereas a
    test that only checks the image arm would keep passing.
    """
    rows = []
    for seed in seeds:
        for split in (by_image(frame, seed), by_lesion(frame, seed)):
            rows.append(
                {
                    "split": split.name,
                    "seed": seed,
                    "train_images": len(split.train),
                    "test_images": len(split.test),
                    "leaked_rows": split.leaked_rows,
                    "leaked_fraction": split.leaked_fraction,
                }
            )
    return pd.DataFrame(rows)
