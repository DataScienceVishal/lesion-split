"""The split functions, against a corpus small enough to check by hand.

No image is read here and none needs to be. Everything this project claims about
leakage is a property of one column in an 830 KB metadata file, which is why the
claim can be checked before a 2.8 GB download and why these tests run offline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lesionsplit import splits


def corpus(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """`pairs` is (lesion_id, image_id), which is all either split function reads."""
    return pd.DataFrame(
        {
            "lesion_id": [lesion for lesion, _ in pairs],
            "image_id": [image for _, image in pairs],
            "dx": ["nv"] * len(pairs),
        }
    )


def repeated(lesions: int, per_lesion: int) -> pd.DataFrame:
    return corpus(
        [(f"L{i:04d}", f"I{i:04d}_{k}") for i in range(lesions) for k in range(per_lesion)]
    )


def test_a_lesion_split_never_puts_one_lesion_on_both_sides():
    frame = repeated(lesions=200, per_lesion=3)
    for seed in range(20):
        split = splits.by_lesion(frame, seed)
        assert split.leaked_rows == 0, f"seed {seed} leaked {split.leaked_rows} rows"


def test_an_image_split_does_leak_when_lesions_repeat():
    """The control for the test above, which would otherwise pass on an empty property."""
    frame = repeated(lesions=200, per_lesion=3)
    leaks = [splits.by_image(frame, seed).leaked_rows for seed in range(20)]
    assert min(leaks) > 0, "no seed leaked, so the test above proves nothing"


def test_neither_split_leaks_when_every_lesion_is_photographed_once():
    """The boundary that decides whether this project has a subject at all.

    On a corpus with no repeats the two arms are the same experiment and the
    headline gap must vanish. If this ever fails, the difference being measured
    is not the one being claimed.
    """
    frame = repeated(lesions=300, per_lesion=1)
    for seed in range(10):
        assert splits.by_image(frame, seed).leaked_rows == 0
        assert splits.by_lesion(frame, seed).leaked_rows == 0


def test_every_row_lands_on_exactly_one_side():
    frame = repeated(lesions=150, per_lesion=2)
    for make in (splits.by_image, splits.by_lesion):
        split = make(frame, 7)
        assert len(split.train) + len(split.test) == len(frame)
        assert set(split.train.image_id).isdisjoint(split.test.image_id)
        assert set(split.train.image_id) | set(split.test.image_id) == set(frame.image_id)


def test_the_same_seed_gives_the_same_split():
    frame = repeated(lesions=120, per_lesion=2)
    for make in (splits.by_image, splits.by_lesion):
        first, second = make(frame, 3), make(frame, 3)
        assert list(first.test.image_id) == list(second.test.image_id)


def test_different_seeds_give_different_splits():
    frame = repeated(lesions=120, per_lesion=2)
    for make in (splits.by_image, splits.by_lesion):
        assert list(make(frame, 1).test.image_id) != list(make(frame, 2).test.image_id)


@pytest.mark.parametrize("per_lesion", [1, 2, 4])
def test_the_image_split_hits_its_target_fraction(per_lesion):
    frame = repeated(lesions=250, per_lesion=per_lesion)
    split = splits.by_image(frame, 0, test_fraction=0.2)
    assert split.leaked_fraction >= 0
    assert abs(len(split.test) / len(frame) - 0.2) < 0.01


def test_the_lesion_split_misses_its_target_fraction_and_that_is_expected():
    """Documented rather than fixed, because the fix would be worse.

    Lesions carry different numbers of images, so holding out a fifth of the
    lesions does not hold out a fifth of the images. Trimming the test set to an
    exact fraction would mean splitting a lesion across the divide, which is the
    thing this function exists to prevent.
    """
    frame = corpus(
        [("L0", "a"), ("L0", "b"), ("L0", "c"), ("L0", "d"), ("L0", "e")]
        + [(f"L{i}", f"i{i}") for i in range(1, 20)]
    )
    sizes = {len(splits.by_lesion(frame, seed).test) for seed in range(30)}
    assert len(sizes) > 1, "test size never varied, so the imbalance is not being exercised"


def test_leakage_table_reports_zero_for_the_grouped_arm_on_every_seed():
    frame = repeated(lesions=180, per_lesion=3)
    table = splits.leakage_over_seeds(frame, range(5))
    grouped = table[table.split == "by_lesion"]
    by_image = table[table.split == "by_image"]
    assert (grouped.leaked_rows == 0).all()
    assert (by_image.leaked_rows > 0).all()
    assert len(table) == 10


def test_repeat_counts_finds_the_most_photographed_lesion():
    frame = corpus([("L0", "a"), ("L0", "b"), ("L0", "c"), ("L1", "d"), ("L2", "e")])
    counts = splits.repeat_counts(frame)
    assert counts.iloc[0] == 3
    assert counts.index[0] == "L0"
    assert counts.sum() == len(frame)
