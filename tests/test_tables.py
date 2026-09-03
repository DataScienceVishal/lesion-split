"""The README generator, and the guard that stops a table quietly freezing.

The failure this exists to prevent is specific. If the scan looks only for the
names the generators offer, then deleting a generator while leaving its markers
in the file is invisible: the comparison never sees the block, reports clean, and
the numbers between those markers sit there looking maintained while describing a
run that no longer exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lesionsplit import tables

RUN = {
    "split": "by_image",
    "seed": 0,
    "train_images": 8012,
    "test_images": 2003,
    "leaked_rows": 768,
    "leaked_fraction": 0.383,
    "epochs": 10,
    "seconds": 88.0,
    "scores": {"accuracy": 0.64, "balanced_accuracy": 0.61, "macro_f1": 0.48},
    "per_class_recall": {"nv": 0.7},
    "baselines": {
        "majority_class": {"accuracy": 0.66, "balanced_accuracy": 0.14, "macro_f1": 0.11},
        "stratified_chance": {"accuracy": 0.48, "balanced_accuracy": 0.14, "macro_f1": 0.14},
    },
}


def results_dir(tmp_path: Path) -> Path:
    other = dict(RUN, split="by_lesion", leaked_rows=0, leaked_fraction=0.0)
    other["scores"] = {"accuracy": 0.62, "balanced_accuracy": 0.60, "macro_f1": 0.47}
    (tmp_path / "experiment.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "torch": "2.14.0",
                    "threads": 4,
                    "platform": "macOS",
                    "epochs": 10,
                    "image_side": 64,
                    "wall_seconds": 600.0,
                },
                "parameters": 390_695,
                "runs": [RUN, other],
            }
        )
    )
    (tmp_path / "leakage.json").write_text(
        json.dumps(
            {
                "images": 10015,
                "lesions": 7470,
                "repeated_lesions": 1956,
                "images_in_repeated_lesions": 4501,
                "most_photographs_of_one_lesion": 6,
                "class_counts": {"nv": 6705},
                "seeds": 20,
                "by_split": {
                    "by_image": {
                        "mean_leaked_fraction": 0.383,
                        "min_leaked_fraction": 0.365,
                        "max_leaked_fraction": 0.400,
                        "mean_train_images": 8012,
                        "mean_test_images": 2003,
                    },
                    "by_lesion": {
                        "mean_leaked_fraction": 0.0,
                        "min_leaked_fraction": 0.0,
                        "max_leaked_fraction": 0.0,
                        "mean_train_images": 8011,
                        "mean_test_images": 2004,
                    },
                },
            }
        )
    )
    return tmp_path


def readme(tmp_path: Path, names: list[str]) -> Path:
    body = ["# t", ""]
    for name in names:
        body += [f"<!-- lesion-split:{name} -->", "stale", f"<!-- /lesion-split:{name} -->", ""]
    target = tmp_path / "README.md"
    target.write_text("\n".join(body))
    return target


def test_every_block_renders_and_the_stale_body_goes(tmp_path):
    results = results_dir(tmp_path)
    page = readme(tmp_path, list(tables.BLOCKS))
    out = tables.rendered(page, results)
    assert "stale" not in out
    assert "10,015" in out and "38.3%" in out


def test_a_marker_with_no_generator_is_an_error(tmp_path):
    results = results_dir(tmp_path)
    page = readme(tmp_path, [*tables.BLOCKS, "invented"])
    with pytest.raises(tables.MarkerError) as raised:
        tables.rendered(page, results)
    assert "invented" in str(raised.value)


def test_a_generator_with_no_marker_is_an_error(tmp_path):
    """The half that a name-driven scan misses, which is the whole point."""
    results = results_dir(tmp_path)
    page = readme(tmp_path, [name for name in tables.BLOCKS if name != "gap"])
    with pytest.raises(tables.MarkerError) as raised:
        tables.rendered(page, results)
    assert "gap" in str(raised.value)


def test_an_unclosed_marker_is_an_error(tmp_path):
    with pytest.raises(tables.MarkerError):
        tables.blocks_in("<!-- lesion-split:corpus -->\nbody\n")


def test_a_close_without_an_open_is_an_error(tmp_path):
    with pytest.raises(tables.MarkerError):
        tables.blocks_in("<!-- /lesion-split:corpus -->\n")


def test_nested_markers_are_an_error(tmp_path):
    with pytest.raises(tables.MarkerError):
        tables.blocks_in("<!-- lesion-split:corpus -->\n<!-- lesion-split:gap -->\n")


def test_the_gap_is_computed_paired_by_seed_not_as_a_difference_of_means(tmp_path):
    """Two arms at one seed differ by 2 points here, and the table must say 2.

    Averaging each arm and subtracting gives the same answer only when every
    seed has both arms. Pairing is what makes the number a within-seed
    comparison, which is the comparison the experiment was designed to make.
    """
    experiment, leakage = tables.load(results_dir(tmp_path))
    rows = tables.gap_table(experiment, leakage)
    assert any("+2.00 points" in row for row in rows)
    assert any("1 seeds" in row for row in rows)


def test_one_seed_says_so_rather_than_printing_nan(tmp_path):
    """A single seed has no spread, and nan in a published table reads as a bug."""
    experiment, leakage = tables.load(results_dir(tmp_path))
    rows = tables.gap_table(experiment, leakage)
    assert not any("nan" in row for row in rows)
    assert any("needs two seeds" in row for row in rows)
