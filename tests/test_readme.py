"""Figures written into the README prose, held against the artifacts.

Five tables here are generated and cannot drift. The prose around them is
hand-written, and every number in it is one somebody could fail to update after
rerunning the experiment. That is not hypothetical: a sibling project published
seven figures that were correct when typed and wrong by the time anyone checked.

So each figure that also exists in an artifact gets asserted against it. A number
that appears only in prose and nowhere else is listed at the bottom, because an
unguarded figure the reader cannot check is worth knowing about.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "results"

pytestmark = pytest.mark.skipif(
    not (RESULTS / "experiment.json").exists(),
    reason="no results/experiment.json. Run `lesion-split train` to include these",
)


def prose() -> str:
    """The README with every generated block removed."""
    return re.sub(
        r"<!-- lesion-split:[a-z-]+ -->.*?<!-- /lesion-split:[a-z-]+ -->",
        "",
        README.read_text(),
        flags=re.S,
    )


def artifacts() -> tuple[dict, dict]:
    return (
        json.loads((RESULTS / "experiment.json").read_text()),
        json.loads((RESULTS / "leakage.json").read_text()),
    )


def test_the_accuracy_gap_quoted_in_prose_is_the_measured_one():
    experiment, _ = artifacts()
    runs = experiment["runs"]
    gaps = []
    for seed in sorted({r["seed"] for r in runs}):
        arm = {r["split"]: r["scores"]["accuracy"] for r in runs if r["seed"] == seed}
        gaps.append(arm["by_image"] - arm["by_lesion"])
    measured = f"{np.mean(gaps) * 100:.2f}"

    assert measured in prose(), f"prose does not quote the measured accuracy gap of {measured}"


def test_the_parameter_count_in_prose_matches_the_network():
    from lesionsplit.model import SmallCNN

    assert f"{SmallCNN().parameter_count:,}" in prose()


def test_the_lesion_count_in_prose_matches_the_corpus():
    _, leakage = artifacts()
    assert f"{leakage['lesions']:,}" in prose()


def test_the_leakage_percentage_in_prose_matches_the_measurement():
    _, leakage = artifacts()
    measured = leakage["by_split"]["by_image"]["mean_leaked_fraction"] * 100
    assert str(round(measured)) in prose(), f"prose does not quote {measured:.1f}% rounded"


def test_the_majority_floor_in_prose_matches_the_baseline():
    experiment, _ = artifacts()
    floor = np.mean([r["baselines"]["majority_class"]["accuracy"] for r in experiment["runs"]])
    assert f"{floor:.2f}" in prose(), f"prose does not quote the {floor:.3f} majority floor"


def test_the_readme_does_not_claim_a_metric_excludes_zero_when_it_does_not():
    """The one claim that would be dishonest rather than merely stale.

    Two of the three metrics have intervals crossing zero. If a rerun moved them
    and the prose still said so, or stopped saying so when it should, the README
    would be overstating what the experiment shows.
    """
    experiment, _ = artifacts()
    runs = experiment["runs"]
    crossing = []
    for metric in ("balanced_accuracy", "macro_f1"):
        gaps = []
        for seed in sorted({r["seed"] for r in runs}):
            arm = {r["split"]: r["scores"][metric] for r in runs if r["seed"] == seed}
            gaps.append(arm["by_image"] - arm["by_lesion"])
        gaps = np.array(gaps) * 100
        half = 2.365 * gaps.std(ddof=1) / np.sqrt(len(gaps))
        if gaps.mean() - half < 0 < gaps.mean() + half:
            crossing.append(metric)

    text = prose()
    if crossing:
        assert "crosses zero" in text or "cannot separate" in text or "nothing this" in text, (
            f"{crossing} cross zero and the prose does not say so"
        )


def test_no_generated_figure_is_also_typed_into_the_prose():
    """A number in two places is a number that can disagree with itself.

    The tables carry the full precision figures. Prose is allowed to round, and
    is allowed to name a figure the tests above check, but repeating a table
    cell verbatim is how the two drift apart.
    """
    experiment, _ = artifacts()
    text = prose()
    duplicated = [
        f"{r['scores']['balanced_accuracy']:.3f}"
        for r in experiment["runs"]
        if f"{r['scores']['balanced_accuracy']:.3f}" in text
    ]
    assert not duplicated, f"per-run figures repeated in prose: {sorted(set(duplicated))}"
