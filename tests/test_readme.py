"""Figures written into the README prose, held against the artifacts.

Four tables and the runtime block are generated, and the first test below fails
if the committed README is not what the artifacts render. The prose around them
is hand-written, and every number in it is one somebody could fail to update
after rerunning the experiment. That is not hypothetical: a sibling project
published seven figures that were correct when typed and wrong by the time
anyone checked.

So the figures the README leans on are each asserted against the artifact or
code that holds them, one test per figure. The rest go unchecked: rounded
restatements such as "two fifths" for the leak, the settings, and the download
sizes.
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


# A phrase can wrap onto the next line of the README, so any run of whitespace matches.
def says(phrase: str) -> bool:
    return re.search(r"\s+".join(map(re.escape, phrase.split())), prose()) is not None


def test_the_committed_readme_is_what_the_artifacts_render():
    from lesionsplit import tables

    assert tables.rendered(README, RESULTS) == README.read_text(), (
        "README blocks are stale. Run `lesion-split report --format md` and commit what it writes."
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


def test_the_time_per_run_in_prose_matches_the_recorded_run():
    experiment, _ = artifacts()
    measured = np.mean([r["seconds"] for r in experiment["runs"]])
    assert says(f"{measured:.0f} seconds"), f"prose does not quote {measured:.0f}s per run"


def test_the_default_three_seeds_fit_in_the_ten_minutes_the_prose_gives_them():
    """The default trains seeds 0 to 2, and the artifact timed those six runs.

    The wall time outside the runs is mostly the decode, which a default run pays
    once as well. The margin is thin, so a slower rerun can fail this.
    """
    experiment, _ = artifacts()
    runs = experiment["runs"]
    outside = experiment["provenance"]["wall_seconds"] - sum(r["seconds"] for r in runs)
    default = outside + sum(r["seconds"] for r in runs if r["seed"] < 3)
    assert says("inside ten minutes"), "prose no longer says the default fits in ten minutes"
    assert default < 600, f"seeds 0 to 2 took {default / 60:.1f} minutes"


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

    if crossing:
        assert says("crosses zero") or says("cannot separate") or says("nothing this"), (
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
