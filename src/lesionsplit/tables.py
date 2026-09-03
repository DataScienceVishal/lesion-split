"""README tables, rendered from the committed artifacts.

Every figure in the README lives between a pair of HTML comments and is written
by one of the generators below. `report --check` re-renders and compares, so a
number that has been edited by hand, or one that has drifted because the
experiment was rerun and the prose was not, fails rather than sits there.

A marker with no generator and a generator with no marker are both errors. That
matters more than it sounds: checking only one direction lets a block quietly
freeze at whatever it last said while everything still reports clean.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

MARKER = re.compile(r"^<!--\s*(/?)lesion-split:([a-z0-9-]+)\s*-->$")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
PRETTY = {"accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "macro_f1": "macro F1"}


class MarkerError(RuntimeError):
    pass


def load(results: Path) -> tuple[dict, dict]:
    experiment = json.loads((results / "experiment.json").read_text())
    leakage = json.loads((results / "leakage.json").read_text())
    return experiment, leakage


def _arm(runs: list[dict], name: str, metric: str) -> np.ndarray:
    return np.array([r["scores"][metric] for r in runs if r["split"] == name])


def _paired_gaps(runs: list[dict], metric: str) -> np.ndarray:
    gaps = []
    for seed in sorted({r["seed"] for r in runs}):
        arm = {r["split"]: r["scores"][metric] for r in runs if r["seed"] == seed}
        if len(arm) == 2:
            gaps.append(arm["by_image"] - arm["by_lesion"])
    return np.array(gaps)


def corpus(experiment: dict, leakage: dict) -> list[str]:
    return [
        f"| images | {leakage['images']:,} |",
        f"| distinct lesions | {leakage['lesions']:,} |",
        f"| lesions photographed more than once | {leakage['repeated_lesions']:,} "
        f"({100 * leakage['repeated_lesions'] / leakage['lesions']:.1f}%) |",
        f"| images that are a repeat of one already present | "
        f"{leakage['images_in_repeated_lesions']:,} "
        f"({100 * leakage['images_in_repeated_lesions'] / leakage['images']:.1f}%) |",
        f"| most photographs of one lesion | {leakage['most_photographs_of_one_lesion']} |",
        f"| largest class | nv, {100 * leakage['class_counts']['nv'] / leakage['images']:.1f}% |",
    ]


def leakage_table(experiment: dict, leakage: dict) -> list[str]:
    rows = ["| split | train | test | test images leaked |", "|---|---|---|---|"]
    for name in ("by_image", "by_lesion"):
        part = leakage["by_split"][name]
        rows.append(
            f"| `{name}` | {part['mean_train_images']:,.0f} | {part['mean_test_images']:,.0f} | "
            f"{100 * part['mean_leaked_fraction']:.1f}% "
            f"({100 * part['min_leaked_fraction']:.1f} to "
            f"{100 * part['max_leaked_fraction']:.1f}) |"
        )
    rows.append(f"\nOver {leakage['seeds']} seeds, 20 percent held out.")
    return rows


def accuracy_table(experiment: dict, leakage: dict) -> list[str]:
    runs = experiment["runs"]
    seeds = len({r["seed"] for r in runs})
    rows = ["| split | " + " | ".join(PRETTY[m] for m in METRICS) + " |", "|---|---|---|---|"]
    for name in ("by_image", "by_lesion"):
        cells = []
        for metric in METRICS:
            values = _arm(runs, name, metric)
            cells.append(f"{values.mean():.3f} +/- {values.std():.3f}")
        rows.append(f"| `{name}` | " + " | ".join(cells) + " |")
    floor = np.mean([r["baselines"]["majority_class"]["accuracy"] for r in runs])
    chance = np.mean([r["baselines"]["stratified_chance"]["balanced_accuracy"] for r in runs])
    rows += [
        f"| always answer `nv` | {floor:.3f} | {1 / 7:.3f} | 0.114 |",
        f"| guess in proportion | "
        f"{np.mean([r['baselines']['stratified_chance']['accuracy'] for r in runs]):.3f} | "
        f"{chance:.3f} | "
        f"{np.mean([r['baselines']['stratified_chance']['macro_f1'] for r in runs]):.3f} |",
        f"\nMean and standard deviation over {seeds} seeds, "
        f"{experiment['provenance']['epochs']} epochs, "
        f"{experiment['provenance']['image_side']} pixel input, "
        f"{experiment['parameters']:,} parameters.",
    ]
    return rows


def gap_table(experiment: dict, leakage: dict) -> list[str]:
    """The headline, with an interval, because a paired mean alone is a guess.

    The interval is a t interval over the per-seed paired differences. It is the
    difference between "leakage inflates this metric" and "leakage inflated this
    metric in the runs I happened to do", and on two of the three metrics here
    it is the difference between a claim and no claim.
    """
    runs = experiment["runs"]
    rows = [
        "| metric | image split minus lesion split | 95% interval | seeds above zero |",
        "|---|---|---|---|",
    ]
    # Two-sided t, seeds - 1 degrees of freedom. Enough of the table to avoid a
    # scipy dependency for one number, and the seed count is fixed by the run.
    critical = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
    for metric in METRICS:
        gaps = _paired_gaps(runs, metric) * 100
        if len(gaps) < 2:
            # One seed has no spread to estimate, and printing nan would look
            # like a broken interval rather than an absent one.
            interval = "needs two seeds or more"
        else:
            half = critical.get(len(gaps) - 1, 1.96) * gaps.std(ddof=1) / np.sqrt(len(gaps))
            low, high = gaps.mean() - half, gaps.mean() + half
            interval = f"{low:+.2f} to {high:+.2f}"
            if low < 0 < high:
                interval += ", crosses zero"
        rows.append(
            f"| {PRETTY[metric]} | {gaps.mean():+.2f} points | {interval} | "
            f"{(gaps > 0).sum()} of {len(gaps)} |"
        )
    rows.append(f"\nPaired within seed, {len(_paired_gaps(runs, 'accuracy'))} seeds.")
    return rows


def runtime(experiment: dict, leakage: dict) -> list[str]:
    provenance = experiment["provenance"]
    runs = experiment["runs"]
    return [
        "```",
        f"{len(runs)} runs in {provenance['wall_seconds'] / 60:.1f} minutes",
        f"{np.mean([r['seconds'] for r in runs]):.0f}s per run, "
        f"{provenance['epochs']} epochs, batch 128",
        f"torch {provenance['torch']}, {provenance['threads']} threads, CPU",
        f"{provenance['platform']}",
        "```",
    ]


BLOCKS: dict[str, Callable[[dict, dict], list[str]]] = {
    "corpus": corpus,
    "leakage": leakage_table,
    "accuracy": accuracy_table,
    "gap": gap_table,
    "runtime": runtime,
}


def blocks_in(markdown: str) -> dict[str, tuple[int, int]]:
    """Every marker pair, whatever it is called, so an orphan is visible."""
    found: dict[str, tuple[int, int]] = {}
    opened: tuple[str, int] | None = None
    for number, line in enumerate(markdown.splitlines()):
        match = MARKER.match(line.strip())
        if not match:
            continue
        closing, name = match.group(1), match.group(2)
        if closing:
            if opened is None or opened[0] != name:
                raise MarkerError(f"line {number + 1} closes {name}, which is not open")
            found[name] = (opened[1], number)
            opened = None
        else:
            if opened is not None:
                raise MarkerError(f"line {number + 1} opens {name} inside {opened[0]}")
            opened = (name, number)
    if opened is not None:
        raise MarkerError(f"{opened[0]} is opened and never closed")
    return found


def rendered(readme: Path, results: Path) -> str:
    experiment, leakage = load(results)
    lines = readme.read_text().splitlines()
    found = blocks_in("\n".join(lines))

    orphan_markers = sorted(set(found) - set(BLOCKS))
    orphan_generators = sorted(set(BLOCKS) - set(found))
    if orphan_markers or orphan_generators:
        raise MarkerError(
            "the README and the generators do not describe the same blocks. "
            + (f"Nothing generates: {', '.join(orphan_markers)}. " if orphan_markers else "")
            + (f"Nowhere to put: {', '.join(orphan_generators)}. " if orphan_generators else "")
            + "Add the marker pair, or delete the generator."
        )

    for name, (start, end) in sorted(found.items(), key=lambda item: -item[1][0]):
        lines[start + 1 : end] = BLOCKS[name](experiment, leakage)
    return "\n".join(lines) + "\n"
