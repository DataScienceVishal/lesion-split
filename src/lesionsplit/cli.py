"""`lesion-split` on the command line.

Three commands. `leakage` needs the 690 KB metadata file and answers the question
this project is about. `train` needs the 2.8 GB of images and produces the
accuracy numbers. `report` reads what `train` wrote and renders the tables.

Exit codes: 0 for a clean run, 1 for a download that does not match its recorded
digest, 2 for a command that needed data it could not find. A missing corpus and
a corrupt one are different problems for whoever is reading the output, so they
do not share a code.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from lesionsplit import data, fetch, splits, tables, train

RESULTS = Path("results")


def _provenance(seconds: float, seeds: range, epochs: int) -> dict:
    """Stamped into every artifact, because a number without its machine is thin."""
    return {
        "dataset": f"HAM10000, Harvard Dataverse {fetch.VERSION}, {fetch.DOI}",
        "licence": "CC BY-NC 4.0",
        "citation": "Tschandl, Rosendahl and Kittler 2018, Scientific Data 5:180161",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "image_side": data.SIDE,
        "epochs": epochs,
        "seeds": list(seeds),
        "wall_seconds": round(seconds, 1),
    }


def _leakage(args: argparse.Namespace) -> int:
    frame = fetch.metadata(args.cache)
    counts = splits.repeat_counts(frame)
    table = splits.leakage_over_seeds(frame, range(args.seeds))

    print(f"HAM10000, Dataverse {fetch.VERSION}")
    print(f"  {len(frame):,} images over {frame.lesion_id.nunique():,} lesions")
    print(f"  {(counts > 1).sum():,} lesions appear more than once, up to {counts.max()} times")
    print(f"  {counts[counts > 1].sum():,} images are a repeat look at a lesion already present")
    print()
    print(f"  {args.seeds} seeds, {int(splits.TEST_FRACTION * 100)}% held out")
    print(f"    {'split':<11}{'train':>7}{'test':>7}{'leaked':>9}{'leaked':>9}")
    for name, part in table.groupby("split"):
        print(
            f"    {name:<11}{part.train_images.mean():>7.0f}{part.test_images.mean():>7.0f}"
            f"{part.leaked_rows.mean():>9.0f}{part.leaked_fraction.mean() * 100:>8.1f}%"
        )

    RESULTS.mkdir(exist_ok=True)
    artifact = {
        "images": int(len(frame)),
        "lesions": int(frame.lesion_id.nunique()),
        "repeated_lesions": int((counts > 1).sum()),
        "images_in_repeated_lesions": int(counts[counts > 1].sum()),
        "most_photographs_of_one_lesion": int(counts.max()),
        "class_counts": {k: int(v) for k, v in frame.dx.value_counts().items()},
        "seeds": args.seeds,
        "by_split": {
            name: {
                "mean_leaked_fraction": float(part.leaked_fraction.mean()),
                "min_leaked_fraction": float(part.leaked_fraction.min()),
                "max_leaked_fraction": float(part.leaked_fraction.max()),
                "mean_train_images": float(part.train_images.mean()),
                "mean_test_images": float(part.test_images.mean()),
            }
            for name, part in table.groupby("split")
        },
    }
    (RESULTS / "leakage.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\n  wrote {RESULTS / 'leakage.json'}")
    return 0


def _train(args: argparse.Namespace) -> int:
    folder = args.cache / "images"
    if not folder.is_dir() or not any(folder.glob("*.jpg")):
        print(
            f"lesion-split: no images at {folder}. Run `lesion-split fetch` first, "
            "which downloads 2.8 GB from Dataverse.",
            file=sys.stderr,
        )
        return 2

    frame = fetch.metadata(args.cache)
    started = time.perf_counter()
    print(f"decoding {len(frame):,} images at {data.SIDE}px", flush=True)
    source = data.decode(frame, folder)
    print(f"  decoded in {time.perf_counter() - started:.0f}s", flush=True)

    runs = []
    for seed in range(args.seeds):
        for make in (splits.by_image, splits.by_lesion):
            split = make(frame, seed)
            run = train.train_once(source, split, seed, epochs=args.epochs)
            runs.append(run)
            print(
                f"  seed {seed}  {run.split:<10} "
                f"acc {run.scores['accuracy']:.3f}  "
                f"balanced {run.scores['balanced_accuracy']:.3f}  "
                f"macro F1 {run.scores['macro_f1']:.3f}  "
                f"({run.seconds:.0f}s)",
                flush=True,
            )

    total = time.perf_counter() - started
    RESULTS.mkdir(exist_ok=True)
    artifact = {
        "provenance": _provenance(total, range(args.seeds), args.epochs),
        "parameters": train.models.SmallCNN().parameter_count,
        "runs": [asdict(run) for run in runs],
    }
    (RESULTS / "experiment.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\n  {len(runs)} runs in {total / 60:.1f} min, wrote {RESULTS / 'experiment.json'}")
    return 0


def _report(args: argparse.Namespace) -> int:
    path = RESULTS / "experiment.json"
    if not path.exists():
        print(f"lesion-split: no {path}. Run `lesion-split train` first.", file=sys.stderr)
        return 2

    if args.format == "md":
        page = Path(args.readme)
        fresh = tables.rendered(page, RESULTS)
        if args.check:
            if fresh != page.read_text():
                print(
                    f"lesion-split: {page} is not what the artifacts render. "
                    "Rerun `lesion-split report --format md` and commit what it writes.",
                    file=sys.stderr,
                )
                return 1
            blocks = tables.blocks_in(page.read_text())
            print(f"{page}: {len(blocks)} blocks, all of them what the artifacts render")
            return 0
        page.write_text(fresh)
        print(f"rewrote {len(tables.blocks_in(fresh))} blocks in {page}")
        return 0

    loaded = json.loads(path.read_text())
    runs = loaded["runs"]

    print(f"{'split':<11}{'accuracy':>18}{'balanced':>18}{'macro F1':>18}")
    for name in ("by_image", "by_lesion"):
        arm = [r for r in runs if r["split"] == name]
        cells = []
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = np.array([r["scores"][metric] for r in arm])
            cells.append(f"{values.mean():.3f} +/- {values.std():.3f}")
        print(f"{name:<11}" + "".join(f"{c:>18}" for c in cells))

    print()
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        pairs = []
        for seed in {r["seed"] for r in runs}:
            arm = {r["split"]: r["scores"][metric] for r in runs if r["seed"] == seed}
            if len(arm) == 2:
                pairs.append(arm["by_image"] - arm["by_lesion"])
        gap = np.array(pairs)
        print(
            f"  {metric:<18} image split scores {gap.mean() * 100:+.1f} points higher, "
            f"paired over {len(gap)} seeds, range {gap.min() * 100:+.1f} to {gap.max() * 100:+.1f}"
        )

    floor = np.mean([r["baselines"]["majority_class"]["accuracy"] for r in runs])
    print(f"\n  majority-class accuracy floor: {floor:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lesion-split", description=__doc__)
    parser.add_argument("--cache", type=Path, default=fetch.CACHE)
    sub = parser.add_subparsers(dest="command", required=True)

    leak = sub.add_parser("leakage", help="measure what each split gives away, metadata only")
    leak.add_argument("--seeds", type=int, default=20)
    leak.set_defaults(run=_leakage)

    grab = sub.add_parser("fetch", help="download the corpus, 2.8 GB, verified against Dataverse")
    grab.set_defaults(run=lambda a: (fetch.images(a.cache), 0)[1])

    fit = sub.add_parser("train", help="train both arms at every seed")
    fit.add_argument("--seeds", type=int, default=3)
    fit.add_argument("--epochs", type=int, default=train.EPOCHS)
    fit.set_defaults(run=_train)

    say = sub.add_parser("report", help="render what train wrote")
    say.add_argument("--format", choices=["text", "md"], default="text")
    say.add_argument("--readme", default="README.md")
    say.add_argument("--check", action="store_true", help="compare instead of writing")
    say.set_defaults(run=_report)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except fetch.DigestMismatch as mismatch:
        print(f"lesion-split: {mismatch}", file=sys.stderr)
        return 1
    except tables.MarkerError as broken:
        print(f"lesion-split: {broken}", file=sys.stderr)
        return 1
    except FileNotFoundError as missing:
        print(f"lesion-split: {missing}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
