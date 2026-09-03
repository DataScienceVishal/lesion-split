<!--
Skeleton, not a form to fill in. Delete this comment and any heading that does
not earn its place. Notes on each section are in _factory/templates/README-notes.md.
Rules that get a README rejected: badge walls, a features list padded past what
the thing does, "Contributions are welcome", and any sentence that would survive
being pasted into a different project.
-->

# lesion-split

A skin lesion classifier scored two ways: splitting by image, and splitting by lesion

{{Two or three sentences on why it exists. The problem, and what was wrong with
the obvious approach. Name the real constraint that shaped it.}}

```
{{Terminal output or a short screenshot. Put it here, before the prose, if it
makes the thing obvious faster than a paragraph would.}}
```

## Running it

```bash
git clone https://github.com/DataScienceVishal/lesion-split.git
cd lesion-split
uv sync --all-extras
./scripts/install-hooks.sh
```

{{Every command a stranger needs, in order, with nothing assumed. `adversary`
follows this literally from a fresh clone and reports anything that fails.}}

The test suite runs with no credentials and no network:

```bash
uv run pytest
```

{{If the project can also run against Azure, say what to put in .env and what
changes. If it cannot, say that instead.}}

## How it works

{{The comprehension brief. Enough that Vishal can defend every file in an
interview. Name the two or three modules that carry the idea and what each one
decides. Diagrams only if a diagram is faster than the sentence.}}

## Results

{{Real numbers from the eval harness, including the failure cases. A results
section with only good news is the tell that the eval is grading its own
homework. State the dataset, the split, and the metric, and say what the number
would be for a trivial baseline.}}

## What this does not do

{{Scope that was cut and why. This section is not modesty, it is the thing that
separates an engineer who made choices from one who ran out of time.}}

## Data

{{Source, licence, and how it was cleaned. Every cleaning decision that changed
a number gets a line. If the data is synthetic, say so here and say why
synthetic is the point.}}
