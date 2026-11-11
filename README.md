# lesion-split

[![ci](https://github.com/DataScienceVishal/lesion-split/actions/workflows/ci.yml/badge.svg)](https://github.com/DataScienceVishal/lesion-split/actions/workflows/ci.yml)

HAM10000 photographs some skin lesions more than once. Split the dataset by
image and roughly two fifths of your test set turns out to be repeat photographs
of lesions the model trained on.

This trains the same small CNN twice, changing only how the data is divided, and
reports what that costs.

<!-- lesion-split:leakage -->
| split | train | test | test images leaked |
|---|---|---|---|
| `by_image` | 8,012 | 2,003 | 38.3% (36.5 to 40.0) |
| `by_lesion` | 8,011 | 2,004 | 0.0% (0.0 to 0.0) |

Means over 20 seeds, with the range across them in brackets, 20 percent held out.
<!-- /lesion-split:leakage -->

On this model, at this resolution, the correction costs about one and a
half points of accuracy, and on both class-balanced metrics the interval
crosses zero.

<!-- lesion-split:gap -->
| metric | image split minus lesion split | 95% interval | seeds above zero |
|---|---|---|---|
| accuracy | +1.54 points | +0.44 to +2.64 | 7 of 8 |
| balanced accuracy | +1.94 points | -0.84 to +4.73, crosses zero | 5 of 8 |
| macro F1 | +1.31 points | -0.97 to +3.59, crosses zero | 5 of 8 |

Paired within seed, 8 seeds.
<!-- /lesion-split:gap -->

That is the result, and it is not the one I expected. A flaw that puts 38
percent of the test set on the wrong side of the divide buys a point and a half
of accuracy, and on the two metrics that account for class imbalance it buys
nothing this experiment can distinguish from noise.

The leakage is real and easy to show.

```bash
git clone https://github.com/DataScienceVishal/lesion-split && cd lesion-split
uv sync --all-extras
uv run lesion-split leakage          # the table above, from a 690 KB download
```

Needs [uv](https://docs.astral.sh/uv/). The leakage measurement needs no images.
Training needs 2.8 GB, fetched on demand and never committed.

<details>
<summary><b>Why splitting by image is the wrong experiment here</b></summary>

The corpus ships a `lesion_id` column, and it is not decorative.

<!-- lesion-split:corpus -->
| images | 10,015 |
| distinct lesions | 7,470 |
| lesions photographed more than once | 1,956 (26.2%) |
| images of lesions photographed more than once | 4,501 (44.9%) |
| most photographs of one lesion | 6 |
| largest class | nv, 66.9% |
<!-- /lesion-split:corpus -->

A lesion photographed six times contributes six rows. Shuffle rows and hold out
a fifth, and every one of those images has five chances to have a near-duplicate
sitting in the training set. Grouping by `lesion_id` before splitting keeps all
six on one side.

The two arms train on almost the same number of images, which a fair comparison
needs. Holding out a fifth of the lesions does not hold out a fifth of the
images, so the lesion split's training set moves with the seed. Across the eight
seeds trained it ran from 6 images larger than the image split's to 43 smaller,
about half a percent at the worst.

</details>

<details>
<summary><b>What the model scores, against the two floors it has to clear</b></summary>

<!-- lesion-split:accuracy -->
| split | accuracy | balanced accuracy | macro F1 |
|---|---|---|---|
| `by_image` | 0.641 +/- 0.006 | 0.626 +/- 0.015 | 0.487 +/- 0.013 |
| `by_lesion` | 0.625 +/- 0.011 | 0.606 +/- 0.026 | 0.474 +/- 0.021 |
| always answer `nv` | 0.667 | 0.143 | 0.114 |
| guess in proportion | 0.475 | 0.140 | 0.140 |

Mean and standard deviation over 8 seeds, 10 epochs, 64 pixel input, 390,695 parameters.
<!-- /lesion-split:accuracy -->

Accuracy is the least useful column and it is included because leaving it out
would be worse. Two thirds of this corpus is one class, so answering `nv` every
time scores about 0.67, and the network scores below that. It is not that the
network is worse than a constant: the loss is weighted by inverse class
frequency, which deliberately trades majority accuracy for the rare classes.
Balanced accuracy is the column to read, and at about 0.6 it is four times a
chance floor near 0.14.

</details>

<details>
<summary><b>Why the effect might be this small</b></summary>

This measures 1.54 on accuracy and cannot separate the class-balanced metrics
from zero. Three explanations, none of which this evidence distinguishes
between:

The model may be too small to memorise individual lesions. 390,695 parameters at
64 by 64 has limited capacity to store a specific image, and leakage only pays
if the network can recognise the duplicate. A fine-tuned ImageNet backbone at
224 pixels has far more room to do exactly that.

The resolution may be destroying the cue. Two photographs of one lesion differ
in framing and lighting; at 64 pixels much of what makes them recognisably the
same object is gone.

Ten epochs may not be enough to overfit hard enough for it to matter.

Each one predicts the gap grows along its own axis: with a larger network at 64
pixels, with this network at 224, or with more epochs. Changing one at a time
would test each, though if 64 pixels destroys the cue, the two tests that stay
at 64 pixels will find nothing either way. This repository runs none of them.

The resolution test is the expensive one. A run here averages 85 seconds at 64
pixels. At 224 every image has twelve times the pixels, so if time scales with
them the eight-seed comparison, sixteen runs, takes hours on a CPU. Twice the
epochs roughly doubles a run, and the same network at twice the width does
about four times the arithmetic, roughly an hour and a half for the eight-seed
comparison.

</details>

<details>
<summary><b>Running it, and what each command needs</b></summary>

```bash
uv run lesion-split leakage    # metadata only, 690 KB, seconds
uv run lesion-split fetch      # 2.8 GB from Harvard Dataverse, MD5 verified
uv run lesion-split train      # both arms at every seed
uv run lesion-split report     # render what train wrote
```

Exit codes: 0 clean, 1 a download that does not match its recorded digest, 2 a
command that needed data it could not find, or a usage error. A missing corpus
and a corrupt one are different problems and do not share a code. Anything
uncaught, a network error for instance, ends in a traceback and exit 1.

`report --format md` rewrites the tables in this file, and the timing summary
below, which sit between pairs of HTML comments that GitHub does not display. It
exits 1 if those comments are unbalanced or nested, or do not match the
generators in `tables.py` one for one. With `--check` it writes nothing, and
exits 1 if the file is not what the artifacts render.

<!-- lesion-split:runtime -->
```
16 runs in 23.6 minutes
85s per run, 10 epochs, batch 128
torch 2.14.0, 4 threads, CPU
macOS-26.6.2-arm64-arm-64bit
```
<!-- /lesion-split:runtime -->

The block above records the eight-seed experiment that every score in this
README comes from, and `uv run lesion-split train --seeds 8` reruns it. The
default is three seeds, six runs, which at the recorded pace finishes inside ten
minutes. Either one overwrites `results/experiment.json`.

</details>

<details>
<summary><b>What each file does</b></summary>

`src/lesionsplit/splits.py` is the whole argument: two functions that differ in
one column, and a measurement of what that difference gives away. Start there.

`src/lesionsplit/fetch.py` pins the Dataverse file ids and MD5s. It reads the
original upload rather than the `.tab` Dataverse generates, because the API
reports the checksum of the original next to the size of the conversion, so a
digest check against the default download fails every time.

`src/lesionsplit/data.py` decodes every image once into one uint8 array, which
is what makes CPU training practical. Augmentation is flips and rotations only:
dermatoscopic images have no canonical orientation, but hue carries diagnostic
information, so colour jitter would be teaching the model something untrue.

`src/lesionsplit/model.py` is four convolutional blocks and nothing clever. The
first block strides, which halves the side of every feature map from that
convolution on, so every convolution computes a quarter as many outputs as it
would without the stride. It is there to keep CPU training short.

`src/lesionsplit/train.py` counts the class weights from the training labels
alone, so no test label reaches the loss. Each run also scores two baselines
that never look at the image, on the same test set.

`src/lesionsplit/tables.py` renders every table in this file, and the timing
summary under "Running it", from the committed artifacts. The one exception is
the batch size, which no artifact records. A marker pair (the two HTML comments
around each table) with no generator, and a generator with no marker pair, are
both errors, and `tests/test_readme.py` fails if the committed README is not
what the artifacts render. The numbers in the prose
are typed, and that file checks some of them, one test per figure.

</details>

<details>
<summary><b>What this does not do</b></summary>

No transfer learning. An ImageNet backbone would score higher and would make the
comparison harder to trust, since the question is about this corpus and a
pretrained network has seen a hundred times more data than it holds.

No claim about clinical use. 64 pixel inputs and a 390k parameter network are a
reproducibility choice, not a medical one, and nothing here should be read as a
diagnostic result.

No tuning. Both arms get identical hyperparameters, chosen once and not searched.
Searching them per arm would make the arms differ in more than the split.

No statement about which published papers are affected. Demonstrating that a
protocol leaks is not the same as auditing the literature that used it, and the
second is a much larger piece of work.

</details>

<details>
<summary><b>Data, licence, and why not the Kaggle copy</b></summary>

HAM10000, Harvard Dataverse version 4.0, `doi:10.7910/DVN/DBW86T`. CC BY-NC 4.0.

Tschandl, P., Rosendahl, C. and Kittler, H. The HAM10000 dataset, a large
collection of multi-source dermatoscopic images of common pigmented skin
lesions. *Scientific Data* 5, 180161 (2018).

Nothing downloaded is committed. Non-commercial terms and a public repository do
not mix, so the images land in a gitignored directory on the reader's machine
under whatever terms they accepted.

The code in this repository is MIT licensed and the dataset is not. [NOTICE.md](NOTICE.md)
says which licence covers what.

Kaggle hosts this dataset in many forms: balanced, augmented, hair removed,
resized to 28 by 28. Training on a pre-augmented copy leaks by
construction, because derived images of one original sit on both sides of any
split no matter how careful the code downstream is. Jiménez-Sánchez et al.,
`Copycats: the many lives of a publicly available medical imaging dataset`
(NeurIPS 2024 Datasets and Benchmarks, arXiv:2402.06353), survey datasets
re-hosted on Kaggle and HuggingFace, and counted 141 copies of this one on
Kaggle in October 2024. The Kaggle page is a reasonable way to find the data and
a poor way to take it.

</details>

<details>
<summary><b>How this was built</b></summary>

The checks in this repository exist because of specific things that went wrong
while writing it, not as decoration. Two worth naming, both caught by the code
rather than by reading it:

The fetch module originally claimed that Dataverse's quoted `.tab` would make
every lesion look unique and silently zero the leakage measurement. It sounded
right and it is false: pandas removes the quotes on read and both files give
7,470 lesions. Measuring it is what showed the claim was invented. There is a
test named for the withdrawal so it stays withdrawn.

The model docstring claimed 293,000 parameters. The network has 390,695. A
docstring is not checked by anything, which is why the figure in this README is
generated from the artifact and the parameter count is asserted in a test.

</details>
