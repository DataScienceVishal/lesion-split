# Notice

Two different licences apply here, and they do not cover the same things.

## Code

Everything in this repository is MIT licensed. See [LICENSE](LICENSE).
Copyright (c) 2026 Vishal Khan.

## Data

The HAM10000 dataset is **not** in this repository and is **not** MIT licensed.
No image is committed here. `uv run lesion-split fetch` downloads the corpus at
runtime into a gitignored directory on your own machine, under the dataset's own
terms rather than this repository's.

HAM10000 is licensed **CC BY-NC 4.0**, which permits non-commercial use only.
Harvard Dataverse version 4.0, `doi:10.7910/DVN/DBW86T`.

Tschandl, P., Rosendahl, C. and Kittler, H. The HAM10000 dataset, a large
collection of multi-source dermatoscopic images of common pigmented skin
lesions. *Scientific Data* 5, 180161 (2018).

The MIT grant on the code does not extend to the data, and the non-commercial
restriction on the data does not restrict the code. If you run the training
arms you are using CC BY-NC 4.0 material and its terms apply to what you do
with it.
