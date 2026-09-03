"""Turning 10,015 JPEGs into batches, on a laptop, without a GPU.

The originals are 600 by 450. Decoding and resizing those on every epoch is what
makes CPU training take hours, so the pipeline decodes once into a uint8 array
at the working resolution and keeps it in memory. At 64 by 64 that array is
10,015 x 64 x 64 x 3 bytes, about 123 MB, which fits comfortably and turns each
epoch into arithmetic rather than JPEG decoding.

The resolution is the honest compromise in this project. Dermatologists read
these at full size and 64 pixels discards detail that matters clinically. It is
chosen so the experiment finishes on a CPU in minutes, and the accuracies below
should be read as "what this architecture gets at this resolution", not as what
the task allows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

SIDE = 64

# Fixed here rather than derived from whichever split is loaded. Deriving it
# would make the label meaning depend on the arm, and two arms with different
# label orderings cannot be compared.
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_INDEX = {name: position for position, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class Decoded:
    """Every image, decoded once, plus the label vector aligned to it.

    `by_image_id` maps an image identifier to its row, so a split expressed as a
    DataFrame of metadata can be turned into array indices without touching the
    disk again.
    """

    pixels: np.ndarray
    labels: np.ndarray
    image_ids: tuple[str, ...]

    @property
    def by_image_id(self) -> dict[str, int]:
        return {name: position for position, name in enumerate(self.image_ids)}

    def rows_for(self, frame: pd.DataFrame) -> np.ndarray:
        index = self.by_image_id
        missing = [name for name in frame.image_id if name not in index]
        if missing:
            raise KeyError(
                f"{len(missing)} images are in the metadata and not on disk, "
                f"the first being {missing[0]}"
            )
        return np.array([index[name] for name in frame.image_id], dtype=np.int64)


def decode(frame: pd.DataFrame, folder: Path, side: int = SIDE) -> Decoded:
    """Decode every image in `frame` once, at `side` by `side`, into one array."""
    pixels = np.zeros((len(frame), side, side, 3), dtype=np.uint8)
    labels = np.zeros(len(frame), dtype=np.int64)
    for position, (image_id, diagnosis) in enumerate(zip(frame.image_id, frame.dx, strict=True)):
        with Image.open(folder / f"{image_id}.jpg") as handle:
            small = handle.convert("RGB").resize((side, side), Image.BILINEAR)
            pixels[position] = np.asarray(small)
        labels[position] = CLASS_INDEX[diagnosis]
    return Decoded(pixels=pixels, labels=labels, image_ids=tuple(frame.image_id))


class Lesions(Dataset):
    """A view onto `Decoded` for one side of one split.

    Augmentation is flips and 90 degree rotations only. Dermatoscopic images have
    no canonical orientation, so those are label-preserving here in a way that
    colour jitter would not be: hue carries diagnostic information in pigmented
    lesions, and shifting it would be teaching the model something untrue.
    """

    def __init__(self, source: Decoded, rows: np.ndarray, augment: bool, seed: int = 0):
        self.source = source
        self.rows = rows
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        row = self.rows[position]
        image = self.source.pixels[row]
        if self.augment:
            if self.rng.random() < 0.5:
                image = image[:, ::-1]
            if self.rng.random() < 0.5:
                image = image[::-1, :]
            image = np.rot90(image, self.rng.integers(4))
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        return tensor, int(self.source.labels[row])


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse frequency, so the loss does not simply learn to answer `nv`.

    Two thirds of this corpus is one class. Without weighting, predicting the
    majority every time scores 66.9% accuracy and the four rarest classes
    together are under 11%, so a model can look reasonable while never emitting
    them at all. The weighting is why balanced accuracy is the headline metric
    and plain accuracy is reported beside it rather than instead of it.
    """
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(CLASSES) * counts)
    return torch.from_numpy(weights).float()
