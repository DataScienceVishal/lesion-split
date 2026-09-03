"""The download layer, without downloading anything.

The suite runs with `--disable-socket`, so a test that reached Dataverse would
fail rather than pass slowly. What is worth testing here is not the transfer, it
is the two things that go quietly wrong around it: a digest checked against the
wrong representation of a file, and quoted fields turning every lesion unique.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from lesionsplit import fetch

ROWS = (
    "lesion_id,image_id,dx,dx_type,age,sex,localization,dataset\n"
    "HAM_0000118,ISIC_0027419,bkl,histo,80.0,male,scalp,vidir_modern\n"
    "HAM_0000118,ISIC_0025030,bkl,histo,80.0,male,scalp,vidir_modern\n"
    "HAM_0002730,ISIC_0026769,bkl,histo,80.0,male,scalp,vidir_modern\n"
)


def planted(tmp_path: Path, body: str, name: str = "HAM10000_metadata.csv") -> Path:
    target = tmp_path / name
    target.write_text(body, encoding="utf-8")
    return target


def test_the_digest_is_checked_against_what_is_on_disk(tmp_path):
    planted(tmp_path, ROWS)
    real = hashlib.md5(ROWS.encode(), usedforsecurity=False).hexdigest()
    item = fetch.Archived("HAM10000_metadata.csv", 0, len(ROWS), real, ingested=True)

    assert fetch.download(item, tmp_path).exists()


def test_a_file_that_does_not_match_its_digest_is_an_error(tmp_path):
    """The case that matters: a half-finished download looks complete on disk."""
    planted(tmp_path, ROWS[: len(ROWS) // 2])
    real = hashlib.md5(ROWS.encode(), usedforsecurity=False).hexdigest()
    item = fetch.Archived("HAM10000_metadata.csv", 0, len(ROWS), real, ingested=True)

    with pytest.raises(fetch.DigestMismatch) as raised:
        fetch.download(item, tmp_path)
    assert "HAM10000_metadata.csv" in str(raised.value)
    assert real in str(raised.value)


def test_an_ingested_file_is_asked_for_in_its_original_form():
    """Regression: the recorded checksum and the default download disagree.

    Dataverse converts tabular uploads and serves the conversion by default,
    while the API reports the checksum of the original next to the size of the
    conversion. Downloading the default and checking it against that checksum
    fails every time, and the fix is to ask for the original.
    """
    assert fetch.METADATA.ingested is True
    assert fetch.METADATA.url.endswith("?format=original")


def test_the_image_archives_are_not_asked_for_in_original_form():
    """Zips are not ingested, so the parameter would be meaningless on them."""
    for part in fetch.IMAGES:
        assert part.ingested is False
        assert "format=original" not in part.url


def test_both_representations_parse_to_the_same_lesions(tmp_path):
    """The claim I had to withdraw, kept as a test so it stays withdrawn.

    The first version of this module said Dataverse's quoted `.tab` would make
    every lesion look unique and silently zero the leakage measurement. It does
    not: pandas removes the quotes on read and both files give 7,470 lesions.
    The original is still what gets downloaded, because the recorded checksum
    describes it, but that is a digest argument and not a parsing one.
    """
    quoted = ROWS.replace("HAM_", '"HAM_').replace(",ISIC", '",ISIC')
    plain = pd.read_csv(planted(tmp_path, ROWS), dtype=str)
    converted = pd.read_csv(planted(tmp_path, quoted, "quoted.csv"), dtype=str)

    assert '"' in quoted, "the fixture is not actually quoted"
    assert converted.lesion_id.nunique() == plain.lesion_id.nunique() == 2
    assert list(converted.lesion_id) == list(plain.lesion_id)


def test_the_pinned_sizes_and_digests_are_the_shape_dataverse_returns():
    assert fetch.METADATA.md5 == "8f85fb1aa29d80a2797247e434deb79d"
    assert fetch.METADATA.size == 690_218
    assert len(fetch.IMAGES) == 2
    assert sum(part.size for part in fetch.IMAGES) == 2_770_088_655
    for part in (fetch.METADATA, *fetch.IMAGES):
        assert len(part.md5) == 32 and int(part.md5, 16) >= 0
