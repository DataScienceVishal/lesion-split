"""Getting HAM10000, from the archive the authors published rather than a copy.

The dataset is on Kaggle in at least a dozen forms: balanced, augmented, hair
removed, resized to 28 by 28. Several of those are worse than useless here.
An augmented copy has multiple derived images per original, so a random split
over it leaks by construction and no amount of care downstream undoes that.
`Copycats: the many lives of a publicly available medical imaging dataset`
(arXiv:2402.06353) is about this dataset specifically.

So this reads Harvard Dataverse, version 4.0, file identities and MD5 digests
pinned below. It is the copy the data descriptor points at, it needs no account,
and pinning the digest means a changed file is an error rather than a surprise.

CC BY-NC 4.0, which is why nothing downloaded here is ever committed. The images
land in a gitignored directory on the reader's own machine.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

DOI = "doi:10.7910/DVN/DBW86T"
VERSION = "4.0"
BASE = "https://dataverse.harvard.edu/api/access/datafile"
CACHE = Path(".ham10000")


@dataclass(frozen=True)
class Archived:
    """One file in the published archive, pinned by id and digest.

    Dataverse serves a file by numeric id, and those ids do not move between
    versions the way a filename can, so the id is the stable handle. The MD5 is
    the archive's own, not one computed here: verifying against a digest this
    project chose would only prove the download matched itself.
    """

    filename: str
    file_id: int
    size: int
    md5: str
    ingested: bool = False

    @property
    def url(self) -> str:
        # Dataverse ingests tabular uploads and then serves the converted copy by
        # default. For this dataset the API reports the size of the conversion
        # (830,428 bytes, every field quoted) next to the checksum of the
        # original (690,218 bytes, plain CSV), so the two fields describe
        # different files and a digest check against the default download fails.
        # `format=original` asks for the bytes the checksum is actually about.
        return f"{BASE}/{self.file_id}" + ("?format=original" if self.ingested else "")


METADATA = Archived(
    "HAM10000_metadata.csv", 4338392, 690_218, "8f85fb1aa29d80a2797247e434deb79d", ingested=True
)
IMAGES = (
    Archived(
        "HAM10000_images_part_1.zip", 3172585, 1_366_522_108, "4639bfa73ab251610530a97c898e6e46"
    ),
    Archived(
        "HAM10000_images_part_2.zip", 3172584, 1_403_566_547, "da43d6cc50f6613013be07e8986b384b"
    ),
)


class DigestMismatch(RuntimeError):
    pass


def digest(path: Path) -> str:
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            md5.update(block)
    return md5.hexdigest()


def download(item: Archived, into: Path = CACHE) -> Path:
    """Fetch once, verify every time.

    Re-verifying a file already on disk costs a few seconds on 1.4 GB and is
    what turns the cache from an optimisation into a guarantee. A truncated
    download that was interrupted halfway looks exactly like a complete one
    until something reads the digest.
    """
    into.mkdir(parents=True, exist_ok=True)
    target = into / item.filename
    if not target.exists():
        # Dataverse answers urllib's default user agent with 403 while serving
        # the same URL to curl, so the header is load-bearing rather than polite.
        ask = Request(item.url, headers={"User-Agent": f"lesion-split (+{DOI})"})
        with urlopen(ask) as response, target.open("wb") as handle:
            while block := response.read(1 << 20):
                handle.write(block)
    found = digest(target)
    if found != item.md5:
        raise DigestMismatch(
            f"{item.filename} hashes to {found}, and Dataverse {VERSION} records {item.md5}. "
            f"Delete {target} and run this again."
        )
    return target


def metadata(into: Path = CACHE) -> pd.DataFrame:
    """The 830 KB file that carries the whole argument.

    This reads the original upload rather than Dataverse's ingested `.tab`, and
    the reason is the digest rather than the contents. Both parse to the same
    10,015 rows over 7,470 lesions; the conversion quotes its string fields and
    pandas removes those quotes on read, so nothing downstream can tell.

    I first wrote that the quoting was a hazard, that quoted and bare ids would
    compare unequal and every lesion would look unique. That is wrong, and
    measuring it is what showed it: both files give 7,470 either way. The reason
    to ask for the original is narrower and real, and it is in `Archived.url`.
    """
    return pd.read_csv(download(METADATA, into), dtype=str)


def images(into: Path = CACHE) -> Path:
    """Both zips, unpacked into one flat directory of 10,015 JPEGs."""
    unpacked = into / "images"
    unpacked.mkdir(parents=True, exist_ok=True)
    for part in IMAGES:
        archive = download(part, into)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                if not member.lower().endswith(".jpg"):
                    continue
                destination = unpacked / Path(member).name
                if destination.exists():
                    continue
                with bundle.open(member) as source, destination.open("wb") as handle:
                    handle.write(source.read())
    return unpacked


def one_image(image_id: str, into: Path = CACHE) -> io.BytesIO:
    """A single JPEG straight out of the zip, for a fixture without a 2.8 GB unpack."""
    for part in IMAGES:
        archive = into / part.filename
        if not archive.exists():
            continue
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                if Path(member).stem == image_id:
                    return io.BytesIO(bundle.read(member))
    raise FileNotFoundError(f"{image_id} is in neither downloaded archive")
