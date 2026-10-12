"""Download the small Zenodo files and verify their md5 checksums."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import quote

import requests

from larvalens.paths import RAW

RECORD = 13359376
URL = "https://zenodo.org/records/{record}/files/{name}?download=1"

# File name -> md5, from the Zenodo record.
FILES: dict[str, str] = {
    "Larvae--size-data.xlsx": "8aa2c6926db0d1554c07a99d12dd060d",
    "BSF_Larvae_v1.zip": "a5a104e2c10bbc499ad3c45b71d2ffd8",
    "Larva_sex_classification.zip": "0e28870daeb6bb4ffb1142aeb4811206",
}
BIG_FILE = ("Single Larvae data.zip", "ea931d09ac8533f909d73371d45bf43d")


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    """Return the md5 hex digest of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def file_url(name: str, record: int = RECORD) -> str:
    """Return the Zenodo download URL for a file name."""
    return URL.format(record=record, name=quote(name))


def fetch(name: str, md5: str, outdir: Path = RAW) -> Path:
    """Download one file unless it is present, then verify its md5.

    Raises ValueError if the checksum does not match.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        with requests.get(file_url(name), stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for block in resp.iter_content(1 << 20):
                    fh.write(block)
        tmp.rename(path)
    found = md5sum(path)
    if found != md5:
        raise ValueError(f"md5 mismatch for {name}: expected {md5}, found {found}")
    return path


def fetch_all(outdir: Path = RAW) -> list[Path]:
    """Download and verify every small file."""
    return [fetch(name, md5, outdir) for name, md5 in FILES.items()]
