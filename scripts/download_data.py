"""Download and extract the public PAS MRI dataset from Mendeley Data.

Dataset: Placenta accreta spectrum disorders (PASDs), Mendeley Data, V1
         https://doi.org/10.17632/284gwmf5bh.1  (CC BY 4.0)

The four archives are RAR files. Extraction needs one of: `unar`, `7z` or `unrar`
(e.g. `sudo apt install unar`, `brew install unar`, or 7-Zip on Windows).

Usage:
    python scripts/download_data.py --data-dir data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

API_URL = "https://data.mendeley.com/public-api/datasets/284gwmf5bh/files?folder_id=root&version=1"
ARCHIVES = ("BTFE_image", "BTFE_mask", "ssh_TSE_image", "ssh_TSE_mask")
HEADERS = {"User-Agent": "Mozilla/5.0 (dataset download script)"}


def fetch(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.which("unar"):
        command = ["unar", "-q", "-f", "-D", "-o", str(destination), str(archive)]
    elif shutil.which("7z"):
        command = ["7z", "x", "-y", f"-o{destination}", str(archive)]
    elif shutil.which("unrar"):
        command = ["unrar", "x", "-o+", str(archive), f"{destination}/"]
    else:
        raise SystemExit("No RAR extractor found. Install `unar`, `7z` or `unrar` and rerun.")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    archive_dir = args.data_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    with fetch(API_URL) as response:
        files = {Path(f["filename"]).stem: f for f in json.load(response)}

    for name in ARCHIVES:
        details = files[name]["content_details"]
        archive = archive_dir / f"{name}.rar"
        if not archive.exists() or sha256(archive) != details["sha256_hash"]:
            print(f"Downloading {archive.name} ({details['size'] / 1e6:.0f} MB)")
            with fetch(details["download_url"]) as response, archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        if sha256(archive) != details["sha256_hash"]:
            raise SystemExit(f"Checksum mismatch for {archive}")
        # Each archive holds a single folder (JPEGImages/ or SegmentationClass/)
        # with identically named files, so each one is extracted separately.
        print(f"Extracting {archive.name} -> {args.data_dir / name}")
        extract(archive, args.data_dir / name)

    for name in ARCHIVES:
        count = sum(1 for p in (args.data_dir / name).rglob("*") if p.is_file())
        print(f"{name:<15} {count} files")


if __name__ == "__main__":
    main()
