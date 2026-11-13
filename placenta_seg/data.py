"""Dataset indexing, patient-level split, preprocessing and augmentation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

from .config import SEQUENCES, Config

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def parse_subject_and_slice(stem: str) -> tuple[str, int]:
    """Split a file stem such as ``sub001_10`` or ``v12_3`` into (subject, slice)."""
    match = re.match(r"^(?P<subject>.+?)[_-](?P<slice>\d+)$", stem.strip())
    if match is None:
        raise ValueError(f"Cannot parse subject and slice from file name: {stem!r}")
    subject = re.sub(r"[^a-z0-9]+", "_", match.group("subject").lower()).strip("_")
    return subject, int(match.group("slice"))


def _index_files(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(
            f"Directory not found: {root}. Run `python scripts/download_data.py` first."
        )
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    index: dict[str, Path] = {}
    for path in files:
        key = path.stem.lower()
        if key in index:
            raise ValueError(f"Duplicate file key {key!r} in {root}")
        index[key] = path
    return index


def build_manifest(cfg: Config) -> pd.DataFrame:
    """Pair every image with its mask and attach the fixed patient-level split.

    The same subject ID in BTFE and ssh-TSE denotes the same patient, so the
    split is applied per patient across both sequences.
    """
    rows = []
    for sequence in SEQUENCES:
        image_dir, mask_dir = cfg.sequence_dirs(sequence)
        images, masks = _index_files(image_dir), _index_files(mask_dir)
        if images.keys() != masks.keys():
            raise ValueError(
                f"{sequence}: {len(images.keys() ^ masks.keys())} images/masks are unpaired"
            )
        for key in sorted(images):
            subject, slice_index = parse_subject_and_slice(images[key].stem)
            rows.append({
                "sequence": sequence,
                "subject_id": subject,
                "slice_index": slice_index,
                "image_path": str(images[key]),
                "mask_path": str(masks[key]),
            })
    manifest = pd.DataFrame(rows)

    split = pd.read_csv(cfg.split_file)[["global_subject_id", "split"]]
    split = split.rename(columns={"global_subject_id": "subject_id"})
    manifest = manifest.merge(split, on="subject_id", how="left", validate="many_to_one")
    missing = manifest.loc[manifest["split"].isna(), "subject_id"].unique()
    if len(missing):
        raise ValueError(f"Subjects missing from {cfg.split_file}: {sorted(missing)[:10]}")
    return manifest


def select(manifest: pd.DataFrame, split: str, sequences: tuple[str, ...]) -> pd.DataFrame:
    """Rows of one split ('train'/'validation'/'test') for the given sequences."""
    keep = (manifest["split"] == split) & manifest["sequence"].isin(sequences)
    return manifest[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessing and augmentation
# ---------------------------------------------------------------------------
def robust_normalize(array: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """Clip to the [lower, upper] percentile range and rescale to [0, 1]."""
    array = array.astype(np.float32)
    lo, hi = np.percentile(array, [lower, upper])
    if hi <= lo:
        lo, hi = float(array.min()), float(array.max())
    if hi <= lo:
        return np.zeros_like(array)
    return ((np.clip(array, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def augment(image: Image.Image, mask: Image.Image, rng: np.random.Generator):
    """Horizontal flip, small rotation, contrast and gamma jitter (no vertical flip)."""
    if rng.random() < 0.5:
        image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
    if rng.random() < 0.7:
        angle = float(rng.uniform(-10.0, 10.0))
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    if rng.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.85, 1.15)))
    if rng.random() < 0.3:
        gamma = float(rng.uniform(0.85, 1.15))
        array = np.power(np.asarray(image).astype(np.float32) / 255.0, gamma)
        image = Image.fromarray(np.uint8(np.clip(array * 255.0, 0, 255)))
    return image, mask


class PlacentaSlices(Dataset):
    """2D MRI slices resized to a square grid, with binary placenta masks.

    Augmentation randomness is seeded per sample (``seed + len(dataset) + index``),
    so each training slice receives the same augmentation in every epoch. This is
    the behaviour of the original study runs and is kept for reproducibility.
    """

    def __init__(self, df: pd.DataFrame, image_size: int, augment: bool = False, seed: int = 2026):
        self.df = df.reset_index(drop=True)
        self.size = (image_size, image_size)
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("L").resize(self.size, Image.Resampling.BILINEAR)
        mask = Image.open(row["mask_path"]).convert("L").resize(self.size, Image.Resampling.NEAREST)

        if self.augment:
            rng = np.random.default_rng(self.seed + len(self.df) + index)
            image, mask = augment(image, mask, rng)

        image = robust_normalize(np.asarray(image))
        mask = (np.asarray(mask) > 0).astype(np.float32)
        return {
            "image": torch.from_numpy(image[None]),
            "mask": torch.from_numpy(mask[None]),
            "index": index,
        }


def make_loader(df: pd.DataFrame, cfg: Config, train: bool) -> DataLoader:
    dataset = PlacentaSlices(df, cfg.image_size, augment=train, seed=cfg.seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
