"""Experiment configuration for the U-Net cross-sequence experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# MRI sequences (domains) in the public dataset.
BTFE = "BTFE"
SSH_TSE = "ssh-TSE"
SEQUENCES = (BTFE, SSH_TSE)

# Training sources. Each source model is trained once and then evaluated on
# both target sequences.
TRAINING_SOURCES: dict[str, tuple[str, ...]] = {
    "btfe": (BTFE,),
    "sshtse": (SSH_TSE,),
    "pooled": (BTFE, SSH_TSE),
}

# The six evaluation conditions reported in Table 1 of the paper:
# (label, training source, target sequence).
CONDITIONS: list[tuple[str, str, str]] = [
    ("BTFE -> BTFE", "btfe", BTFE),
    ("BTFE -> ssh-TSE", "btfe", SSH_TSE),
    ("ssh-TSE -> ssh-TSE", "sshtse", SSH_TSE),
    ("ssh-TSE -> BTFE", "sshtse", BTFE),
    ("Pooled -> BTFE", "pooled", BTFE),
    ("Pooled -> ssh-TSE", "pooled", SSH_TSE),
]


@dataclass
class Config:
    # Paths
    data_dir: Path = Path("data")
    split_file: Path = Path("patient_level_split.csv")
    output_dir: Path = Path("outputs")

    # Preprocessing
    image_size: int = 256
    seed: int = 2026

    # U-Net
    base_channels: int = 32
    dropout: float = 0.20

    # Optimisation
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    early_stopping_patience: int = 12
    grad_clip_norm: float = 5.0
    use_amp: bool = True  # mixed precision: float16 on CUDA, bfloat16 on CPU
    num_workers: int = 4

    # Statistics
    bootstrap_resamples: int = 10_000

    def sequence_dirs(self, sequence: str) -> tuple[Path, Path]:
        """Return the (image_dir, mask_dir) of a sequence."""
        prefix = {BTFE: "BTFE", SSH_TSE: "ssh_TSE"}[sequence]
        return self.data_dir / f"{prefix}_image", self.data_dir / f"{prefix}_mask"
