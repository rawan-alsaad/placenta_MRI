"""U-Net training with early stopping on validation loss."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import make_loader
from .metrics import bce_dice_loss
from .model import UNet2D


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg: Config, device: torch.device) -> UNet2D:
    return UNet2D(base_channels=cfg.base_channels, dropout=cfg.dropout).to(device)


def load_model(checkpoint_path: Path, cfg: Config, device: torch.device) -> UNet2D:
    model = build_model(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def _amp_dtype(device: torch.device) -> torch.dtype:
    """float16 mixed precision on CUDA (as in the paper); bfloat16 on CPU."""
    return torch.float16 if device.type == "cuda" else torch.bfloat16


def _run_epoch(model, loader: DataLoader, device, optimizer=None, scaler=None,
               amp: bool = False, clip: float = 5.0) -> float:
    """One pass over ``loader``; trains when an optimizer is given. Returns mean loss."""
    training = optimizer is not None
    model.train(training)
    total, batches = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(device), enabled=amp):
                loss = bce_dice_loss(model(images), masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            total += float(loss.detach())
            batches += 1
    return total / batches


def train_unet(train_df: pd.DataFrame, validation_df: pd.DataFrame, cfg: Config,
               checkpoint_path: Path, device: torch.device) -> pd.DataFrame:
    """Train one U-Net and save the checkpoint with the lowest validation loss.

    The full training state is saved after every epoch to ``training_state.pt``
    next to the checkpoint, so an interrupted run resumes from the last completed
    epoch. The state file is removed once training finishes.

    Returns the per-epoch training history (also written next to the checkpoint).
    """
    seed_everything(cfg.seed)
    model = build_model(cfg, device)
    train_loader = make_loader(train_df, cfg, train=True)
    validation_loader = make_loader(validation_df, cfg, train=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs,
                                                           eta_min=1e-6)
    amp = cfg.use_amp
    # Loss scaling is only needed for float16 (CUDA); bfloat16 does not need it.
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = training_state_path(checkpoint_path)
    best_loss, best_epoch, history, start_epoch = float("inf"), 0, [], 1

    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        best_loss, best_epoch, history = state["best_loss"], state["best_epoch"], state["history"]
        start_epoch = state["epoch"] + 1
        print(f"  resuming from epoch {state['epoch']} (best epoch so far {best_epoch})")

    for epoch in range(start_epoch, cfg.max_epochs + 1):
        start = time.time()
        train_loss = _run_epoch(model, train_loader, device, optimizer, scaler, amp,
                                cfg.grad_clip_norm)
        validation_loss = _run_epoch(model, validation_loader, device, amp=amp)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "seconds": round(time.time() - start, 1)})
        improved = validation_loss < best_loss
        print(f"  epoch {epoch:3d} | train loss {train_loss:.4f} | "
              f"val loss {validation_loss:.4f} | {time.time() - start:5.0f}s"
              + ("  *" if improved else ""), flush=True)

        if improved:
            best_loss, best_epoch = validation_loss, epoch
            _atomic_save({"model_state_dict": model.state_dict(), "epoch": epoch,
                          "validation_loss": best_loss}, checkpoint_path)
        elif epoch - best_epoch >= cfg.early_stopping_patience:
            print(f"  early stopping at epoch {epoch}; best epoch {best_epoch}")
            break

        _atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                      "torch_rng": torch.get_rng_state(), "epoch": epoch,
                      "best_loss": best_loss, "best_epoch": best_epoch,
                      "history": history}, state_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(checkpoint_path.with_name("history.csv"), index=False)
    state_path.unlink(missing_ok=True)
    return history_df


def training_state_path(checkpoint_path: Path) -> Path:
    """Location of the resumable training state that belongs to a checkpoint."""
    return checkpoint_path.with_name("training_state.pt")


def _atomic_save(obj: dict, path: Path) -> None:
    """Save via a temporary file so an interruption never leaves a corrupt file."""
    temporary = path.with_suffix(".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)
