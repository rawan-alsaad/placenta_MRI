"""End-to-end U-Net cross-sequence experiment.

1. Index the public dataset and apply the fixed patient-level split.
2. Train three U-Nets: BTFE-only, ssh-TSE-only, and pooled BTFE + ssh-TSE.
   An interrupted run resumes from its last completed epoch. If a finished
   checkpoint already exists in the output folder, training of that model is
   skipped and the saved model is evaluated (use --retrain to override).
3. Evaluate each model on the 13 held-out test patients for both sequences.
4. Write patient-level results and the U-Net rows of Tables 1 and 2.

Usage:
    python run_experiment.py --data-dir data --output-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from placenta_seg.config import CONDITIONS, TRAINING_SOURCES, Config
from placenta_seg.data import build_manifest, select
from placenta_seg.evaluate import patient_metrics, predict
from placenta_seg.stats import table1_dice, table2_cross_sequence
from placenta_seg.train import load_model, train_unet, training_state_path


def parse_args() -> argparse.Namespace:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--split-file", type=Path, default=defaults.split_file)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--max-epochs", type=int, default=None,
                        help=f"default {defaults.max_epochs} (1 with --smoke-test)")
    parser.add_argument("--retrain", action="store_true",
                        help="retrain all models even if checkpoints already exist")
    parser.add_argument("--smoke-test", action="store_true",
                        help="2 patients per split and 1 epoch, to check the installation")
    return parser.parse_args()


def smoke_subset(manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep two patients per split (that have both sequences) for a quick test."""
    both = manifest.groupby("subject_id")["sequence"].nunique() == 2
    keep = (manifest[manifest["subject_id"].isin(both[both].index)]
            .groupby("split")["subject_id"].unique().apply(lambda s: list(s[:2])).sum())
    return manifest[manifest["subject_id"].isin(keep)].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    cfg = Config(data_dir=args.data_dir, split_file=args.split_file, output_dir=args.output_dir,
                 num_workers=args.num_workers)
    if args.smoke_test:
        cfg.max_epochs, cfg.bootstrap_resamples = 1, 200
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    device = torch.device(args.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data -------------------------------------------------------------
    manifest = build_manifest(cfg)
    if args.smoke_test:
        manifest = smoke_subset(manifest)
    print(manifest.groupby(["sequence", "split"])
          .agg(images=("subject_id", "size"), patients=("subject_id", "nunique")), "\n")

    # 2. Training ---------------------------------------------------------
    checkpoints = {}
    for source, sequences in TRAINING_SOURCES.items():
        checkpoints[source] = cfg.output_dir / "models" / source / "model_best.pt"
        state = training_state_path(checkpoints[source])
        if args.retrain:
            state.unlink(missing_ok=True)
        elif checkpoints[source].exists() and not state.exists():
            print(f"[{source}] SKIPPING TRAINING: found existing checkpoint "
                  f"{checkpoints[source]}\n"
                  f"          Results will come from this saved model. To retrain, use "
                  f"--retrain, delete the checkpoint, or choose a new --output-dir.")
            continue
        print(f"[{source}] training U-Net on {' + '.join(sequences)} ({device})")
        train_unet(select(manifest, "train", sequences), select(manifest, "validation", sequences),
                   cfg, checkpoints[source], device)

    # 3. Evaluation on the held-out patients -------------------------------
    results = []
    for label, source, target in CONDITIONS:
        model = load_model(checkpoints[source], cfg, device)
        test_df = select(manifest, "test", (target,))
        patients = patient_metrics(test_df, *predict(model, test_df, cfg, device))
        patients.insert(0, "condition", label)
        results.append(patients)
        print(f"{label:<20} mean patient-level Dice = {patients['dice'].mean():.4f}")
    results = pd.concat(results, ignore_index=True)
    results.to_csv(cfg.output_dir / "patient_results.csv", index=False)

    # 4. Tables -----------------------------------------------------------
    table1 = table1_dice(results, cfg.bootstrap_resamples)
    table2 = table2_cross_sequence(results, cfg.bootstrap_resamples)
    table1.to_csv(cfg.output_dir / "table1_dice.csv", index=False)
    table2.to_csv(cfg.output_dir / "table2_cross_sequence.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\nTable 1. Mean patient-level Dice (95% BCa CI)\n", table1.round(3).to_string(index=False))
        print("\nTable 2. Cross-sequence loss and pooled-training recovery\n",
              table2.set_index("target_sequence").T.to_string())
    print(f"\nResults written to {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()
