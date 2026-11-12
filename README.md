# Cross-Sequence Robustness of Deep Learning for Placental MRI Segmentation

### A Patient-Matched Study in Placenta Accreta Spectrum

**Research code and reproducibility resources** accompanying the manuscript *Cross-Sequence Robustness of Deep Learning for Placental MRI Segmentation: A Patient-Matched Study in Placenta Accreta Spectrum*.

## Overview

Deep learning can automate placental segmentation on magnetic resonance imaging (MRI), but strong performance on one acquisition sequence does not necessarily translate to another. This study investigates **cross-sequence robustness** in placenta accreta spectrum (PAS) using paired acquisitions from the same patients.

We evaluated segmentation models trained on **balanced turbo field echo (BTFE)** and **single-shot turbo spin echo (ssh-TSE)** MRI, examining both directions of transfer and whether **pooled multisequence training** improves generalization. By maintaining patient-matched evaluation across sequences, the study reduces confounding from differences in patient populations.

## Study at a glance

| Component | Description |
|---|---|
| Dataset | 7,106 MRI image–mask pairs: 3,411 BTFE and 3,695 ssh-TSE |
| Patients | 133 unique subjects; 130 represented in both sequences |
| Held-out evaluation | 13 patients with both sequences (345 BTFE and 371 ssh-TSE slices) |
| Segmentation pipelines | **U-Net**, **nnU-Net v2**, and **U-Mamba Bot** |
| Training conditions | BTFE-only, ssh-TSE-only, and pooled BTFE + ssh-TSE |
| Primary metric | Patient-level aggregated Dice coefficient |
| Additional analyses | Boundary error (HD95), segmentation-derived size error, and size-matched sensitivity analysis |

All partitions were made **at the patient level**, with the same patient assignments preserved across both sequences to avoid train–test leakage.

## Key findings

1. **Cross-sequence transfer reduced segmentation accuracy.** In-domain mean patient-level Dice ranged from **0.888 to 0.917**, but performance declined when models were applied to a sequence absent from their training data.
2. **Degradation was directionally asymmetric.** Dice losses were **0.066–0.119** for **BTFE → ssh-TSE**, versus **0.039–0.051** for **ssh-TSE → BTFE** (all *p* < 0.001).
3. **Multisequence training largely restored performance.** Joint training recovered **0.044–0.118 Dice** relative to direct cross-sequence transfer, achieving mean performance close to the corresponding sequence-specific models.
4. **The pattern was consistent across three segmentation pipelines.** The magnitude differed by pipeline, but the study was not designed to establish that one architecture is intrinsically more robust than another.
5. **The benefit of sequence diversity was supported by a sensitivity analysis.** A size-matched, patient-exclusive U-Net trained on both sequences retained improved cross-sequence performance, although this analysis was limited to one pipeline.

### Mean patient-level Dice on the held-out cohort

| Training → Testing | U-Net | nnU-Net v2 | U-Mamba Bot |
|---|---:|---:|---:|
| BTFE → BTFE (in-domain) | 0.908 | 0.917 | 0.915 |
| BTFE → ssh-TSE (cross-sequence) | 0.822 | 0.791 | 0.784 |
| ssh-TSE → ssh-TSE (in-domain) | 0.888 | 0.899 | 0.903 |
| ssh-TSE → BTFE (cross-sequence) | 0.870 | 0.866 | 0.867 |
| Pooled → BTFE | 0.914 | 0.915 | 0.916 |
| Pooled → ssh-TSE | 0.894 | 0.901 | 0.902 |

*Results are mean patient-level aggregated Dice scores from the same 13 held-out patients per target sequence. See the manuscript for confidence intervals and paired statistical analyses.*

## Data and reproducibility

**Source imaging dataset:** [Placenta accreta spectrum disorders (PASDs), Mendeley Data, Version 1](https://doi.org/10.17632/284gwmf5bh.1) (CC BY 4.0).

**Patient-level partition file:** `patient_level_split.csv`. This supplementary file records the fixed training, validation, and test patient assignments used across MRI sequences and is intended to facilitate independent reproduction of the experimental splits. The original split assignments—not newly generated random splits—should be used when reproducing the reported evaluation.

Implementation details, training configurations, statistical methodology, and limitations are described in the accompanying manuscript.

## Code: reproducing the main U-Net experiment

This repository contains one end-to-end pipeline: the **U-Net cross-sequence experiment**. It covers the U-Net column of Table 1 and the U-Net rows of Table 2 in the paper. Three U-Nets are trained (BTFE-only, ssh-TSE-only and pooled BTFE + ssh-TSE). Each is then evaluated on the same 13 held-out patients in both sequences. The nnU-Net v2 and U-Mamba Bot pipelines were run with their official frameworks, as described in the manuscript. They are not included here.

### Repository layout

```text
placenta_mri/
├── README.md
├── requirements.txt
├── patient_level_split.csv     # fixed patient-level train/validation/test split
├── run_experiment.py           # end-to-end pipeline (train -> evaluate -> tables)
├── scripts/
│   └── download_data.py        # downloads and extracts the Mendeley dataset
└── placenta_seg/
    ├── config.py               # hyperparameters, training sources, evaluation conditions
    ├── data.py                 # file pairing, split, preprocessing, augmentation
    ├── model.py                # 2D U-Net
    ├── metrics.py              # BCE + soft-Dice loss, Dice, HD95, RVD
    ├── train.py                # training with early stopping
    ├── evaluate.py             # inference and patient-level aggregation
    └── stats.py                # BCa bootstrap CIs and Wilcoxon tests (Tables 1-2)
```

### 1. Install

Requires Python 3.10 or newer. A CUDA GPU is strongly recommended; see the run times below.

```bash
pip install -r requirements.txt
```

### 2. Download the data

```bash
python scripts/download_data.py --data-dir data
```

This downloads the four RAR archives (about 170 MB) from Mendeley Data, verifies their SHA-256 checksums and extracts them to `data/BTFE_image`, `data/BTFE_mask`, `data/ssh_TSE_image` and `data/ssh_TSE_mask`. Extraction needs `unar`, `7z` or `unrar` on your system (for example, `sudo apt install unar` or `brew install unar`).

### 3. Run

```bash
python run_experiment.py --data-dir data --output-dir outputs
```

To check that everything is installed correctly, first run a short smoke test (2 patients per split, 1 epoch, about 2 minutes on a CPU):

```bash
python run_experiment.py --data-dir data --output-dir outputs_smoke --smoke-test
```

The full run writes:

| File | Content |
|---|---|
| `outputs/models/{btfe,sshtse,pooled}/model_best.pt` | checkpoint with the lowest validation loss |
| `outputs/models/*/history.csv` | per-epoch training and validation loss |
| `outputs/patient_results.csv` | patient-level Dice, RVD and HD95 for all six conditions |
| `outputs/table1_dice.csv` | mean patient-level Dice with 95% BCa CI (Table 1, U-Net) |
| `outputs/table2_cross_sequence.csv` | cross-sequence loss, pooled-training recovery and pooled minus sequence-specific Dice, with BCa CIs and Wilcoxon tests (Table 2, U-Net) |

If a checkpoint already exists at `outputs/models/<source>/model_best.pt`, training of that model is skipped and the saved model is evaluated; the script prints a warning when this happens. This lets an interrupted run resume, and lets you evaluate previously trained models by placing them there. If you change any setting, use a new `--output-dir` or add `--retrain` so models are trained from scratch:

```bash
python run_experiment.py --data-dir data --output-dir outputs --retrain
```

If a run is interrupted (for example, a pre-empted GPU job), rerun the same command without `--retrain`. Training resumes from the last completed epoch, using the `training_state.pt` file that is saved after every epoch and removed when training finishes.

### Method summary

| Step | Setting |
|---|---|
| Split | Fixed patient-level split from `patient_level_split.csv`, shared by both sequences: 105/13/13 BTFE and 106/13/13 ssh-TSE patients (train/validation/test) |
| Preprocessing | Grayscale; resized to 256 × 256 (bilinear for images, nearest neighbour for masks); intensities clipped to the 1st–99th percentiles and rescaled to [0, 1] |
| Augmentation | Horizontal flip (p = 0.5), rotation ±10° (p = 0.7), contrast 0.85–1.15 (p = 0.5), gamma 0.85–1.15 (p = 0.3). The random draw is seeded per slice, so each slice receives the same augmentation in every epoch, as in the original runs |
| Model | 2D U-Net, 4 encoder levels, 32 initial channels, dropout 0.20 (7.76 M parameters) |
| Optimisation | BCE + soft-Dice loss, AdamW (lr 1e-4, weight decay 1e-5), cosine annealing, batch size 8, gradient clipping at 5.0, mixed precision, at most 100 epochs, early stopping after 12 epochs without improvement in validation loss |
| Validation | Same sequence(s) as training; no target-sequence data is used for model selection |
| Primary outcome | Patient-level aggregated Dice (all test slices of a patient pooled), on the 256 × 256 grid |
| Statistics | 95% BCa bootstrap CIs (10,000 patient-level resamples); two-sided Wilcoxon signed-rank tests on paired patient-level Dice |

### Expected run time

| Hardware | Approximate time for the full run |
|---|---|
| NVIDIA A100 (as used in the paper) | under 1 hour (about 10 s per epoch for one sequence, 20 s for pooled) |
| 4-core CPU (bfloat16 mixed precision) | about 17 hours (7–16 min per epoch) |

### Reproducibility notes

- Seeds are fixed (2026). GPU kernels, mixed-precision arithmetic and library versions still introduce small numerical differences. Retrained models therefore match the paper closely, but not to every decimal.
- Mixed precision uses float16 on CUDA, as in the paper, and bfloat16 on CPU.


## Manuscript and citation

This repository accompanies the following peer-reviewed, open-access publication:

**AlSaad, R., Ahmed, S., Al-Dewik, N., Farrell, T., & Thomas, R. (2026).** Cross-Sequence Robustness of Deep Learning for Placental MRI Segmentation: A Patient-Matched Study in Placenta Accreta Spectrum. *Bioengineering, 13*(11), 1247.

**Published article:** https://doi.org/10.3390/bioengineering13111247

**Supplementary materials:** https://www.mdpi.com/article/10.3390/bioengineering13111247/s1

If you use this repository, its code, or the experimental methodology in your research, please cite the publication above.

**BibTeX citation:**

```bibtex
@article{AlSaad2026PlacentaMRI,
  author  = {AlSaad, Rawan and Ahmed, Sanaa
             and Al-Dewik, Nader and Farrell, Thomas
             and Thomas, Rajat},
  title   = {Cross-Sequence Robustness of Deep Learning
             for Placental MRI Segmentation:
             A Patient-Matched Study in Placenta
             Accreta Spectrum},
  journal = {Bioengineering},
  year    = {2026},
  volume  = {13},
  number  = {11},
  pages   = {1247},
  doi     = {10.3390/bioengineering13111247},
  url     = {https://doi.org/10.3390/bioengineering13111247}
}
```



## License

The code in this repository is released under the [MIT License](LICENSE). You may use, modify and redistribute it, including commercially, provided the copyright and license notice are kept. Please also cite the publication above.

The MRI dataset is not part of this repository. It is distributed separately by its authors on Mendeley Data under the CC BY 4.0 license, which has its own attribution requirements.
