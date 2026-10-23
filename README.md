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

Implementation details, training configurations, statistical methodology, and limitations are described in the accompanying manuscript. Additional analysis scripts and environment information should be documented alongside the files made available in this repository.


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


