# APTOS source confidence-interval reproduction, 2026-10-19

## Scope

- Material: the retained 733-image APTOS prediction set for each of four trained checkpoints, canonical full-precision referral risks, and frozen source thresholds.
- Status: VERIFIED by execution, independent multiplicity-based metric checks, and a second deterministic execution.
- Reported outputs: Table 2 six source-domain confidence intervals for each model, and Table 5 source-domain sensitivity and specificity intervals. Point values, model predictions, full-precision thresholds, external results, and original files are unchanged.

This directory contains the executed 2026-10-19 confidence-interval computation from retained APTOS predictions and frozen source thresholds. It preserves the reported point estimates and uses 2,000 image-level bootstrap resamples with PCG64 seed 42. Input provenance, runtime details and output identities are recorded in the accompanying audit files.

## Reproduce

Install Python 3.10+ and NumPy 1.26+ in the environment of your choice, then run:

```console
python recompute_source_intervals.py --inputs inputs --output recomputed
```

The chosen output directory must not already exist. No GPU, model weights, images, PyTorch, sklearn, or internet connection is required. The executed environment used Python 3.12 and NumPy 1.26.4; exact version details and hashes are recorded in `executed/source_bootstrap_audit.json`.

The input CSV removes the source file's machine-specific image paths, copies every numerical string, and preserves every model's original row order. The canonical-risk CSV contains the exact saved full-precision source risks used by the 2026-10-16 frozen threshold contract. Their hash provenance is in `inputs/source_provenance.json`.

## Computation

- 2,000 ordinary image bootstrap draws of size 733 with replacement; no stratification.
- NumPy `Generator(PCG64(42))`, reset independently at the start of every model.
- Original retained CSV row order. All four models share exactly the same 733 image identifiers and order.
- Each resample supplies all six Table 2 metrics and the fixed source-threshold sensitivity/specificity calculations.
- 95% intervals are the 2.5th and 97.5th percentiles using `method="linear"`.
- Accuracy, macro F1 and QWK use the saved five-class argmax predictions. Table 2 sensitivity/specificity collapse those classes to grades 2-4 versus 0-1. Table 2 AUC uses the continuous canonical sum of probabilities for grades 2-4. These are intentionally distinct definitions.
- Table 5 uses canonical referable risk `>=` its unchanged full-precision MobileViT-S threshold. Thresholds are never refitted within resamples.
- All 24 Table 2 displayed point values and all eight Table 5 source displayed point values match the existing manuscript exactly. All 32 source interval cells have minor endpoint changes. No external interval is replaced.
- Original four grade distributions, 298 referable and 435 nonreferable images, and all frozen-threshold decision counts were checked.
- Independent verification uses a separately implemented multiplicity-weighted confusion matrix and score-bin AUC calculation for every replicate. All 8,000 six-metric draws agree within 2.23e-16; all 8,000 fixed-threshold operating-point draws agree exactly.

## Files

- `executed/table_cell_updates.json`: 32 old/new cell strings, 0-based DOCX table/row/column locations, and Chinese explanations. It records the numerical revision mapping and does not edit a DOCX.
- `executed/APTOS_image_draw_indices_seed42.npy`: all 2,000 actual draws, with 0-based image positions as little-endian int16.
- `executed/APTOS_image_order.csv`: map from saved draw positions to image identifiers and labels.
- `executed/*_source_bootstrap_draws.csv`: all six metric and eight operating-point values for every replicate and model.
- `executed/source_intervals_recomputed.csv`, `source_point_estimates_verified.csv`, `frozen_threshold_source_counts.csv`: full-precision results and checked counts.
- `executed/source_bootstrap_audit.json`: method, runtime, source/output hashes, and independent checks.
- `reproduction_and_integrity_check.json`: byte identity of numerical results after a second execution and unchanged original-file hashes.

## Interpretation and coverage

This work completes the source interval calculation and retention of the resampling records. It does not change model development, training, clinical endpoint definitions, reported point estimates, or referral thresholds. The manuscript retains the existing external examination-cluster results and all calibration analyses.
