# Diabetic retinopathy cross-domain and mobile evaluation

Selected supporting code and numerical records for **Cross-domain evaluation and on-device validation of deep learning models for diabetic retinopathy grading**, by Lu Qing and Khairunnisa Hasikin.

## Contents and images

Directories 01-05 provide preprocessing, training/evaluation notebooks, the original Android project, fixed image membership and retained predictions/timings. Directory 07 contains statistical analyses; 08 contains figure sources; 09 records external label correspondence. Directories 10-12 contain conversion, encoding and device verification. Directory 13 contains source confidence intervals; 15 records ablation provenance. Directory numbers need not be consecutive.

Obtain original APTOS 2019 and MESSIDOR-2 photographs from their providers under the applicable access conditions:
- https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
- https://www.adcis.net/en/third-party/messidor2/

Use the exact retained manifests; do not generate a new random partition to represent the recorded split. Configure the original Colab notebook input paths for your authorized data locations. The notebook training environment and the later verification environments are separate.

## Statistical recomputation

From this repository root, in a separate Python 3.12 environment:

```console
python -m pip install -r requirements_statistics.txt
python run_statistics.py
```

This uses retained per-image predictions and writes `07_statistics/recomputed`. It also checks the newly calculated threshold and calibration results. `08_figures/Figure2_original_20261015.png` supplies the retained upper panels used by the plotting program. Existing recomputation output must be moved or renamed before a new run.

For source confidence intervals, use another Python 3.12 environment:

```console
python -m pip install -r requirements_source_intervals.txt
python run_source_intervals.py
```

The source wrapper accepts either `13_source_intervals` or `13_source_intervals_20261019`; keep only one spelling in a checkout. The computation uses the recorded 2,000-resample method and frozen source thresholds. These entry scripts do not train models or run inference.

## Models and inference

The companion `Model_Archive_S1.zip` contains four PyTorch checkpoints and four float32 TFLite models. Its public archival link/DOI has not yet been added to this selected repository. Model identity information is in `MODEL_IDENTITIES.json`.

Extract the archive's `PeerJCS_models_20261019` folder beside this repository, then run:

```console
python connect_model_archive.py --models ../PeerJCS_models_20261019
```

This verifies all eight model SHA-256 identities and creates a new local configuration. Set authorized image roots in that configuration. Follow `10_offline_fidelity/reproduction/README.md` for the recorded inference environment and commands. The original 100-image Android app in 03 and the separate 218-image diagnostic in 11 have different documented input/build workflows.

## Status and reuse

This is a selected supporting-materials repository. No article acceptance, DOI or software/model licence is inferred here. Authors must add the actual archival identifier and confirmed licence before the intended public release. See the manuscript disclosure for AI-assisted editing, code and figure preparation; the authors remain responsible for the scientific content and permissions.
