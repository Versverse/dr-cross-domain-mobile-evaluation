# Reproducing the expanded desktop fidelity checks

These are portable adaptations of the inference and encoding-control programs actually used on 2026-10-16. Packaging and syntax checking these copies did not run new inference. The completed evidence remains in `../completed_csv`, `../executed_run_status.json`, and the independent audit files. The adapted programs write only into a newly named output folder and refuse to overwrite an existing output folder.

The inference computation is unchanged: a retained timm checkpoint is loaded strictly with `pretrained=False`; RGB float32 NHWC inputs in [0,1] enter the same ImageNet-normalizing PyTorch wrapper and saved float32 TFLite graph; logits, five probabilities, classes and source-frozen referral decisions are compared. The canonical retained-risk postprocessing that was applied separately after the original execution is integrated into the portable program. The fixed threshold JSON is verified by SHA-256 and is never refit.

## 1. Set up a separate environment

The executed inference used Python 3.12.0, PyTorch 2.8.0+cpu, torchvision 0.23.0+cpu, timm 1.0.15, NumPy 1.26.4, Pillow 11.3.0 and LiteRT 2.1.2. The encoding control used OpenCV 4.10.0.84. This environment is separate from the NumPy 2.3.5/pandas 3.0.1 statistical recomputation environment.

From this `reproduction` directory, create and activate a Python 3.12 environment, then install the CPU builds and remaining packages:

```text
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements_inference.txt
```

The reported experiment used the CPU runtime. A different operating system, numerical library or execution provider can change floating-point rounding, particularly at a source-selected threshold boundary. Save those outcomes; do not adjust the threshold to force agreement.

## 2. Configure retained artifacts and images

First extract the companion `Model_Archive_S1.zip` and run `python connect_model_archive.py --models ../PeerJCS_models_20261019` from the repository root. This verifies all eight model identities and creates `local_config.json` in this reproduction directory. Edit only the authorized data paths. Relative paths resolve from that JSON file; authorized absolute paths are also accepted. Do not modify the threshold values, expected hashes or frozen membership to accommodate a rerun.

- `data_roots.APTOS.raw`: original APTOS image folder, containing image filenames directly.
- `data_roots.APTOS.processed`: retained 733-image validation root, containing the grade folders such as `2_Moderate`.
- `data_roots.MESSIDOR2.raw`: original MESSIDOR-2 image folder, containing image filenames directly.
- `data_roots.MESSIDOR2.processed`: retained 1,744-image processed root, containing grade folders.
- Each model's `checkpoint` and `tflite`: the exact artifact whose identity is recorded in the configuration. All four PyTorch checkpoints and all four float32 TFLite files are supplied in the companion model archive. `connect_model_archive.py` configures their verified paths. The two Android TFLite assets are additionally retained in `03_android` for the original app build.

The selected and full CSVs contain exact frozen membership, row order, grades, selection reasons and the original available file hashes. Only `raw_path` and `processed_path` were replaced with paths relative to those data roots. `input_freeze.portable.json` records both original local and portable CSV identities. It does not claim that the path-sanitized CSV has the historical local CSV hash. The images themselves are not redistributed.

Check configuration, canonical thresholds, prediction-file hashes and frozen membership without importing model libraries or running inference:

```text
python check_configuration.py --config local_config.json
```

This metadata check lists absent weights and does not claim that private image files are present. Actual inference verifies processed-image identities for every selected input, model identities before loading, and raw-image identities when running the raw arm.

## 3. Run exactly the reported model coverage

Choose new output folders; the following names must not exist yet. The completed experiment used four threads for the full Lite0 run, and two threads for the full MobileNetV3 run, frozen MobileViT-S/ResNet-50 runs and raw-input runs. These are fidelity checks, not performance benchmarks.

```text
python run_fidelity.py --config local_config.json --models efficientnet_lite0 --threads 4 --output reruns/full_lite0
python run_fidelity.py --config local_config.json --models mobilenetv3_large --threads 2 --output reruns/full_mobilenet
python run_fidelity.py --config local_config.json --selected --models mobilevit_s resnet50 --threads 2 --output reruns/selected_218
python run_fidelity.py --config local_config.json --raw --models efficientnet_lite0 mobilenetv3_large --threads 2 --output reruns/raw_218
```

The first two commands each cover all 733 APTOS and 1,744 MESSIDOR-2 inputs. The third command covers exactly the frozen 108 APTOS and 110 MESSIDOR-2 inputs per model. The raw command covers that same set for the two Android candidates, saving processed PNGs for the encoding control. The 218-image set has five-grade and threshold-boundary coverage and is not a prevalence-representative accuracy sample. Do not report a smoke run or a partially written CSV as a completed full run. `run_status.json` is marked `complete` only after all requested models finish.

For a smaller environment check, add `--smoke` and use another new output folder. A smoke run selects one input per grade and domain and is not part of the reported coverage.

## 4. Reproduce the encoding control

After both raw-input models finish:

```text
python analyze_raw_encoding.py --config local_config.json --raw-results reruns/raw_218 --output reruns/encoding_control
```

This compares directly processed raw RGB with retained processed-file RGB and separately applies default OpenCV encoding in the retained file's original extension, followed by Pillow decoding. It writes a 218-row CSV and grouped JSON summary; it does not run another model or change inputs to the already saved inference records. PNG and JPEG results remain separate.

For the completed execution, 108 APTOS PNG and 74 MESSIDOR-2 PNG images matched directly; 36 MESSIDOR-2 JPEG images matched after the format roundtrip. The direct raw arm changed 5 Lite0 and 1 MobileNetV3 top-1 results on those JPEG images relative to retained processed-file predictions. This is input-encoding evidence, not a PyTorch-to-TFLite conversion error.

## Saved-output checks and audit provenance

From the package root, `python verify_saved_fidelity.py` recomputes the saved logits/probabilities and fixed-threshold comparisons without model execution or private images. It is the portable saved-output check.

`../independently_verify_runtime.py` is preserved as the source of the separately executed work-area audit. It expects that original work-area hierarchy, model files and images and is not a package-root entry point. Its PASS output is `../independent_runtime_audit.json`. The original audit re-decoded and hashed 2,477 retained images and checked 5,390 model-image pairs; the portable metadata check here has a smaller, explicitly named scope.

Real Android execution is documented separately. These desktop scripts do not install an APK, use ADB or assert any new device result.
