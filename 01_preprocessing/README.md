# Preprocessing and fixed membership

`aptos_preprocessing_original.py` is the retained APTOS batch preprocessing source. Configure `SOURCE_CLASSIFIED_DIR` and a new `TARGET_PREPROCESSED_DIR` in a working copy before running it. The input contains the five grade directories named in the script. Use a new output directory to protect existing processed data.

The code uses grayscale threshold 15, the largest external contour, a centred bounding-box crop limited by image boundaries, `INTER_AREA` resizing to 224 x 224, and CLAHE on LAB L with `clipLimit=2.0` and `tileGridSize=(8,8)`. It saves through OpenCV with the original extension. Its dependencies are listed in `requirements.txt`; these are replay dependencies, not a reconstruction of an unrecorded historical environment.

`repack_from_manifests.py` is a later membership-verification/repacking utility. It reads an existing preprocessed ZIP and the retained manifests; it does not preprocess original images or generate a new random partition. Use `python repack_from_manifests.py --help` for its explicit input/output arguments. With no `--output`, it checks membership and writes only the requested report. A new archive is written only when a new `--output` is supplied.

`compare_preprocessing_sample.py` compares supplied raw/processed samples. The expanded raw-image and encoding controls are in `10_offline_fidelity`. The retained batch source and the later utilities have separate identities in `SOURCE_MANIFEST.json`.
