# Completed Android paired fidelity experiment — 2026-10-16

The experiment was executed on the Huawei DBR-W00, Android 12/API 31, using the separate native Java validation application, TensorFlow Lite 2.16.1, OpenCV 4.10.0, four threads, requested XNNPACK and five warm-up inferences per model. The recorded device state before/after was 100% battery, external power connected, power saving disabled and thermal status 0; battery temperature was 25.8/28.8 °C. These are actual device records. The debug-build times are diagnostic measurements and do not replace the historical July 7 performance tables.

`completed_records` preserves the two JSONL outputs, exact 218-image manifest, runtime configuration, environment and completion marker from the successful run. It contains 2 models × 218 images × 3 arms = 1,308 records. The same fixed 108 APTOS and 110 MESSIDOR-2 images were used for every arm. Only this successful run is included. No photographs, input tensor binaries, APK, signing key or failed-run output is redistributed here.

- `fixed_tensor`: exact retained preprocessed-file RGB float32 input, isolating the inference-runtime comparison.
- `raw_pipeline`: device decoding and the study's crop/resize/CLAHE pipeline, without an offline JPEG-saving step.
- `raw_encoding_matched`: the raw arm with a separate default OpenCV JPEG encode/decode control for JPEG inputs. This is an input-encoding control; it does not assume that device and desktop decoders produce identical pixels.

`comparison_records` contains the main per-image comparison CSVs and summary. Its raw-arm desktop reference is the separately executed desktop raw pipeline. `independent_records` contains the independent saved-output audit and its two comparison CSVs. Its raw-arm comparison uses the Android fixed-tensor arm. These reference choices explain why some columns in the two analyses differ; the underlying run records are unchanged.

## Portable saved-output verification

From the package root, run Python 3.12 with the standard library only:

```text
python verify_saved_android.py
```

The program was actually run during packaging and passed. It checks completion status, record/model/configuration identities, unique coverage of all 1,308 expected combinations, scalar softmax reconstruction, referable-risk arithmetic, exact frozen thresholds and decisions, and fixed-tensor identities against the desktop records. It compares fixed-tensor class/probability/decision results with `10_offline_fidelity/completed_csv`, then recomputes raw-arm comparisons against each image's Android fixed-tensor output. Generated summaries are written to `12_android_fidelity/recomputed`; completed records are never edited. Its summaries also reproduce the previously saved independent audit.

PASS means the saved identities, arithmetic and summaries agree. The program does not require images, weights or an APK and does not run model inference or a device experiment. Genuine output differences remain in the summaries and are not treated as verification failures. Decision comparisons use `risk >= threshold` at full precision, without a tolerance around the boundary; a small tolerance is used only when verifying reconstructed floating-point arithmetic.

## Completed results

All 436 fixed-tensor Android/desktop LiteRT top-1 comparisons agreed. Fixed-tensor external referral decisions agreed at all five checked thresholds. Two APTOS fixed-tensor decisions differed at the source-98% threshold, one for each model; those boundary records are retained.

Against the Android fixed-tensor arm, the raw pipeline changed 5/110 external top-1 results for Lite0 and 1/110 for MobileNetV3; the encoding-control arm changed 1/110 and 0/110, respectively. Both raw arms retained all APTOS top-1 classifications. The full probability and threshold comparisons are supplied rather than being reduced to class agreement alone.

These balanced, boundary-enriched inputs test consistency of the conversion and complete input pipeline. They do not provide a new prevalence-representative estimate of clinical accuracy.
