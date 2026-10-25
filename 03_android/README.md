# Original 100-image Android app

`AptosEdgeTest` is the retained Gradle project for the original Android demonstration and end-to-end timing. It includes Kotlin sources, resources, Gradle Wrapper/configuration and two float32 TFLite model assets.

Open `AptosEdgeTest` in Android Studio and allow the configured Gradle dependencies to resolve. Its configuration specifies TensorFlow Lite 2.16.1 and OpenCV 4.10.0. The retained runtime requests four threads and XNNPACK, with five warm-up calls. These settings identify the original source snapshot; a new build has its own APK identity.

The photographs are obtained separately. Reconstruct the 100 raw APTOS assets from `android_100_image_manifest.csv`, checking their SHA-256 values, and place them at:

```text
AptosEdgeTest/app/src/main/assets/val100/{asset_path}
```

`asset_path` already includes the grade directory. The app also accepts the `APTOS_val100_original` asset-root name. Keep these photograph directories private; `.gitignore` excludes them. The manifest records 20 images per grade.

This original 100-image app and its historical timing evidence are separate from the later 218-image matched-tensor/raw-image validation program. For that program, use `../11_android_validation_source/README.md` and `../12_android_fidelity`.
