# Native Android validation source and rebuild instructions

`FidelityActivity.java` and `AndroidManifest.xml` are exact copies of the final source used for the completed 2026-10-16 experiment. Inputs are read from the application's private `files/fidelity_inputs` directory. `runtime_config.json` is copied from the successful device run and contains the two preserved model identities and exact frozen thresholds. The completed outputs are in `../12_android_fidelity`.

`build_integrity.json` identifies the actual SDK build and separately links its subsequent completed device execution in `actual_device_run`. The executed APK identity is `297adba97c725c397868d49c530c12ca7e3f8bf6e5e6225fb866d50393fbf2ee`, also recorded directly in the successful run's `environment.json`. The build-stage record and later device evidence have separate scopes.

The original signing key and APK are not redistributed. A local rebuild creates a new debug key, and the signed APK's hash will differ. The recorded hash identifies the APK that actually ran; it is not a promise of a byte-identical signed rebuild. Keep newly generated debug keys and image-containing input archives out of a public review package.

## Windows SDK rebuild

The portable `build_manual.py` retains the executed SDK-only build sequence: javac, D8, aapt2, zipalign and apksigner. It does not require Gradle or AndroidX. This sharing copy was syntax-checked but was not rebuilt during packaging.

Install/configure an Android SDK with Build Tools **36.0.0** and platform **android-36.1**, plus a Java installation with javac/keytool (the executed build used Android Studio's bundled Java runtime). Set these PowerShell environment variables to your actual local directories:

```powershell
$env:ANDROID_SDK_ROOT = 'D:\Android\Sdk'
$env:JAVA_HOME = 'D:\Java\jdk'
$env:FIDELITY_AAR_ROOT = 'D:\ResearchDependencies\aar'
$env:FIDELITY_BUILD_ROOT = 'D:\ResearchRuns\fidelity_build_01'
python build_manual.py
```

`FIDELITY_BUILD_ROOT` must not exist yet. The script writes new build outputs and a new local signing key there. It reads the two TFLite assets from the review package's `03_android/AptosEdgeTest/app/src/main/assets` directory and verifies their recorded hashes.

`FIDELITY_AAR_ROOT` must contain exactly one copy of each of these artifacts, directly or in subfolders. Obtain them from the corresponding original Maven repository; their exact recorded SHA-256 identities are in `build_integrity.json` and are checked before use:

| Maven artifact | AAR filename |
| --- | --- |
| `org.opencv:opencv:4.10.0` | `opencv-4.10.0.aar` |
| `org.tensorflow:tensorflow-lite-api:2.16.1` | `tensorflow-lite-api-2.16.1.aar` |
| `org.tensorflow:tensorflow-lite:2.16.1` | `tensorflow-lite-2.16.1.aar` |

The script packages only arm64-v8a native libraries. The validation manifest uses minSdk 26, targetSdk 32, and a separate debuggable package `com.example.aptosedgetest.fidelity20261016`. Its measurements are diagnostic fidelity timings, not a replacement benchmark.

## Prepare the exact 218-image inputs

Configure `../10_offline_fidelity/reproduction/local_config.json` as described in that folder's README, including the authorized raw/processed image roots. In the separate inference environment with NumPy 1.26.4 and Pillow 11.3.0, run:

```text
python prepare_device_inputs.py --config ../10_offline_fidelity/reproduction/local_config.json --output D:/ResearchRuns/fidelity_inputs_01
```

This helper was syntax-checked but not run during packaging. When invoked, it verifies the retained image identities, reconstructs the same little-endian float32 RGB tensors, checks them against the completed run manifest, and creates `fidelity_inputs.tar` with a top-level `fidelity_inputs` directory. It does not train a model or run inference. The raw photographs and tensors are generated only in your chosen local output folder. The new manifest must reproduce the recorded `4da3680c51f2d39080046753248bb773a1eb5d707a4848dff58928a4eb07d56d` identity.

## Install, stage, run and retrieve

Connect an authorized arm64 Android device with USB debugging enabled. Read its identifier from `adb devices -l`; substitute that identifier for `DEVICE_ID` below. Keep the device unlocked, power saving disabled, and its thermal/battery state recorded. The completed study run used the recorded Huawei device; another device is a new execution.

```powershell
& "$env:ANDROID_SDK_ROOT/platform-tools/adb.exe" devices -l
& "$env:ANDROID_SDK_ROOT/platform-tools/adb.exe" -s DEVICE_ID install -r "$env:FIDELITY_BUILD_ROOT/DR_Fidelity_native_debug_20261016.apk"
python device_archive_io.py --serial DEVICE_ID push --archive D:/ResearchRuns/fidelity_inputs_01/fidelity_inputs.tar
& "$env:ANDROID_SDK_ROOT/platform-tools/adb.exe" -s DEVICE_ID shell am start -W -n 'com.example.aptosedgetest.fidelity20261016/.FidelityActivity'
```

Press **Run frozen 218-image experiment** once. The input helper streams binary TAR data through `adb exec-in run-as com.example.aptosedgetest.fidelity20261016 tar -xf - -C files`; it uses private internal storage, not the restricted `/sdcard/Android/data` path.

After the application reports completion:

```text
python device_archive_io.py --serial DEVICE_ID pull --output D:/ResearchRuns/android_results_01.tar
```

This retrieves binary bytes using `adb exec-out run-as com.example.aptosedgetest.fidelity20261016 tar -cf - -C files fidelity_results`. Use a TAR extractor to inspect the local archive. Each new run has its own directory. Check `COMPLETE.txt`, environment, manifest and model identities before interpreting a run; a `FAILED.txt` run or a partial JSONL is not completed evidence. Preserve new device results separately from the recorded run supplied here. A new signing key may require a separate device or removal of that device's previous experimental package by its owner; the rebuild script does not uninstall anything.

For an immediate check requiring no Android SDK, images, models or APK execution, run `python verify_saved_android.py` from the package root. That portable saved-output check has already been executed successfully.
