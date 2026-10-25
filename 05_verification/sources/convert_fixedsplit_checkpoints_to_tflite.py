#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image
import tensorflow as tf
import timm
import torch
import torch.nn as nn


NUM_CLASSES = 5
INPUT_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

BASE_DIR = Path(r"[source]/fixed_DR")
OUTPUT_DIR = BASE_DIR / "dr_fixed_split_tflite_exports"

ONNX_DIR = OUTPUT_DIR / "onnx"
TFLITE_DIR = OUTPUT_DIR / "tflite"
SAVED_MODEL_DIR = OUTPUT_DIR / "saved_model"
VERIFY_DIR = OUTPUT_DIR / "verification"
LOG_DIR = OUTPUT_DIR / "logs"

VAL_ROOT_CANDIDATES = [
    BASE_DIR / "split_data_preprocessed" / "val",
    Path(r"split_data_preprocessed/val"),
    Path(r"split_data_preprocessed/val"),
    Path(r"split_data_preprocessed/val"),
]


@dataclass(frozen=True)
class ModelSpec:
    model: str
    timm_name: str
    checkpoint_path: Path
    onnx_name: str
    tflite_name: str
    verification_name: str


MODELS = [
    ModelSpec(
        model="efficientnet_lite0",
        timm_name="efficientnet_lite0",
        checkpoint_path=BASE_DIR / "best_fixedsplit_efficientnet_lite0.pth",
        onnx_name="aptos_fixedsplit_efficientnet_lite0_float32.onnx",
        tflite_name="aptos_fixedsplit_efficientnet_lite0_float32.tflite",
        verification_name="verification_efficientnet_lite0.csv",
    ),
    ModelSpec(
        model="mobilenetv3_large",
        timm_name="mobilenetv3_large_100",
        checkpoint_path=BASE_DIR / "best_fixedsplit_mobilenetv3_large.pth",
        onnx_name="aptos_fixedsplit_mobilenetv3_large_float32.onnx",
        tflite_name="aptos_fixedsplit_mobilenetv3_large_float32.tflite",
        verification_name="verification_mobilenetv3_large.csv",
    ),
    ModelSpec(
        model="mobilevit_s",
        timm_name="mobilevit_s",
        checkpoint_path=BASE_DIR / "best_fixedsplit_mobilevit_s.pth",
        onnx_name="aptos_fixedsplit_mobilevit_s_float32.onnx",
        tflite_name="aptos_fixedsplit_mobilevit_s_float32.tflite",
        verification_name="verification_mobilevit_s.csv",
    ),
    ModelSpec(
        model="resnet50",
        timm_name="resnet50",
        checkpoint_path=BASE_DIR / "best_fixedsplit_resnet50.pth",
        onnx_name="aptos_fixedsplit_resnet50_float32.onnx",
        tflite_name="aptos_fixedsplit_resnet50_float32.tflite",
        verification_name="verification_resnet50.csv",
    ),
]


class ExportWrapper(nn.Module):
    """NHWC RGB float32 [0,1] -> NCHW normalized -> logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = (x - self.mean) / self.std
        return self.model(x)


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        text = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(text, flush=True)
        self.handle.write(text + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def clean_state_dict(raw_state: object) -> dict[str, torch.Tensor]:
    if not isinstance(raw_state, dict):
        raise TypeError(f"Expected checkpoint dict/state_dict, got {type(raw_state)!r}")
    if "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]
    elif "model_state_dict" in raw_state:
        raw_state = raw_state["model_state_dict"]
    elif "model" in raw_state and isinstance(raw_state["model"], dict):
        raw_state = raw_state["model"]
    if not isinstance(raw_state, dict):
        raise TypeError(f"Expected nested state_dict, got {type(raw_state)!r}")
    return {
        str(key).removeprefix("module."): value
        for key, value in raw_state.items()
        if hasattr(value, "shape")
    }


def load_wrapped_model(spec: ModelSpec) -> ExportWrapper:
    model = timm.create_model(spec.timm_name, pretrained=False, num_classes=NUM_CLASSES)
    checkpoint = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(clean_state_dict(checkpoint), strict=True)
    wrapped = ExportWrapper(model)
    wrapped.eval()
    return wrapped


def find_val_root() -> Path:
    for candidate in VAL_ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate
    for base in [BASE_DIR, Path(r"[source]/Desktop"), Path.cwd()]:
        if not base.exists():
            continue
        matches = list(base.rglob("split_data_preprocessed"))
        for match in matches:
            val = match / "val"
            train = match / "train"
            if val.is_dir() and train.is_dir():
                return val
    raise FileNotFoundError("Could not locate split_data_preprocessed/val")


def label_from_folder(folder_name: str) -> int | None:
    digits = ""
    for char in folder_name:
        if char.isdigit():
            digits += char
        else:
            break
    if digits:
        label = int(digits)
        if 0 <= label < NUM_CLASSES:
            return label
    return None


def collect_validation_images(val_root: Path, limit: int = 50) -> list[tuple[Path, int | None, str]]:
    items: list[tuple[Path, int | None, str]] = []
    for class_dir in sorted([p for p in val_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        y_true = label_from_folder(class_dir.name)
        for path in sorted(class_dir.iterdir(), key=lambda p: p.name.lower()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, y_true, path.stem))
                if len(items) >= limit:
                    return items
    if not items:
        for path in sorted(val_root.rglob("*"), key=lambda p: str(p).lower()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, label_from_folder(path.parent.name), path.stem))
                if len(items) >= limit:
                    return items
    return items


def read_image_nhwc_rgb01(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    return (np.asarray(image, dtype=np.float32) / 255.0)[None, ...]


def export_onnx(model: nn.Module, onnx_path: Path) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, INPUT_SIZE, INPUT_SIZE, 3, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input_rgb_01"],
            output_names=["logits"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )


def convert_onnx_to_tflite(onnx_path: Path, saved_model_output: Path, tflite_path: Path) -> None:
    import onnx2tf

    if saved_model_output.exists():
        shutil.rmtree(saved_model_output)
    saved_model_output.mkdir(parents=True, exist_ok=True)
    tflite_path.parent.mkdir(parents=True, exist_ok=True)

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_output),
        copy_onnx_input_output_names_to_tflite=True,
        keep_shape_absolutely_input_names=["input_rgb_01"],
        keep_nwc_or_nhwc_or_ndhwc_input_names=["input_rgb_01"],
        non_verbose=True,
    )

    candidates = sorted(saved_model_output.glob("*float32.tflite"))
    if not candidates:
        candidates = sorted(saved_model_output.glob("*.tflite"))
    if not candidates:
        candidates = sorted(saved_model_output.rglob("*.tflite"))
    if not candidates:
        raise FileNotFoundError(f"No .tflite file generated in {saved_model_output}")
    tflite_path.write_bytes(candidates[0].read_bytes())


def make_tflite_interpreter(tflite_path: Path) -> tuple[tf.lite.Interpreter, dict[str, Any], dict[str, Any]]:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return interpreter, input_detail, output_detail


def run_tflite(
    interpreter: tf.lite.Interpreter,
    input_detail: dict[str, Any],
    output_detail: dict[str, Any],
    x: np.ndarray,
) -> np.ndarray:
    interpreter.set_tensor(input_detail["index"], x.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_detail["index"]).astype(np.float32)


def compare_logits(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return float(diff.mean()), float(diff.max())


def verify_random(model: nn.Module, interpreter, input_detail, output_detail) -> dict[str, float]:
    rng = np.random.default_rng(42)
    mean_diffs = []
    max_diffs = []
    matches = 0
    with torch.no_grad():
        for _ in range(10):
            x = rng.random((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
            pt = model(torch.from_numpy(x)).cpu().numpy().astype(np.float32)
            tf_logits = run_tflite(interpreter, input_detail, output_detail, x)
            mean_diff, max_diff = compare_logits(pt, tf_logits)
            mean_diffs.append(mean_diff)
            max_diffs.append(max_diff)
            matches += int(np.argmax(pt, axis=1)[0] == np.argmax(tf_logits, axis=1)[0])
    return {
        "random_mean_abs_diff": float(np.mean(mean_diffs)),
        "random_max_abs_diff": float(np.max(max_diffs)),
        "random_pred_match_rate": float(matches / 10),
    }


def verify_images(
    spec: ModelSpec,
    model: nn.Module,
    interpreter,
    input_detail,
    output_detail,
    image_items: list[tuple[Path, int | None, str]],
) -> dict[str, float]:
    rows = []
    mean_diffs = []
    max_diffs = []
    matches = 0

    with torch.no_grad():
        for path, y_true, image_id in image_items:
            x = read_image_nhwc_rgb01(path)
            pt = model(torch.from_numpy(x)).cpu().numpy().astype(np.float32)
            tf_logits = run_tflite(interpreter, input_detail, output_detail, x)
            mean_diff, max_diff = compare_logits(pt, tf_logits)
            pt_pred = int(np.argmax(pt, axis=1)[0])
            tf_pred = int(np.argmax(tf_logits, axis=1)[0])
            pred_match = pt_pred == tf_pred
            matches += int(pred_match)
            mean_diffs.append(mean_diff)
            max_diffs.append(max_diff)
            row = {
                "image_id": image_id,
                "image_path": str(path),
                "y_true": "" if y_true is None else y_true,
                "pytorch_pred": pt_pred,
                "tflite_pred": tf_pred,
                "pred_match": pred_match,
                "max_abs_diff": f"{max_diff:.10f}",
                "mean_abs_diff": f"{mean_diff:.10f}",
            }
            for index in range(NUM_CLASSES):
                row[f"pytorch_logit_{index}"] = f"{float(pt[0, index]):.10f}"
                row[f"tflite_logit_{index}"] = f"{float(tf_logits[0, index]):.10f}"
            rows.append(row)

    csv_path = VERIFY_DIR / spec.verification_name
    write_csv(csv_path, rows)
    return {
        "real_mean_abs_diff": float(np.mean(mean_diffs)),
        "real_max_abs_diff": float(np.max(max_diffs)),
        "real_pred_match_rate": float(matches / len(rows)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def inspect_tflite_io(tflite_path: Path) -> dict[str, Any]:
    interpreter, input_detail, output_detail = make_tflite_interpreter(tflite_path)
    return {
        "input_shape": list(map(int, input_detail["shape"])),
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "output_shape": list(map(int, output_detail["shape"])),
        "output_dtype": np.dtype(output_detail["dtype"]).name,
    }


def verification_status(row: dict[str, Any]) -> tuple[str, str]:
    notes = []
    if row.get("conversion_status") != "SUCCESS":
        return "FAILED", "conversion failed"
    if row.get("input_shape") != [1, INPUT_SIZE, INPUT_SIZE, 3]:
        notes.append("unexpected input shape")
    if row.get("input_dtype") != "float32":
        notes.append("unexpected input dtype")
    if row.get("output_shape") != [1, NUM_CLASSES]:
        notes.append("unexpected output shape")
    if row.get("output_dtype") != "float32":
        notes.append("unexpected output dtype")
    if float(row.get("random_pred_match_rate", 0.0)) < 1.0:
        notes.append("random prediction mismatch")
    if float(row.get("real_pred_match_rate", 0.0)) < 1.0:
        notes.append("real image prediction mismatch")
    if float(row.get("real_max_abs_diff", 999.0)) > 1e-3:
        notes.append("large real-image logit difference")
    if float(row.get("random_max_abs_diff", 999.0)) > 1e-3:
        notes.append("large random logit difference")
    return ("FAILED" if notes else "PASSED"), "; ".join(notes) if notes else "OK"


def adb_commands(tflite_paths: list[Path]) -> str:
    lines = ["ADB push commands:"]
    for path in tflite_paths:
        lines.append(f"adb push \"{path}\" /data/local/tmp/{path.name}")
    lines.append("")
    lines.append("benchmark_model commands:")
    for path in tflite_paths:
        lines.extend(
            [
                "adb shell /data/local/tmp/benchmark_model \\",
                f"  --graph=/data/local/tmp/{path.name} \\",
                "  --num_threads=4 \\",
                "  --use_xnnpack=true \\",
                "  --num_runs=50 \\",
                "  --warmup_runs=5",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    for directory in [ONNX_DIR, TFLITE_DIR, SAVED_MODEL_DIR, VERIFY_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    script_copy = OUTPUT_DIR / "convert_fixedsplit_checkpoints_to_tflite.py"
    if Path(__file__).resolve() != script_copy.resolve():
        script_copy.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    val_root = find_val_root()
    image_items = collect_validation_images(val_root, limit=50)
    if len(image_items) < 1:
        raise FileNotFoundError(f"No validation images found under {val_root}")

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Validation images: {len(image_items)} from {val_root}")

    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    final_tflite_paths: list[Path] = []
    created_at = datetime.now().isoformat(timespec="seconds")

    for spec in MODELS:
        logger = Logger(LOG_DIR / f"{spec.model}_conversion.log")
        onnx_path = ONNX_DIR / spec.onnx_name
        saved_model_path = SAVED_MODEL_DIR / spec.model
        tflite_path = TFLITE_DIR / spec.tflite_name
        row: dict[str, Any] = {
            "model": spec.model,
            "timm_name": spec.timm_name,
            "checkpoint_path": str(spec.checkpoint_path),
            "onnx_path": str(onnx_path),
            "tflite_path": str(tflite_path),
            "conversion_status": "FAILED",
            "verification_status": "FAILED",
            "notes": "",
        }
        try:
            logger.log(f"Starting {spec.model}")
            logger.log(f"Checkpoint: {spec.checkpoint_path}")
            if not spec.checkpoint_path.exists():
                raise FileNotFoundError(spec.checkpoint_path)

            model = load_wrapped_model(spec)
            logger.log("Loaded checkpoint strictly into timm model and ExportWrapper")

            logger.log(f"Exporting ONNX: {onnx_path}")
            export_onnx(model, onnx_path)
            logger.log(f"ONNX exported, size MB={onnx_path.stat().st_size / (1024 * 1024):.3f}")

            logger.log("Converting ONNX -> TensorFlow SavedModel/TFLite FLOAT32 via onnx2tf")
            convert_onnx_to_tflite(onnx_path, saved_model_path, tflite_path)
            logger.log(f"TFLite exported: {tflite_path}")

            interpreter, input_detail, output_detail = make_tflite_interpreter(tflite_path)
            io_info = {
                "input_shape": list(map(int, input_detail["shape"])),
                "input_dtype": np.dtype(input_detail["dtype"]).name,
                "output_shape": list(map(int, output_detail["shape"])),
                "output_dtype": np.dtype(output_detail["dtype"]).name,
            }
            logger.log(f"TFLite IO: {json.dumps(io_info)}")

            logger.log("Running random tensor verification")
            random_metrics = verify_random(model, interpreter, input_detail, output_detail)
            logger.log(f"Random metrics: {json.dumps(random_metrics)}")

            logger.log("Running real image verification")
            real_metrics = verify_images(spec, model, interpreter, input_detail, output_detail, image_items)
            logger.log(f"Real image metrics: {json.dumps(real_metrics)}")

            row.update(io_info)
            row.update(random_metrics)
            row.update(real_metrics)
            row["tflite_size_mb"] = round(tflite_path.stat().st_size / (1024 * 1024), 6)
            row["conversion_status"] = "SUCCESS"
            status, notes = verification_status(row)
            row["verification_status"] = status
            row["notes"] = notes
            final_tflite_paths.append(tflite_path)
            logger.log(f"Finished {spec.model}: {status} - {notes}")
        except Exception as exc:
            row["notes"] = repr(exc)
            logger.log("FAILED")
            logger.log(traceback.format_exc())
        finally:
            summary_rows.append(row)
            manifest_rows.append(
                {
                    "model": spec.model,
                    "timm_name": spec.timm_name,
                    "checkpoint_path": str(spec.checkpoint_path),
                    "checkpoint_exists": spec.checkpoint_path.exists(),
                    "onnx_path": str(onnx_path),
                    "onnx_exists": onnx_path.exists(),
                    "tflite_path": str(tflite_path),
                    "tflite_exists": tflite_path.exists(),
                    "tflite_size_mb": round(tflite_path.stat().st_size / (1024 * 1024), 6) if tflite_path.exists() else "",
                    "input_dtype": row.get("input_dtype", ""),
                    "input_shape": row.get("input_shape", ""),
                    "output_dtype": row.get("output_dtype", ""),
                    "output_shape": row.get("output_shape", ""),
                    "normalization_inside_model": True,
                    "softmax_inside_model": False,
                    "conversion_status": row.get("conversion_status", "FAILED"),
                    "verification_status": row.get("verification_status", "FAILED"),
                    "created_at": created_at,
                }
            )
            write_csv(VERIFY_DIR / "fixedsplit_tflite_verification_summary.csv", summary_rows)
            write_csv(VERIFY_DIR / "fixedsplit_tflite_export_manifest.csv", manifest_rows)
            logger.close()

    commands = adb_commands([TFLITE_DIR / spec.tflite_name for spec in MODELS])
    (VERIFY_DIR / "android_adb_benchmark_commands.txt").write_text(commands, encoding="utf-8")

    print("\nTFLite files:")
    for path in [TFLITE_DIR / spec.tflite_name for spec in MODELS]:
        size = f"{path.stat().st_size / (1024 * 1024):.3f} MB" if path.exists() else "MISSING"
        print(f"- {path} ({size})")

    print("\nVerification summary:")
    for row in summary_rows:
        print(
            row["model"],
            row.get("conversion_status"),
            row.get("verification_status"),
            "random_match=",
            row.get("random_pred_match_rate"),
            "real_match=",
            row.get("real_pred_match_rate"),
            "real_max_diff=",
            row.get("real_max_abs_diff"),
            "notes=",
            row.get("notes"),
        )

    print(f"\nExport manifest: {VERIFY_DIR / 'fixedsplit_tflite_export_manifest.csv'}")
    print(f"Verification summary CSV: {VERIFY_DIR / 'fixedsplit_tflite_verification_summary.csv'}")
    print(f"ADB commands: {VERIFY_DIR / 'android_adb_benchmark_commands.txt'}")
    print("\n" + commands)
    return 0 if all(row.get("verification_status") == "PASSED" for row in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
