"""Prepare the retained 218 inputs and TAR; no model or device execution."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent / '10_offline_fidelity/reproduction'))
from runtime_paths import load_config, load_manifest


def long_path(path):
    value = os.path.abspath(os.fspath(path))
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value = '\\\\?\\' + value
    return Path(value)


def sha(path):
    return hashlib.sha256(long_path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = long_path(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh local output directory.')
    # These imports are required only when actually preparing local inputs.
    import numpy as np
    from PIL import Image
    cfg = load_config(args.config)
    rows, _ = load_manifest(cfg, selected=True)
    recorded_dir = SOURCE.parent / '12_android_fidelity/completed_records'
    with long_path(recorded_dir / 'manifest.tsv').open(encoding='utf-8', newline='') as stream:
        recorded = {(r['dataset'], r['image_id']): r for r in csv.DictReader(stream, delimiter='\t')}
    inputs = output / 'fidelity_inputs'
    inputs.mkdir(parents=True)
    manifest = []
    for row in rows:
        dataset, image_id = row['dataset'], row['image_id']
        assert '\t' not in image_id and '\n' not in image_id
        expected = recorded[(dataset, image_id)]
        raw = long_path(row['raw_path'])
        processed = long_path(row['processed_path'])
        assert sha(raw) == row['raw_sha256'] == expected['raw_sha256']
        assert sha(processed) == row['processed_sha256']
        raw_relative = 'raw/' + dataset + '/' + image_id
        tensor_relative = 'tensor/' + dataset + '/' + image_id + '.bin'
        raw_target = inputs / raw_relative
        raw_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw, raw_target)
        with Image.open(processed) as image:
            rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
        assert rgb.shape == (224, 224, 3)
        tensor = np.ascontiguousarray(rgb.astype(np.float32) / np.float32(255))
        tensor_bytes = tensor.astype('<f4', copy=False).tobytes()
        tensor_sha = hashlib.sha256(tensor_bytes).hexdigest()
        assert tensor_sha == expected['tensor_sha256']
        tensor_target = inputs / tensor_relative
        tensor_target.parent.mkdir(parents=True, exist_ok=True)
        tensor_target.write_bytes(tensor_bytes)
        manifest.append([dataset, image_id, row['label'], raw_relative, tensor_relative,
                         row['raw_sha256'], tensor_sha, row['selection_reason']])
    text = io.StringIO(newline='')
    writer = csv.writer(text, delimiter='\t')
    writer.writerow(['dataset', 'image_id', 'label', 'raw_relative', 'tensor_relative',
                     'raw_sha256', 'tensor_sha256', 'selection_reason'])
    writer.writerows(manifest)
    manifest_bytes = text.getvalue().encode('utf-8')
    recorded_environment = json.loads(long_path(recorded_dir / 'environment.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(manifest_bytes).hexdigest() == recorded_environment['input_manifest_sha256']
    (inputs / 'manifest.tsv').write_bytes(manifest_bytes)
    archive = output / 'fidelity_inputs.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(inputs, arcname='fidelity_inputs')
    print(json.dumps({'images': len(rows), 'manifest_sha256': sha(inputs / 'manifest.tsv'),
                      'archive': str(archive), 'training_or_inference': False}, indent=2))


if __name__ == '__main__':
    main()
