"""Write a new local model configuration after verifying complete archived weights."""
from pathlib import Path
import os as _package_os
import argparse
import hashlib
import json
import os


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(('\\\\?\\' if _package_os.name == 'nt' and not str(Path(__file__).resolve().parent).startswith('\\\\?\\') else '') + str(Path(__file__).resolve().parent))
    directory = root / '10_offline_fidelity/reproduction'
    output = (args.output or directory / 'local_config.json').resolve()
    if output.exists():
        raise FileExistsError('Choose a new --output path; an existing local configuration is never overwritten.')
    models = args.models.resolve()
    config = json.loads((directory / 'local_config.example.json').read_text(encoding='utf-8'))
    identities = json.loads((root / 'MODEL_IDENTITIES.json').read_text(encoding='utf-8'))
    for spec in identities:
        for kind in ['checkpoint', 'tflite']:
            path = models / spec[kind + '_archive_path']
            if sha(path) != spec[kind + '_sha256']:
                raise ValueError('Model identity mismatch: ' + path.name)
            config['models'][spec['model']][kind] = os.path.relpath(path, output.parent).replace(os.sep, '/')
    # Preserve other example-relative locations if the caller chooses an output elsewhere.
    if output.parent != directory:
        for field in ['frozen_thresholds', 'canonical_retained_risks', 'input_freeze', 'full_manifest', 'android_manifest']:
            config[field] = os.path.relpath((directory / config[field]).resolve(), output.parent).replace(os.sep, '/')
        for entry in config['source_prediction_csv'].values():
            entry['path'] = os.path.relpath((directory / entry['path']).resolve(), output.parent).replace(os.sep, '/')
    config['status'] = 'model_identities_verified_local_configuration_requires_authorized_data_roots'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Eight model identities verified. New configuration written. Configure authorized data_roots before inference.')
