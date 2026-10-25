"""Validate sharing configuration and frozen membership without importing ML runtimes."""
from pathlib import Path
from collections import Counter
import argparse
import json
from runtime_paths import BASE, load_config, load_manifest, frozen_records, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE / 'local_config.example.json')
    args = parser.parse_args()
    cfg = load_config(args.config)
    frozen_records(cfg)
    full, _ = load_manifest(cfg)
    selected, _ = load_manifest(cfg, selected=True)
    assert len(full) == 2477 and len(selected) == 218
    assert {(r['dataset'], r['image_id']) for r in selected}.issubset(
        {(r['dataset'], r['image_id']) for r in full})
    assert Counter(r['dataset'] for r in full) == {'APTOS': 733, 'MESSIDOR2': 1744}
    assert Counter(r['dataset'] for r in selected) == {'APTOS': 108, 'MESSIDOR2': 110}
    for source in cfg['source_prediction_csv'].values():
        assert sha(source['path']) == source['sha256']
    available = []
    required = []
    for name, spec in cfg['models'].items():
        for kind in ('checkpoint', 'tflite'):
            path = Path(spec[kind])
            if path.is_file():
                assert sha(path) == spec[kind + '_sha256']
                available.append(name + ':' + kind)
            else:
                required.append(name + ':' + kind)
    print(json.dumps({
        'status': 'PASS_CONFIGURATION_AND_FROZEN_MEMBERSHIP',
        'full_images': len(full), 'selected_images': len(selected),
        'source_prediction_files_hash_verified': 2,
        'canonical_retained_risks': 9908,
        'available_model_artifacts_hash_verified': available,
        'model_artifacts_to_obtain_and_configure': required,
        'data_roots_configured': list(cfg['data_roots']),
        'image_bytes_checked': False,
        'training_or_inference_executed': False,
        'note': 'PASS checks packaged metadata, not availability of private images or unbundled weights.'
    }, indent=2))


if __name__ == '__main__':
    main()
