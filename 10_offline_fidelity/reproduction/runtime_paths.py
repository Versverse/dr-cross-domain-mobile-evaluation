"""Portable path and frozen-identity helpers; no model imports or inference."""
from pathlib import Path
import csv
import hashlib
import json

BASE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve(directory, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path(directory) / path).resolve()


def load_config(filename):
    filename = Path(filename).resolve()
    if not filename.is_file():
        raise FileNotFoundError(
            'Copy local_config.example.json to local_config.json and configure '
            'the authorized image and retained weight locations first.'
        )
    cfg = json.loads(filename.read_text(encoding='utf-8'))
    cfg['_config_dir'] = str(filename.parent)
    for spec in cfg['models'].values():
        for key in ('checkpoint', 'tflite'):
            spec[key] = str(resolve(filename.parent, spec[key]))
    for spec in cfg['source_prediction_csv'].values():
        spec['path'] = str(resolve(filename.parent, spec['path']))
    for roots in cfg['data_roots'].values():
        for key in ('raw', 'processed'):
            roots[key] = str(resolve(filename.parent, roots[key]))
    for key in ('frozen_thresholds', 'canonical_retained_risks', 'input_freeze'):
        cfg[key] = str(resolve(filename.parent, cfg[key]))
    return cfg


def frozen_records(cfg):
    assert sha(cfg['frozen_thresholds']) == cfg['frozen_thresholds_sha256']
    assert sha(cfg['canonical_retained_risks']) == cfg['canonical_retained_risks_sha256']
    frozen = json.loads(Path(cfg['frozen_thresholds']).read_text(encoding='utf-8'))
    thresholds = frozen['thresholds']
    assert cfg['thresholds'] == thresholds, 'Do not alter source-frozen thresholds.'
    with Path(cfg['canonical_retained_risks']).open(encoding='utf-8', newline='') as stream:
        risks = {(r['dataset'], r['model'], r['image_id']): float(r['risk_sum_prob234'])
                 for r in csv.DictReader(stream)}
    assert len(risks) == 9908
    return thresholds, risks


def load_manifest(cfg, selected=False):
    freeze = json.loads(Path(cfg['input_freeze']).read_text(encoding='utf-8'))
    key = 'android' if selected else 'full'
    path = resolve(cfg['_config_dir'], cfg[key + '_manifest'])
    assert sha(path) == freeze['portable_manifest_sha256'][key]
    with path.open(encoding='utf-8', newline='') as stream:
        items = list(csv.DictReader(stream))
    assert len(items) == freeze[key + '_N']
    assert len({(r['dataset'], r['image_id']) for r in items}) == len(items)
    for row in items:
        for field, root_key in (('raw_path', 'raw'), ('processed_path', 'processed')):
            relative = Path(row[field])
            assert not relative.is_absolute() and '..' not in relative.parts
            row[field] = str(resolve(cfg['data_roots'][row['dataset']][root_key], relative))
    return items, freeze['portable_manifest_sha256'][key]
