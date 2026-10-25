"""Reproduce APTOS intervals from retained predictions, without model inference.

Requires Python 3.10+ and NumPy 1.26+. Run beside inputs/, or supply --inputs.
All data inputs are read-only. Outputs are written to a new --output directory.
No threshold is selected here: the canonical 2026-10-16 JSON is an input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

MODELS = ['EfficientNet-Lite0', 'MobileNetV3-Large', 'MobileViT-S', 'ResNet-50']
KEYS = ['efficientnet_lite0', 'mobilenetv3_large', 'mobilevit_s', 'resnet50']
METRICS = ['accuracy', 'macro_f1', 'qwk', 'referable_sensitivity',
           'referable_specificity', 'referable_auc']
TARGETS = [90, 95, 98, 99]
SEED = 42
N_BOOT = 2000


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(p, fields, rows):
    with Path(p).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def all_metrics(y, pred, risk):
    """Six metrics using five-class argmax for the collapsed binary classes.

    Sensitivity/specificity collapse the five-class prediction to grades 2-4.
    AUC uses continuous sum(prob_2, prob_3, prob_4), not the collapsed class.
    """
    n = len(y)
    cm = np.bincount(5*y+pred, minlength=25).reshape(5, 5)
    row, col = cm.sum(axis=1), cm.sum(axis=0)
    diag = np.diag(cm)
    f1 = np.divide(2*diag, row+col, out=np.zeros(5, float), where=(row+col) > 0)
    weights = (np.arange(5)[:, None]-np.arange(5)[None, :])**2
    qwk = 1.0 - float((weights*cm).sum())/float((weights*np.outer(row, col)/n).sum())
    positive = y >= 2
    binary_pred = pred >= 2
    sensitivity = float(binary_pred[positive].mean())
    specificity = float((~binary_pred[~positive]).mean())
    order = np.argsort(risk, kind='stable')
    sorted_risk = risk[order]
    starts = np.r_[0, np.flatnonzero(sorted_risk[1:] != sorted_risk[:-1])+1]
    ends = np.r_[starts[1:], n]
    average_ranks = (starts+ends+1)/2.0
    ranks = np.empty(n, float)
    ranks[order] = np.repeat(average_ranks, ends-starts)
    positives = int(positive.sum())
    auc = (ranks[positive].sum()-positives*(positives+1)/2)/(positives*(n-positives))
    return np.array([diag.sum()/n, f1.mean(), qwk, sensitivity, specificity, auc])


def independent_weighted(y, pred, risk, counts):
    """Different implementation: multiplicity counts and fixed score bins."""
    cm = np.zeros((5, 5), dtype=np.int64)
    for label in range(5):
        for prediction in range(5):
            cm[label, prediction] = counts[(y == label) & (pred == prediction)].sum()
    total = int(counts.sum())
    f1 = []
    for label in range(5):
        tp = int(cm[label, label])
        denom = int(cm[label, :].sum()+cm[:, label].sum())
        f1.append(2*tp/denom if denom else 0.0)
    observed = 0.0
    expected = 0.0
    for a in range(5):
        for b in range(5):
            observed += (a-b)**2*int(cm[a, b])
            expected += (a-b)**2*int(cm[a, :].sum())*int(cm[:, b].sum())/total
    pos = y >= 2
    tp = int(counts[pos & (pred >= 2)].sum())
    tn = int(counts[~pos & (pred < 2)].sum())
    np_, nn_ = int(counts[pos].sum()), int(counts[~pos].sum())
    unique, inverse = np.unique(risk, return_inverse=True)
    pweights = np.bincount(inverse[pos], weights=counts[pos], minlength=len(unique))
    nweights = np.bincount(inverse[~pos], weights=counts[~pos], minlength=len(unique))
    concordance = np.sum(pweights*(np.cumsum(nweights)-0.5*nweights))
    return np.array([int(np.trace(cm))/total, sum(f1)/5,
                     1-observed/expected, tp/np_, tn/nn_, concordance/(np_*nn_)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=Path(__file__).resolve().parent/'inputs')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'executed')
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit('Output directory exists; choose a new --output directory.')
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    inputs = args.inputs
    required = ['aptos_source_predictions.csv', 'canonical_source_risks.csv',
                'frozen_thresholds_20261016.json', 'baseline_tables.json', 'source_provenance.json']
    for name in required:
        assert (inputs/name).is_file(), name
    provenance = json.loads((inputs/'source_provenance.json').read_text(encoding='utf-8'))
    for name, expected in provenance['prepared_input_sha256'].items():
        assert digest(inputs/name) == expected, f'Input hash mismatch: {name}'
    rows = load_csv(inputs/'aptos_source_predictions.csv')
    canonical = load_csv(inputs/'canonical_source_risks.csv')
    risk_map = {(r['display_model'], r['image_id']): float(r['risk_sum_prob234']) for r in canonical}
    frozen = json.loads((inputs/'frozen_thresholds_20261016.json').read_text(encoding='utf-8'))
    baseline = json.loads((inputs/'baseline_tables.json').read_text(encoding='utf-8'))
    updates, point_rows, interval_rows, counts_rows = [], [], [], []
    verification = []
    common_ids = None
    canonical_draws = None
    for mi, (model, key) in enumerate(zip(MODELS, KEYS)):
        sub = [r for r in rows if r['model'] == model]
        ids = [r['image_id'] for r in sub]
        assert len(ids) == len(set(ids)) == 733
        y = np.array([int(r['y_true']) for r in sub])
        pred = np.array([int(r['y_pred']) for r in sub])
        probs = np.array([[float(r[f'prob_{i}']) for i in range(5)] for r in sub])
        assert np.array_equal(pred, probs.argmax(axis=1))
        assert np.max(np.abs(probs.sum(axis=1)-1)) < 1e-5
        risk = np.array([risk_map[model, image_id] for image_id in ids])
        assert np.max(np.abs(risk-probs[:, 2:].sum(axis=1))) < 1e-14
        assert np.bincount(y, minlength=5).tolist() == [361, 74, 200, 39, 59]
        if common_ids is None:
            common_ids = ids
        assert common_ids == ids, 'All four models share source CSV order.'
        # Reset independently for every model; no seed searching or stream continuation.
        rng = np.random.Generator(np.random.PCG64(SEED))
        indices = np.stack([rng.integers(0, 733, size=733) for _ in range(N_BOOT)])
        if canonical_draws is None:
            canonical_draws = indices
            np.save(args.output/'APTOS_image_draw_indices_seed42.npy', indices.astype('<i2'))
            save_csv(args.output/'APTOS_image_order.csv', ['row_0based', 'image_id', 'y_true'],
                     [{'row_0based': i, 'image_id': image_id, 'y_true': int(y[i])} for i, image_id in enumerate(ids)])
        assert np.array_equal(indices, canonical_draws)
        original = all_metrics(y, pred, risk)
        direct = independent_weighted(y, pred, risk, np.ones(733, dtype=np.int64))
        assert np.allclose(original, direct, rtol=0, atol=2e-15)
        thresholds = np.array([frozen['thresholds'][key][f'source_sens_{t}'] for t in TARGETS])
        decisions = risk[:, None] >= thresholds[None, :]
        positive = y >= 2
        tp = decisions[positive].sum(axis=0)
        tn = (~decisions[~positive]).sum(axis=0)
        for j, target in enumerate(TARGETS):
            counts_rows.append({'model': model, 'source_target': target, 'threshold': thresholds[j],
                                'TP': int(tp[j]), 'FP': int(435-tn[j]), 'FN': int(298-tp[j]), 'TN': int(tn[j]),
                                'sensitivity': float(tp[j]/298), 'specificity': float(tn[j]/435)})
        metric_draws, operating_draws = [], []
        max_independent_error = 0.0
        for idx in indices:
            values = all_metrics(y[idx], pred[idx], risk[idx])
            metric_draws.append(values)
            yy = positive[idx]
            pp = decisions[idx]
            operating_values = np.r_[pp[yy].mean(axis=0), (~pp[~yy]).mean(axis=0)]
            operating_draws.append(operating_values)
            mult = np.bincount(idx, minlength=733)
            independent = independent_weighted(y, pred, risk, mult)
            error = float(np.max(np.abs(values-independent)))
            max_independent_error = max(max_independent_error, error)
            assert error < 3e-15, (model, error)
            independent_op = np.r_[np.sum(decisions*mult[:, None]*positive[:, None], axis=0)/np.sum(mult*positive),
                                  np.sum((~decisions)*mult[:, None]*(~positive[:, None]), axis=0)/np.sum(mult*(~positive))]
            assert np.array_equal(operating_values, independent_op)
        metric_draws = np.array(metric_draws)
        operating_draws = np.array(operating_draws)
        assert np.isfinite(metric_draws).all() and np.isfinite(operating_draws).all()
        ci = np.percentile(metric_draws, [2.5, 97.5], axis=0, method='linear')
        op_ci = np.percentile(operating_draws, [2.5, 97.5], axis=0, method='linear')
        for j, metric in enumerate(METRICS):
            pct = j in [0, 1, 3, 4]
            factor = 100 if pct else 1
            digits = 2 if pct else 4
            old = baseline['table_2'][mi+1][j+1]
            point_literal = old.split(' (')[0]
            assert f'{original[j]*factor:.{digits}f}' == point_literal, (model, metric, original[j], old)
            new = f'{point_literal} ({ci[0,j]*factor:.{digits}f}–{ci[1,j]*factor:.{digits}f})'
            updates.append({'table_index': 1, 'table_label': 'Table 2', 'row': mi+1, 'col': j+1,
                            'model': model, 'metric': metric, 'old': old, 'new': new,
                            'point_literal_unchanged': point_literal, 'changed': old != new,
                            'reason_zh': '用已保存的逐图预测统一重算并保存置信区间过程；原来的准确率等点值不变。'})
            point_rows.append({'model': model, 'metric': metric, 'point': float(original[j]),
                               'point_display_unchanged': point_literal})
            interval_rows.append({'table': 'Table 2', 'model': model, 'metric': metric,
                                  'point': float(original[j]), 'ci_low': float(ci[0, j]), 'ci_high': float(ci[1, j])})
        if model == 'MobileViT-S':
            for j, target in enumerate(TARGETS):
                for metric, col, k, point in [('sensitivity', 2, j, tp[j]/298), ('specificity', 3, j+4, tn[j]/435)]:
                    old = baseline['table_5'][j+1][col]
                    point_literal = old.split(' (')[0]
                    assert f'{point*100:.2f}' == point_literal
                    new = f'{point_literal} ({op_ci[0,k]*100:.2f}–{op_ci[1,k]*100:.2f})'
                    updates.append({'table_index': 4, 'table_label': 'Table 5', 'row': j+1, 'col': col,
                                    'model': model, 'metric': metric, 'source_target': target,
                                    'old': old, 'new': new, 'point_literal_unchanged': point_literal,
                                    'changed': old != new,
                                    'reason_zh': '使用完整精度的原固定阈值重算源域置信区间，并保存每次抽样；阈值和灵敏度/特异度点值均不变。'})
                    interval_rows.append({'table': 'Table 5', 'model': model, 'metric': f'{metric}_{target}',
                                          'point': float(point), 'ci_low': float(op_ci[0,k]), 'ci_high': float(op_ci[1,k])})
        draw_fields = ['draw_1based']+METRICS+[f'sensitivity_{t}' for t in TARGETS]+[f'specificity_{t}' for t in TARGETS]
        save_csv(args.output/f'{key}_source_bootstrap_draws.csv', draw_fields,
                 [{'draw_1based': i+1, **dict(zip(draw_fields[1:], np.r_[metric_draws[i], operating_draws[i]]))}
                  for i in range(N_BOOT)])
        verification.append({'model': model, 'n': 733, 'referable_n': 298, 'nonreferable_n': 435,
                             'max_independent_metric_absolute_error': max_independent_error,
                             'independent_all_2000_operating_draws_exact': True,
                             'six_point_display_values_unchanged': True})
        print(model, 'complete', flush=True)
    (args.output/'table_cell_updates.json').write_text(json.dumps(updates, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    save_csv(args.output/'source_point_estimates_verified.csv', ['model', 'metric', 'point', 'point_display_unchanged'], point_rows)
    save_csv(args.output/'source_intervals_recomputed.csv', ['table', 'model', 'metric', 'point', 'ci_low', 'ci_high'], interval_rows)
    save_csv(args.output/'frozen_threshold_source_counts.csv', ['model', 'source_target', 'threshold', 'TP', 'FP', 'FN', 'TN', 'sensitivity', 'specificity'], counts_rows)
    metadata = {'status': 'PASS', 'analysis_date': '2026-10-19',
                'scope': 'APTOS Table 2 six-metric intervals and Table 5 fixed-threshold sensitivity/specificity intervals; no training, inference, new threshold selection, or source-file changes.',
                'historical_program_search': provenance['historical_program_search'],
                'bootstrap': {'draws': N_BOOT, 'unit': 'image', 'sample_size': 733, 'with_replacement': True,
                              'rng': 'numpy.random.Generator(PCG64)', 'seed': SEED, 'reset': 'at start of each model',
                              'order': 'original retained combined prediction CSV order per model',
                              'shared_draws': 'same sampled indices used for all six metrics and four fixed-threshold operating points',
                              'ci': '2.5 and 97.5 percentiles; numpy.percentile(method="linear")',
                              'invalid_replicates': 0, 'stratified': False},
                'metric_definitions': {'grading': 'five-class argmax', 'macro_f1': 'unweighted mean over the fixed five grades; zero denominator returns zero',
                                       'qwk': 'five-grade quadratic-weighted Cohen kappa',
                                       'Table2_binary_decision': 'five-class argmax grades 2-4 positive (not a 0.5 risk threshold)',
                                       'AUC': 'rank AUC for canonical sum of probabilities grades 2-4; ties have half credit',
                                       'Table5_binary_decision': 'canonical referable risk >= frozen full-precision MobileViT-S threshold'},
                'point_estimates_and_thresholds_unchanged': True, 'external_intervals_unchanged': True,
                'source_cell_updates': len(updates), 'changed_cells': sum(u['changed'] for u in updates),
                'independent_verification': verification,
                'runtime': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()},
                'input_sha256': {name: digest(inputs/name) for name in required},
                'script_sha256': digest(__file__), 'seconds': time.perf_counter()-started}
    metadata['output_sha256'] = {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output/'source_bootstrap_audit.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('PASS:', len(updates), 'cells;', metadata['changed_cells'], 'changed CI cells', flush=True)


if __name__ == '__main__':
    main()
