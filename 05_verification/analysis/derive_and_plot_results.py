from pathlib import Path
import csv
import hashlib
import json
import statistics
from collections import Counter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import shutil

OUT = Path(__file__).resolve().parent.parent / 'results'
OUT.mkdir(exist_ok=True)
SOURCE = OUT.parent / 'inputs'
ROOT_WORK = OUT.parent
FIELDS = ['image_load_ms', 'opencv_preprocess_ms', 'input_conversion_ms',
          'tflite_inference_ms', 'output_decode_ms']
MODEL_NAMES = {'mobilenetv3_large': 'MobileNetV3-Large',
               'efficientnet_lite0': 'EfficientNet-Lite0'}

def provenance(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}

def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def parse_log(path):
    rows = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if 'image=' not in line or ', path=' not in line:
            continue
        payload = line.split('image=', 1)[1]
        row = dict(item.split('=', 1) for item in payload.split(', ')[1:])
        row['image_id'] = Path(row['path'].replace('\\', '/')).stem
        for field in FIELDS + ['end_to_end_ms']:
            row[field] = float(row[field])
        row['component_sum_ms'] = sum(row[field] for field in FIELDS)
        row['unallocated_time_ms'] = row['end_to_end_ms'] - row['component_sum_ms']
        rows.append(row)
    assert len(rows) == 100
    assert len({r['image_id'] for r in rows}) == 100
    assert set(Counter(r['truth'] for r in rows).values()) == {20}
    return rows

latency = {'unit': 'ms', 'source_fields': FIELDS,
           'arithmetic': 'Unallocated time = independently recorded end_to_end_ms minus sum of the five recorded component fields. No process attribution is made for the residual.',
           'aggregation': 'Each bar component is an arithmetic mean over 100 images. Shares are component means divided by measured total mean, not the mean of per-image percentages.',
           'device': 'Huawei DBR-W00; Android 12; QTI SM8250/Snapdragon 865-class; four TFLite threads; XNNPACK enabled',
           'sample': 'The same 100 APTOS tuning/validation images for both models, 20 per grade; five warm-up inference calls before the measured images.',
           'models': []}
idsets = []
for key, label in MODEL_NAMES.items():
    path = SOURCE / 'fixed_DR' / 'android_e2e_fixedsplit' / f'android_e2e_{key}_log.txt'
    rows = parse_log(path)
    idsets.append({r['image_id'] for r in rows})
    means = {field: statistics.mean(r[field] for r in rows) for field in FIELDS}
    total = statistics.mean(r['end_to_end_ms'] for r in rows)
    residuals = [r['unallocated_time_ms'] for r in rows]
    record = {'model': key, 'display_name': label, 'source': provenance(path),
              'n': len(rows), 'unique_image_ids': len(idsets[-1]),
              'grade_counts': dict(Counter(r['truth'] for r in rows)),
              'mean_components_ms': means, 'sum_component_means_ms': sum(means.values()),
              'mean_total_ms': total, 'mean_unallocated_time_ms': statistics.mean(residuals),
              'components_equal_total_per_image': all(abs(r) < 1e-9 for r in residuals),
              'residual_range_ms': [min(residuals), max(residuals)],
              'load_and_preprocessing_share_percent': 100 * (means['image_load_ms'] + means['opencv_preprocess_ms']) / total,
              'inference_share_percent': 100 * means['tflite_inference_ms'] / total,
              'row_addition_check_max_error_ms': max(abs(r['component_sum_ms'] + r['unallocated_time_ms'] - r['end_to_end_ms']) for r in rows)}
    latency['models'].append(record)
assert idsets[0] == idsets[1]
latency['same_image_id_set_for_both_models'] = True
(OUT / 'derived_latency.json').write_text(json.dumps(latency, ensure_ascii=False, indent=2), encoding='utf-8')

threshold_path = OUT / 'verified_full_precision_operating_points.csv'
op = csvrows(threshold_path)
low, high = op[0], op[-1]
assert float(low['target_APTOS_sensitivity']) == .9
assert float(high['target_APTOS_sensitivity']) == .99
threshold = {'source': provenance(threshold_path), 'unit': 'image',
             'source_threshold_from': float(low['full_precision_threshold']),
             'source_threshold_to': float(high['full_precision_threshold']),
             'positive_N': int(low['MESSIDOR2_positive_N']), 'negative_N': int(low['MESSIDOR2_negative_N']),
             'TP_from': int(low['MESSIDOR2_TP']), 'TP_to': int(high['MESSIDOR2_TP']),
             'FP_from': int(low['MESSIDOR2_negative_N']) - int(low['MESSIDOR2_TN']),
             'FP_to': int(high['MESSIDOR2_negative_N']) - int(high['MESSIDOR2_TN'])}
threshold['additional_TP'] = threshold['TP_to'] - threshold['TP_from']
threshold['additional_FP'] = threshold['FP_to'] - threshold['FP_from']

old_path = SOURCE / 'mobilenetv3_old_order_ablation_summary (1) (1).csv'
new_path = SOURCE / 'mobilenetv3_missing_strategy_summary_with_existing_full (1).csv'
baseline = next(r for r in csvrows(old_path) if r['variant_id'] == 'A_baseline_CE_noSampler_noAug')
new_rows = csvrows(new_path)
aug = next(r for r in new_rows if r['variant_id'] == 'B_CE_noSampler_Aug')
sampler = next(r for r in new_rows if r['variant_id'] == 'C_CE_sampler_Aug')
focal = next(r for r in new_rows if r['variant_id'] == 'D_existing_Focal_sampler_Aug')
ablation = {'sources': [provenance(old_path), provenance(new_path)],
            'comparison': 'Recorded single-run configurations on the same 733-image APTOS tuning/validation subset.',
            'baseline_to_CE_augmentation': {
                'macro_f1_percentage_point_difference': 100 * (float(aug['macro_f1']) - float(baseline['macro_f1'])),
                'Severe_recall_percentage_point_difference': 100 * (float(aug['class3_recall']) - float(baseline['class3_recall'])),
                'Proliferative_recall_percentage_point_difference': 100 * (float(aug['class4_recall']) - float(baseline['class4_recall']))},
            'CE_aug_sampler_to_focal_aug_sampler': {
                'accuracy_percentage_point_difference': 100 * (float(focal['accuracy']) - float(sampler['accuracy'])),
                'macro_f1_percentage_point_difference': 100 * (float(focal['macro_f1']) - float(sampler['macro_f1'])),
                'QWK_difference': float(focal['qwk']) - float(sampler['qwk']),
                'referable_AUC_difference': float(focal['referable_auc']) - float(sampler['referable_auc'])}}

conf_path = SOURCE / 'MESSIDOR-2_mobilevit_confusion.csv'
with conf_path.open(encoding='utf-8-sig', newline='') as stream:
    conf = list(csv.reader(stream))[1:]
conf = [[int(n) for n in row] for row in conf]
moderate_fn = sum(conf[2][:2])
all_fn = sum(sum(row[:2]) for row in conf[2:])
gaps={}
gap_sources=[]
for dataset,filename in [('APTOS','aptos_val_all_predictions_combined.csv'),('MESSIDOR2','messidor2_all_predictions_combined.csv')]:
    p=SOURCE/filename
    records=[r for r in csvrows(p) if r['model']=='MobileViT-S']
    prevalence=statistics.mean(int(r['y_true'])>=2 for r in records)
    mean_risk=statistics.mean(sum(float(r['prob_'+str(i)]) for i in [2,3,4]) for r in records)
    gaps[dataset]=100*(prevalence-mean_risk)
    gap_sources.append(provenance(p))
derived = {'threshold_transfer': threshold, 'ablation': ablation,
           'referral_error_structure': {'source': provenance(conf_path), 'Moderate_FN': moderate_fn,
                                       'all_referable_FN': all_fn, 'Moderate_share_percent': 100 * moderate_fn / all_fn},
           'risk_gap': {'unit':'percentage points','calculation':'Observed binary prevalence minus mean summed class-2-to-4 risk, using unrounded retained values.','sources':gap_sources,'values':gaps}}
(OUT / 'derived_results.json').write_text(json.dumps(derived, ensure_ascii=False, indent=2), encoding='utf-8')

# Publication figure: vector PDF, with component means from the independently parsed logs.
pdf = OUT / 'figure3_android_latency.pdf'
W, H = 648, 378
c = canvas.Canvas(str(pdf), pagesize=(W, H))
c.setTitle('Android latency composition for two diabetic retinopathy models')
navy, muted, grid = HexColor('#15293B'), HexColor('#536779'), HexColor('#D8E0E7')
colors = [HexColor(x) for x in ['#25608A', '#3F947E', '#E5A955', '#C65A61', '#7C64A2', '#C6CCD3']]
legend = ['Image loading', 'OpenCV preprocessing', 'Input preparation', 'TFLite inference', 'Output decoding', 'Unallocated time*']
c.setFillColor(navy); c.setFont('Helvetica-Bold', 15)
c.drawString(24, 351, 'Android latency composition')
c.setFont('Helvetica', 9.5); c.setFillColor(muted)
c.drawString(24, 332, 'Same 100 APTOS images (20 per grade) | Huawei DBR-W00 | Android 12 | 4 threads + XNNPACK')
for index, (label, color) in enumerate(zip(legend, colors)):
    x = [25, 243, 464][index % 3]
    y = 302 - (index // 3) * 19
    c.setFillColor(color); c.rect(x, y-3, 9, 9, stroke=0, fill=1)
    c.setFillColor(navy); c.setFont('Helvetica', 9.5); c.drawString(x+14, y-1, label)

x0, x1, axis_y, scale = 162, 617, 125, 455/140
for tick in range(0, 141, 20):
    x = x0+tick*scale
    c.setStrokeColor(grid); c.setLineWidth(.45); c.line(x, axis_y, x, 247)
    c.setFillColor(muted); c.setFont('Helvetica', 9); c.drawCentredString(x, 110, str(tick))
c.setStrokeColor(navy); c.setLineWidth(.6); c.line(x0, axis_y, x1, axis_y)
c.setFillColor(navy); c.setFont('Helvetica', 10); c.drawCentredString((x0+x1)/2, 91, 'Mean latency per image (ms)')
for record, y in zip(latency['models'], [215, 157]):
    c.setFillColor(navy); c.setFont('Helvetica-Bold', 10)
    c.drawRightString(x0-12, y+7, record['display_name'])
    values = [record['mean_components_ms'][key] for key in FIELDS] + [record['mean_unallocated_time_ms']]
    x = x0
    for i, value in enumerate(values):
        width = value*scale
        c.setFillColor(colors[i]); c.rect(x, y, width, 25, fill=1, stroke=0)
        if width > 42:
            c.setFillColor(HexColor('#FFFFFF')); c.setFont('Helvetica-Bold', 9.5)
            c.drawCentredString(x+width/2, y+8.5, f'{value:.2f}')
        x += width
    c.setFillColor(navy); c.setFont('Helvetica-Bold', 10)
    c.drawString(x+5, y+8, f"{record['mean_total_ms']:.2f} ms")

c.setFillColor(navy); c.setFont('Helvetica', 10)
c.drawString(24, 64, 'Loading + preprocessing: 83.45% of total time for MobileNetV3-Large; 77.21% for EfficientNet-Lite0.')
c.setFillColor(muted); c.setFont('Helvetica', 8.5)
c.drawString(24, 43, '*Unallocated time = measured total minus the five recorded components (0.14 / 0.15 ms).')
c.drawString(24, 27, 'Input preparation: 3.63 / 3.61 ms; output decoding: 0.04 / 0.04 ms (MobileNetV3-Large / EfficientNet-Lite0).')
c.showPage(); c.save()

# Optional PNG export; the vector PDF and numeric results are portable outputs.
poppler = shutil.which('pdftoppm')
if poppler:
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(str(pdf), dpi=450, poppler_path=str(Path(poppler).parent))
        pages[0].save(OUT/'figure3_android_latency.png', dpi=(450, 450))
    except ImportError:
        print('Vector PDF written; install pdf2image to export PNG.')
else:
    print('Vector PDF written; optional PNG export requires Poppler.')

caption = ('Figure 3. Components of mean Android end-to-end latency for MobileNetV3-Large and EfficientNet-Lite0. '
           'Both models used the same 100 APTOS tuning/validation images (20 per grade), after five warm-up inference calls, '
           'on a Huawei DBR-W00 running Android 12 with four TFLite threads and XNNPACK. '
           'Segments are arithmetic means of the recorded image_load_ms, opencv_preprocess_ms, input_conversion_ms, '
           'tflite_inference_ms, and output_decode_ms fields. The small grey segment is the calculated difference '
           'between independently recorded end_to_end_ms and the sum of the five components. '
           'Bar totals are the measured mean end-to-end latencies. Loading and preprocessing accounted for '
           '83.45% and 77.21% of the total, respectively.')
(OUT/'figure3_caption.txt').write_text(caption, encoding='utf-8')
print(json.dumps({'figure_pdf':str(pdf), 'figure_png':str(OUT/'figure3_android_latency.png'),
                  'same_100_images': latency['same_image_id_set_for_both_models'],
                  'totals': [m['mean_total_ms'] for m in latency['models']],
                  'residuals': [m['mean_unallocated_time_ms'] for m in latency['models']]}, ensure_ascii=False, indent=2))
