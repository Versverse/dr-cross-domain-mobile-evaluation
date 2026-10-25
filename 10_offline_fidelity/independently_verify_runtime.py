"""Independent complete-record arithmetic audit, without model/device execution."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,math,sys
import numpy as np
from PIL import Image
B=Path(__file__).resolve().parent;STAT=B.parent;RUN=STAT.parent/'runtime_validation'
FULL=RUN/'full_results'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows,keys=None):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys or list(rows[0]));w.writeheader();w.writerows(rows)
def stable_softmax(row,prefix):
    # Scalar math.exp/math.fsum independently of the producer's NumPy routine.
    v=[float(row[f'{prefix}_logit_{k}']) for k in range(5)];mx=max(v)
    e=[math.exp(z-mx) for z in v];den=math.fsum(e)
    return v,[z/den for z in e]
frozen=json.loads((STAT/'frozen_thresholds_20261016.json').read_text(encoding='utf-8'))
FROZEN_SHA=sha(STAT/'frozen_thresholds_20261016.json')
assert FROZEN_SHA=='6874b1e08da06cdc12417c974f437be680548a7a4744f80d9b5004732000d894'
thresholds=frozen['thresholds']
canonical={(r['dataset'],r['model'],r['image_id']):float(r['risk_sum_prob234']) for r in readcsv(STAT/'canonical_retained_risks_20261016.csv')}
assert len(canonical)==9908
freeze=json.loads((RUN/'input_freeze.json').read_text())
meta=json.loads((FULL/'run_status.json').read_text(encoding='utf-8'))
cfg=json.loads((RUN/'local_config.json').read_text(encoding='utf-8'))
assert meta['status']=='complete' and len(meta['completed_models'])==4
assert meta['frozen_thresholds_sha256']==FROZEN_SHA
assert meta['thresholds']==thresholds
assert sha(RUN/'full_inputs.csv')==freeze['full_manifest_sha256']
assert sha(RUN/'android_inputs.csv')==freeze['android_manifest_sha256']
manifests={}
for scope,name in [('full','full_inputs.csv'),('selected','android_inputs.csv')]:
    rows=readcsv(RUN/name)
    manifests[scope]={(r['dataset'],r['image_id']):r for r in rows}
    assert len(rows)==len(manifests[scope])==(2477 if scope=='full' else 218)
assert set(manifests['selected']).issubset(manifests['full'])

# Re-decode each retained input once and independently verify its tensor identity.
tensor_hashes={};filechecks=[]
for identity,item in manifests['full'].items():
    p=Path(item['processed_path']);assert sha(p)==item['processed_sha256']
    with Image.open(p) as im:rgb=np.asarray(im.convert('RGB'),dtype=np.uint8)
    assert rgb.shape==(224,224,3)
    x=np.ascontiguousarray(rgb.astype(np.float32)[None]/np.float32(255))
    tensor_hashes[identity]=hashlib.sha256(x.tobytes()).hexdigest()
for key,spec in cfg['models'].items():
    record=meta['completed_models'][key]
    assert record['spec']['checkpoint_sha256']==spec['checkpoint_sha256']
    assert record['spec']['tflite_sha256']==spec['tflite_sha256']
    for artifact in ['checkpoint','tflite']:
        assert sha(spec[artifact])==spec[artifact+'_sha256']
        filechecks.append({'model':key,'artifact':artifact,'filename':Path(spec[artifact]).name,'sha256':spec[artifact+'_sha256']})

summary=[];decisions=[];fliprows=[];historyflips=[];coverage=[];sourcehash=[]
model_rows={}
for key,record in meta['completed_models'].items():
    path=FULL/f'{key}_paired.csv';raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==record['csv_sha256']
    rows=readcsv(path);model_rows[key]=rows
    scope='full' if key in ['efficientnet_lite0','mobilenetv3_large'] else 'selected'
    expected=manifests[scope]
    assert len(rows)==len(expected) and {(r['dataset'],r['image_id']) for r in rows}==set(expected)
    sourcehash.append({'filename':path.name,'sha256':sha(path),'completed_scope':scope,'n':len(rows)})
    sums={ds:{'N':0,'top1_match':0,'retained_top1_match':0,'max_abs_logit_diff':0.,'sum_abs_logit_diff':0.,'max_abs_probability_diff':0.,'sum_abs_probability_diff':0.,'max_abs_probability_diff_vs_retained':0.,'conversion_decision_agreement':Counter(),'rerun_retained_decision_agreement':Counter()} for ds in ['APTOS','MESSIDOR2']}
    cov=Counter()
    for row in rows:
        identity=row['dataset'],row['image_id'];ds=identity[0]
        assert row['processed_sha256']==expected[identity]['processed_sha256']
        assert row['input_tensor_sha256']==tensor_hashes[identity]
        assert int(row['label'])==int(expected[identity]['label'])
        cov[ds,int(row['label'])]+=1
        pl,pp=stable_softmax(row,'pytorch');tl,tp=stable_softmax(row,'tflite')
        for prefix,probs in [('pytorch',pp),('tflite',tp)]:
            assert max(abs(probs[k]-float(row[f'{prefix}_prob_{k}'])) for k in range(5))<2e-14
        pi=max(range(5),key=lambda i:pl[i]);ti=max(range(5),key=lambda i:tl[i])
        assert pi==int(row['pytorch_top1']) and ti==int(row['tflite_top1'])
        pr=math.fsum(pp[2:]);tr=math.fsum(tp[2:]);oldrisk=canonical[ds,key,row['image_id']]
        assert abs(pr-float(row['pytorch_risk']))<2e-14
        assert abs(tr-float(row['tflite_risk']))<2e-14
        assert oldrisk==float(row['retained_risk_sum_float64'])
        ld=[abs(a-b) for a,b in zip(pl,tl)];pdiff=[abs(a-b) for a,b in zip(pp,tp)]
        retained=[float(row[f'retained_prob_{k}']) for k in range(5)]
        s=sums[ds];s['N']+=1;s['top1_match']+=pi==ti;s['retained_top1_match']+=pi==int(row['retained_top1'])
        s['max_abs_logit_diff']=max(s['max_abs_logit_diff'],max(ld));s['sum_abs_logit_diff']+=math.fsum(ld)
        s['max_abs_probability_diff']=max(s['max_abs_probability_diff'],max(pdiff));s['sum_abs_probability_diff']+=math.fsum(pdiff)
        s['max_abs_probability_diff_vs_retained']=max(s['max_abs_probability_diff_vs_retained'],max(abs(a-b) for a,b in zip(pp,retained)))
        for rule,threshold in thresholds[key].items():
            assert float(row[rule+'_threshold'])==threshold
            ptdec=pr>=threshold;ltdec=tr>=threshold;olddec=oldrisk>=threshold
            assert int(row[rule+'_pytorch_decision'])==ptdec
            assert int(row[rule+'_tflite_decision'])==ltdec
            assert int(row[rule+'_retained_decision'])==olddec
            agree=ptdec==ltdec;hist=ptdec==olddec
            assert int(row[rule+'_agree'])==agree
            assert int(row[rule+'_pytorch_retained_agree'])==hist
            s['conversion_decision_agreement'][rule]+=agree;s['rerun_retained_decision_agreement'][rule]+=hist
            detail={'model':key,'dataset':ds,'image_id':row['image_id'],'grade':int(row['label']),'rule':rule,'threshold':threshold,'pytorch_risk_recomputed':pr,'tflite_risk_recomputed':tr,'retained_risk_canonical':oldrisk,'pytorch_decision':int(ptdec),'tflite_decision':int(ltdec),'retained_decision':int(olddec),'pytorch_margin':pr-threshold,'tflite_margin':tr-threshold}
            if not agree:fliprows.append(detail)
            if not hist:historyflips.append(detail)
    assert sha(path)==record['csv_sha256'],'Input CSV changed during independent verification'
    for (ds,grade),n in sorted(cov.items()):coverage.append({'model':key,'scope':scope,'dataset':ds,'grade':grade,'N':n})
    for ds,s in sums.items():
        s['model']=key;s['dataset']=ds;s['scope']=scope
        s['mean_abs_logit_diff']=s.pop('sum_abs_logit_diff')/(5*s['N']);s['mean_abs_probability_diff']=s.pop('sum_abs_probability_diff')/(5*s['N'])
        producer=record['by_dataset'][ds]
        assert s['N']==producer['N'] and s['top1_match']==producer['top1_matches']
        assert s['retained_top1_match']==producer['pytorch_retained_top1_matches']
        assert abs(s['max_abs_logit_diff']-producer['max_abs_logit_diff'])<1e-14
        assert abs(s['max_abs_probability_diff']-producer['max_abs_prob_diff'])<2e-14
        assert dict(s['conversion_decision_agreement'])==producer['decision_matches']
        assert dict(s['rerun_retained_decision_agreement'])==producer['pytorch_retained_decision_matches']
        summary.append(s)
        for rule,t in thresholds[key].items():decisions.append({'model':key,'dataset':ds,'scope':scope,'N':s['N'],'rule':rule,'threshold':t,'pytorch_tflite_agree_N':s['conversion_decision_agreement'][rule],'new_pytorch_retained_agree_N':s['rerun_retained_decision_agreement'][rule]})
assert sum(r['N'] for r in summary)==5390
assert sum(r['top1_match'] for r in summary)==5390
assert len(fliprows)==meta['conversion_decision_flip_records']==2
assert len(historyflips)==meta['rerun_vs_retained_decision_flip_records']==8
writecsv(B/'independent_coverage.csv',coverage)
writecsv(B/'independent_threshold_decisions.csv',decisions)
writecsv(B/'independent_conversion_boundary_records.csv',fliprows)
writecsv(B/'independent_rerun_retained_boundary_records.csv',historyflips)

# Independent arithmetic and image-identity checks of the separate raw-input arm.
rawmeta=json.loads((RUN/'raw_results/run_status.json').read_text(encoding='utf-8'))
assert rawmeta['status']=='complete'
rawrows={key:{(r['dataset'],r['image_id']):r for r in readcsv(RUN/f'raw_results/{key}_paired.csv')} for key in ['efficientnet_lite0','mobilenetv3_large']}
pixels=readcsv(RUN/'raw_results/raw_vs_retained_preprocessing.csv')
assert len(pixels)==218 and {(r['dataset'],r['image_id']) for r in pixels}==set(manifests['selected'])
rawgroups=defaultdict(lambda:{'N':0,'identical_direct_pixels_N':0,'identical_after_format_roundtrip_N':0,'Lite0_top1_changes':0,'MobileNet_top1_changes':0,'max_RGB_diff':0})
raw_conversion_top1=0;raw_n=0
for key,rows in rawrows.items():
    assert len(rows)==218
    assert sha(RUN/f'raw_results/{key}_paired.csv')==rawmeta['completed_models'][key]['csv_sha256']
    for identity,row in rows.items():
        pl,pp=stable_softmax(row,'pytorch');tl,tp=stable_softmax(row,'tflite')
        assert max(abs(pp[k]-float(row[f'pytorch_prob_{k}'])) for k in range(5))<2e-14
        assert max(abs(tp[k]-float(row[f'tflite_prob_{k}'])) for k in range(5))<2e-14
        raw_conversion_top1+=int(np.argmax(pl)==np.argmax(tl));raw_n+=1
        for rule,t in thresholds[key].items():
            assert int(row[rule+'_agree'])==((math.fsum(pp[2:])>=t)==(math.fsum(tp[2:])>=t))
for row in pixels:
    ds=row['dataset'];identity=ds,row['image_id'];ext=Path(row['image_id']).suffix.lower()
    g=rawgroups[ds,ext];g['N']+=1
    diff=int(row['online_vs_retained_max_RGB_diff']);g['max_RGB_diff']=max(g['max_RGB_diff'],diff)
    g['identical_direct_pixels_N']+=diff==0
    g['identical_after_format_roundtrip_N']+=int(row['after_same_extension_default_encode_decode_identical'])
    for key,name in [('efficientnet_lite0','Lite0_top1_changes'),('mobilenetv3_large','MobileNet_top1_changes')]:
        raw=rawrows[key][identity];changed=int(raw['pytorch_top1'])!=int(raw['retained_top1'])
        assert int(row[key+'_raw_vs_retained_top1_changed'])==changed
        g[name]+=changed
encoding_summary=json.loads((RUN/'raw_results/raw_encoding_summary.json').read_text())
for (ds,ext),g in rawgroups.items():
    other=encoding_summary['groups'][ds+'_'+ext]
    assert g['N']==other['N']
    assert g['identical_direct_pixels_N']==other['identical_online_to_retained_N']
    assert g['identical_after_format_roundtrip_N']==other['identical_after_original_extension_encode_decode_N']
    assert g['Lite0_top1_changes']==other['raw_vs_retained_top1_changes']['efficientnet_lite0']
    assert g['MobileNet_top1_changes']==other['raw_vs_retained_top1_changes']['mobilenetv3_large']
writecsv(B/'independent_raw_encoding_summary.csv',[{'dataset':ds,'extension':ext,**g} for (ds,ext),g in sorted(rawgroups.items())])

audit={'status':'PASS','date':'2026-10-16','scope':'Completed records only. Independent re-decoding/tensor hashing; scalar softmax, argmax, probability/risk and canonical-threshold arithmetic. No model training, inference or device execution by this script.','input_manifest_sha256':{'full':sha(RUN/'full_inputs.csv'),'selected':sha(RUN/'android_inputs.csv')},'frozen_thresholds_sha256':FROZEN_SHA,'canonical_risks_sha256':sha(STAT/'canonical_retained_risks_20261016.csv'),'completed_input_csvs':sourcehash,'artifact_identities_verified_against_actual_bytes':filechecks,'checks':{'retained_preprocessed_files_hash_checked':2477,'decoded_float32_input_tensors_hash_checked':2477,'model_image_pairs':5390,'top1_matches':5390,'conversion_threshold_boundary_records':len(fliprows),'rerun_vs_retained_boundary_records':len(historyflips),'raw_arm_model_image_pairs':raw_n,'raw_arm_pytorch_tflite_top1_matches':raw_conversion_top1,'no_partial_full_runs_used':True,'no_new_android_results_asserted':True},'summary':summary,'grade_domain_coverage':coverage,'conversion_boundary_records':fliprows,'rerun_retained_boundary_records':historyflips,'raw_encoding_groups':[{'dataset':ds,'extension':ext,**g} for (ds,ext),g in sorted(rawgroups.items())],'producer_run_status_sha256':sha(FULL/'run_status.json'),'verification_script_sha256':sha(__file__)}
(B/'independent_runtime_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit['checks'],indent=2));print('Boundary records:',json.dumps(fliprows,indent=2))
