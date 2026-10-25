"""Independent scalar recount and weighted-cluster validation; export text/tables."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib,json,math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'07_statistics/recomputed'
A=json.loads((B/'statistics_audit.json').read_text(encoding='utf-8'))
SOURCE=ROOT/'05_verification/inputs'
models=['EfficientNet-Lite0','MobileNetV3-Large']
targets=[.90,.95,.98,.99]
df={d:pd.read_csv(SOURCE/f) for d,f in [('APTOS','aptos_val_all_predictions_combined.csv'),('MESSIDOR2','messidor2_all_predictions_combined.csv')]}
allchecks=[]
for r in A['results']['operating_points']:
    rows=df[r['dataset']].loc[lambda d:d.model.eq(r['model'])]
    counts=Counter()
    scores=[]
    for item in rows.to_dict('records'):
        p=(item['prob_2']+item['prob_3'])+item['prob_4']
        positive=item['y_true']>=2
        pred=p>=r['full_precision_threshold']
        counts['TP' if pred and positive else 'FP' if pred else 'FN' if positive else 'TN']+=1
        if positive:scores.append(p)
    assert all(counts[k]==r[k] for k in ['TP','FP','FN','TN'])
    assert sum(counts.values())==r['n']
    assert counts['TP']+counts['FN']==(298 if r['dataset']=='APTOS' else 457)
    if r['dataset']=='APTOS':
        assert counts['TP']/len(scores)>=r['target_APTOS_sensitivity']
        # Removing every score tied at the selected boundary misses the target.
        just_above=np.nextafter(r['full_precision_threshold'],np.inf)
        assert sum(s>=just_above for s in scores)/len(scores)<r['target_APTOS_sensitivity']
    allchecks.append({'dataset':r['dataset'],'model':r['model'],'target':r['target_APTOS_sensitivity'],'scalar_counts_match':True})

# Independently validate all threshold bootstrap draws by multiplicity-weighted
# sufficient counts, instead of expanding duplicated examination/image rows.
pairpath=SOURCE/'messidor2_pairs.csv'
pair=pd.read_csv(pairpath,sep=';');pair.columns=pair.columns.str.strip()
lookup={Path(str(row[c]).strip()).stem:i for i,row in pair.iterrows() for c in ['left','right']}
for dataset in ['APTOS','MESSIDOR2']:
    for model in models:
        d=df[dataset].loc[lambda d:d.model.eq(model)].reset_index(drop=True)
        y=d.y_true.to_numpy()>=2;p=d[['prob_2','prob_3','prob_4']].sum(axis=1).to_numpy()
        thresholds=[r['full_precision_threshold'] for r in A['results']['operating_points'] if r['dataset']==dataset and r['model']==model]
        cid=np.arange(len(d)) if dataset=='APTOS' else d.image_id.map(lambda z:lookup[Path(str(z)).stem]).to_numpy()
        k=len(d) if dataset=='APTOS' else 874
        pos=np.bincount(cid,weights=y,minlength=k);neg=np.bincount(cid,weights=~y,minlength=k)
        tp=np.column_stack([np.bincount(cid,weights=y&(p>=t),minlength=k) for t in thresholds])
        tn=np.column_stack([np.bincount(cid,weights=(~y)&(p<t),minlength=k) for t in thresholds])
        rng=np.random.default_rng(42 if dataset=='APTOS' else 43)
        vals=[]
        for _ in range(2000):
            w=np.bincount(rng.integers(0,k,k),minlength=k)
            vals.append(np.r_[(w@tp)/(w@pos),(w@tn)/(w@neg)])
        expected=pd.read_csv(B/f'{dataset}_{model}_threshold_bootstrap_draws.csv').to_numpy()
        assert np.max(abs(np.array(vals)-expected))<1e-14

# Independent reliability validation from per-bin per-examination membership.
members=pd.read_csv(B/'Figure2_fixed_bin_membership.csv')
m=members.loc[members.dataset.eq('MESSIDOR2')]
assert len(m)==1744 and m.image_id.nunique()==1744 and m.pair_row_index.nunique()==874
assert dict(Counter(m.groupby('pair_row_index').size()))=={2:870,1:4}
numer=np.zeros((874,10));denom=np.zeros((874,10))
for item in m.to_dict('records'):
    c=int(item['pair_row_index']);b=int(item['bin'])-1
    numer[c,b]+=item['referable_true'];denom[c,b]+=1
rng=np.random.default_rng(45);vals=[]
for _ in range(2000):
    w=np.bincount(rng.integers(0,874,874),minlength=874)
    vals.append((w@numer)/(w@denom))
expected=pd.read_csv(B/'Figure2_MESSIDOR2_fixed_bin_cluster_bootstrap_draws.csv').to_numpy()
assert np.max(abs(np.array(vals)-expected))<1e-14

# Cross-check diagnostic points by bisection and full-design-matrix Newton fit.
def independent_calibration(d):
    y=(d.y_true.to_numpy()>=2).astype(float)
    p=d[['prob_2','prob_3','prob_4']].sum(axis=1).to_numpy()
    pc=np.clip(p,1e-6,1-1e-6);x=np.log(pc/(1-pc))
    sig=lambda v:1/(1+np.exp(-np.clip(v,-700,700)))
    lo,hi=-50.,50.
    for _ in range(120):
        mid=(lo+hi)/2
        if (sig(x+mid)-y).sum()>0:hi=mid
        else:lo=mid
    X=np.column_stack([np.ones(len(x)),x]);beta=np.array([0.,1.])
    nll=lambda b:np.sum(np.logaddexp(0,X@b)-y*(X@b))
    for _ in range(100):
        q=sig(X@beta);g=X.T@(q-y);H=X.T@((q*(1-q))[:,None]*X)
        if np.max(abs(g))<1e-8:break
        step=np.linalg.solve(H,g);rate=1.
        while nll(beta-rate*step)>nll(beta)+1e-12:
            rate/=2
            assert rate>1e-12
        beta-=rate*step
    assert np.max(abs(X.T@(sig(X@beta)-y)))<1e-5
    return (lo+hi)/2,beta[1],math.fsum((float(v)-int(t))**2 for v,t in zip(p,y))/len(y)
for row in A['results']['all_models_point_estimates']:
    d=df[row['dataset']].loc[lambda d:d.model.eq(row['model'])]
    a,s,brier=independent_calibration(d)
    assert abs(a-row['intercept_slope_fixed_1'])<1e-8
    assert abs(s-row['slope_intercept_free'])<1e-8
    assert abs(brier-row['brier'])<1e-14

verification={'status':'PASS','checks':{'scalar_recount_all_operating_points':len(allchecks),'source_threshold_largest_attaining_target_with_ties':True,'all_threshold_bootstrap_draws_match_independent_multiplicity_counts':True,'Figure2_all_external_cluster_draws_match_independent_multiplicity_counts':True,'external_images':1744,'external_examinations':874,'two_image_examinations':870,'one_image_examinations':4,'independent_bisection_intercept_design_matrix_slope_and_scalar_brier_points':8},'scripts_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).with_name('extend_statistics.py'),Path(__file__)]}}
(B/'independent_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')

op=A['results']['operating_points'];cal=A['results']['calibration'];brief=A['results']['all_models_point_estimates']
fmt=lambda v,d=2:f'{100*v:.{d}f}'
def getop(model,dataset,target):return next(r for r in op if r['model']==model and r['dataset']==dataset and r['target_APTOS_sensitivity']==target)
def getpoint(model,dataset):return next(r for r in brief if r['model']==model and r['dataset']==dataset)
def cint(r,key):return f"{fmt(r[key])} ({fmt(r[key+'_ci_low'])}–{fmt(r[key+'_ci_high'])})"

threshold_header=['Model','APTOS sensitivity target (%)','Threshold','APTOS sensitivity (%)','APTOS specificity (%)','M2 TP/FP/FN/TN','M2 sensitivity (%)','M2 specificity (%)']
threshold_rows=[]
for model in models:
    for target in targets:
        s=getop(model,'APTOS',target);e=getop(model,'MESSIDOR2',target)
        threshold_rows.append([model,str(int(target*100)),f"{s['full_precision_threshold']:.6f}",cint(s,'sensitivity'),cint(s,'specificity'),f"{e['TP']}/{e['FP']}/{e['FN']}/{e['TN']}",cint(e,'sensitivity'),cint(e,'specificity')])
cal_header=['Model','Dataset','Mean risk (%)','Brier score','ECE','Calibration intercept','Calibration slope','Binary log loss']
cal_rows=[]
for model in models:
    for dataset in ['APTOS','MESSIDOR2']:
        point=getpoint(model,dataset)
        values=[]
        for metric in ['brier','ece_equal_frequency','intercept_slope_fixed_1','slope_intercept_free','binary_log_loss']:
            r=next(r for r in cal if r['model']==model and r['dataset']==dataset and r['metric']==metric)
            digits=2 if metric in ['intercept_slope_fixed_1','slope_intercept_free'] else 4
            values.append(f"{r['point_estimate']:.{digits}f} ({r['ci_low']:.{digits}f}–{r['ci_high']:.{digits}f})")
        cal_rows.append([model,'APTOS' if dataset=='APTOS' else 'MESSIDOR-2',fmt(point['mean_predicted_risk']),*values])

methods="To connect operating-point analysis with the Android evaluation, the same APTOS-derived referable-DR threshold rule and calibration diagnostics were applied to MobileNetV3-Large and EfficientNet-Lite0. For each target sensitivity, the largest APTOS threshold attaining the target was selected, with scores equal to the threshold classified as positive; tied scores were included together. Thresholds were then applied unchanged to MESSIDOR-2. Confidence intervals used 2,000 image-level APTOS resamples (thresholds: seed 42; calibration: seed 44) and 2,000 paired-eye examination-cluster MESSIDOR-2 resamples (thresholds: seed 43; calibration: seed 45). Operating-point intervals are conditional on the retained checkpoint and selected threshold. Calibration fits were diagnostic and were not applied to modify predictions. For the MESSIDOR-2 reliability plot, the original equal-count bin membership was fixed and complete examinations were resampled to obtain 95% percentile intervals for the observed fraction within each bin."
results="Extending the audit to the two Android models revealed distinct operating-point tradeoffs. Increasing the APTOS sensitivity target from 90% to 99% changed MESSIDOR-2 sensitivity from 34.57% to 55.36% for MobileNetV3-Large and from 35.89% to 76.59% for EfficientNet-Lite0. The corresponding external true-positive/false-positive counts changed from 158/20 to 253/106 and from 164/49 to 350/441, respectively. At the 99% source target, MobileNetV3-Large therefore retained higher external specificity (91.76% versus 65.73%), whereas EfficientNet-Lite0 detected more referable images. External Brier scores were 0.1582 and 0.1590 and ECE values were 0.1470 and 0.1180, respectively. These results connect the application benchmarks to the transfer of the actual Android candidates' screening decisions."
discussion="The audit supports model and operating-point choices using both decision counts and execution costs. In these retained predictions, the 99%-source-target EfficientNet-Lite0 operating point detected 97 more referable MESSIDOR-2 images than MobileNetV3-Large while producing 335 more false positives. MobileNetV3-Large combined higher specificity at that operating point with lower measured end-to-end latency on the evaluated device. This is a concrete example of why the fastest model, the highest-sensitivity operating point, and the most accurate probability estimates need to be assessed as separate design choices."
caption="Figure 2. Five-class error and referable-DR calibration analysis for MobileViT-S. Panels A and B show confusion-matrix counts and percentages within each reference-standard row. Panels C and D use 10 equal-count bins of referable risk (sum of probabilities for grades 2–4), plotted against the observed referable fraction. Vertical bars are Wilson 95% intervals for APTOS and paired-eye examination-cluster bootstrap 95% percentile intervals for MESSIDOR-2 (2,000 resamples; seed 45), with original bin membership fixed. The diagonal denotes perfect calibration. APTOS included 733 images; MESSIDOR-2 included 1,744 gradable images from 874 examinations."
threshold_note="Sensitivity and specificity are point estimates with 95% percentile bootstrap intervals in parentheses. Thresholds were selected only on APTOS and held fixed during external evaluation and bootstrap. APTOS: 733 images, 298 referable and 435 nonreferable; MESSIDOR-2: 1,744 images, 457 referable and 1,287 nonreferable, grouped into 874 examinations. APTOS intervals use image-level resampling (2,000 draws; seed 42); MESSIDOR-2 intervals resample examinations with all retained eyes (seed 43). Thresholds are shown to six decimals for readability; all decisions used full precision supplied in the companion CSV. TP/FP/FN/TN denote true positives/false positives/false negatives/true negatives. The 99% source target for MobileNetV3-Large attains 297/298 (99.66%) because two positive inputs share the boundary score; no tie was broken using external labels."
cal_note="Values in parentheses are 95% percentile bootstrap intervals (2,000 draws), with images sampled for APTOS (seed 44) and paired-eye examinations for MESSIDOR-2 (seed 45). Observed referable prevalence is 40.65% for APTOS and 26.20% for MESSIDOR-2. ECE uses 10 equal-count bins; intercept fixes logistic slope at one, whereas slope is estimated with a free intercept. Probabilities are clipped to [1e-6, 1−1e-6] before logit transformation for diagnostic fitting, and binary log loss uses natural logarithms with probability clipping at 1e-15. These fits are descriptive diagnostics and were not applied as probability recalibration."
payload={'methods_addition':methods,'results_addition':results,'discussion_addition':discussion,'Figure2_caption':caption,'threshold_table':{'suggested_location':'Supplement; assign final numbering during manuscript integration','title':'Source-selected referral thresholds for the two Android models','header':threshold_header,'rows':threshold_rows,'note':threshold_note},'calibration_table':{'suggested_location':'Supplement, adjacent to existing MobileViT-S Table S3; assign final numbering during manuscript integration','title':'Referable-risk calibration for the two Android models','header':cal_header,'rows':cal_rows,'note':cal_note},'chinese_rationale':['把实际运行手机的两个模型补到同一阈值与校准分析中，让跨域结果和移动实现直接对应。','新增真正能辅助选择的计数：更高检出率需要增加多少误报，不能只看AUC或速度。','Figure 2保留原点值和混淆矩阵，只把外部误差条换成与双眼成组数据相符的方法。','既有MobileViT-S S3区间没有重算或替换；没有训练、推理、重新校准，也没有利用外部标签选阈值。']}
(B/'manuscript_payload.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
md=['# 现有预测补充分析完成记录','', '本次完成的是现有概率输出的统计分析；未新增训练、模型推理或真实设备结果。','', '## 新完成的内容','',*['- '+s for s in payload['chinese_rationale']],'','## 可入稿英文段落','',methods,'',results,'',discussion,'','## Figure 2图注','',caption,'']
for key in ['threshold_table','calibration_table']:
    table=payload[key]
    md+=['## '+table['title'],'','| '+' | '.join(table['header'])+' |','| '+' | '.join(['---']*len(table['header']))+' |']
    md+=['| '+' | '.join(row)+' |' for row in table['rows']]
    md+=['',table['note'],'']
md+=['## 核查结果','','独立逐图计数、32个阈值结果、所有阈值bootstrap样本及外部图每次成组重采样均通过。校准点估计另用二分法和设计矩阵Newton法核对。详见 independent_verification.json。','', '原图上半部A/B混淆矩阵像素不变；下半部仍为原10箱原点值，外部误差条使用固定箱成员的874检查组bootstrap。SVG含原A/B栅格图和下部矢量图，不宣称整张图为纯矢量。','']
(B/'分析完成说明与可入稿内容.md').write_text('\n'.join(md),encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False,indent=2))
