"""Independent checks of saved probabilities, without training or inference.

This is a NEW, fully specified computation. It does not identify or recreate
the unknown historical random-number generator or bin tie-handling code.
"""
from pathlib import Path
import hashlib, json, time
import numpy as np
import pandas as pd

B = Path(__file__).resolve().parent
S = B.parent / 'inputs'
OUT = B.parent / 'results'
OUT.mkdir(exist_ok=True)

def expit(x):
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

def diagnostics(p, y):
    x = np.log(np.clip(p, 1e-6, 1-1e-6)) - np.log1p(-np.clip(p, 1e-6, 1-1e-6))
    a = 0.
    for _ in range(80):
        q = expit(x+a)
        g = (q-y).sum()
        if abs(g) < 1e-8:
            break
        step = g/(q*(1-q)).sum()
        a -= np.clip(step, -5, 5)
    assert abs((expit(x+a)-y).sum()) < 1e-5
    beta = np.array([0., 1.])
    def loss(b):
        z = b[0] + b[1]*x
        return np.sum(np.logaddexp(0, z)-y*z)
    for _ in range(80):
        q = expit(beta[0]+beta[1]*x)
        g = np.array([(q-y).sum(), ((q-y)*x).sum()])
        if np.max(abs(g)) < 1e-8:
            break
        w = q*(1-q)
        H = np.array([[w.sum(), (w*x).sum()], [(w*x).sum(), (w*x*x).sum()]])
        step = np.linalg.solve(H, g)
        rate = 1.
        while loss(beta-rate*step) > loss(beta)+1e-12:
            rate /= 2
            assert rate > 1e-12
        beta -= rate*step
    q = expit(beta[0]+beta[1]*x)
    assert max(abs((q-y).sum()), abs(((q-y)*x).sum())) < 1e-5
    order = np.argsort(p, kind='stable')
    ece = sum(len(k)/len(y)*abs(y[k].mean()-p[k].mean()) for k in np.array_split(order, 10))
    pc = np.clip(p, 1e-15, 1-1e-15)
    ll = -np.mean(y*np.log(pc)+(1-y)*np.log1p(-pc))
    return [np.mean((p-y)**2), ece, a, beta[1], ll]

metrics = ['brier', 'ece_equal_frequency', 'intercept_slope_fixed_1', 'slope_intercept_free', 'binary_log_loss']
old = {
    'APTOS': [(0.0395,0.0604),(0.0202,0.0488),(-0.00,0.59),(0.92,1.34),None],
    'MESSIDOR2': [(0.1204,0.1527),(0.0858,0.1265),(1.20,1.65),(0.62,0.79),None]
}
pair_path = S/'messidor2_pairs.csv'
pair = pd.read_csv(pair_path, sep=';')
pair.columns = pair.columns.str.strip()
lookup = {Path(str(row[c]).strip()).stem:i for i,row in pair.iterrows() for c in ['left','right']}
allrows = []
details = {}
hashes = {str(pair_path):hashlib.sha256(pair_path.read_bytes()).hexdigest()}
for dataset, file, seed in [('APTOS','aptos_val_all_predictions_combined.csv',44),('MESSIDOR2','messidor2_all_predictions_combined.csv',45)]:
    start = time.perf_counter()
    path = S/file
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    d = pd.read_csv(path)
    d = d.loc[d.model=='MobileViT-S'].reset_index(drop=True)
    y = (d.y_true.to_numpy()>=2).astype(float)
    p = d[['prob_2','prob_3','prob_4']].sum(axis=1).to_numpy()
    point = diagnostics(p,y)
    rng = np.random.default_rng(seed)
    if dataset=='MESSIDOR2':
        ids = d.image_id.map(lambda v:Path(str(v)).stem)
        clusters = ids.map(lookup)
        assert clusters.notna().all() and clusters.nunique()==874
        groups = [np.where(clusters.to_numpy()==k)[0] for k in sorted(clusters.unique())]
        assert sum(map(len,groups))==1744
    draws=[]
    for r in range(2000):
        if dataset=='APTOS':
            idx = rng.integers(0,len(y),size=len(y))
        else:
            sampled = rng.integers(0,len(groups),size=len(groups))
            idx = np.concatenate([groups[k] for k in sampled])
        draws.append(diagnostics(p[idx],y[idx]))
    a = np.asarray(draws)
    assert a.shape==(2000,5) and np.isfinite(a).all()
    ci = np.percentile(a,[2.5,97.5],axis=0,method='linear')
    pd.DataFrame(a,columns=metrics).to_csv(OUT/(dataset+'_bootstrap_draws.csv'),index=False)
    rows=[]
    for i,m in enumerate(metrics):
        legacy=old[dataset][i]
        digits=4 if i in [0,1,4] else 2
        match = None if legacy is None else [round(float(ci[0,i]),digits),round(float(ci[1,i]),digits)] == [round(legacy[0],digits),round(legacy[1],digits)]
        row={'dataset':dataset,'metric':m,'point_estimate':point[i],'new_ci_low':float(ci[0,i]),'new_ci_high':float(ci[1,i]),'legacy_ci_low':None if legacy is None else legacy[0],'legacy_ci_high':None if legacy is None else legacy[1],'rounded_ci_matches_legacy':match}
        rows.append(row);allrows.append(row)
    details[dataset]={'n':len(y),'seed':seed,'bootstrap_replicates':2000,'resampling_unit':'image' if dataset=='APTOS' else 'paired-eye examination, all retained images within each sampled examination','examinations':None if dataset=='APTOS' else 874,'seconds':time.perf_counter()-start,'results':rows}
    print(dataset, 'complete', round(details[dataset]['seconds'],2),'seconds',flush=True)
pd.DataFrame(allrows).to_csv(OUT/'independent_calibration_bootstrap_summary.csv',index=False)
result={'date':'2026-10-14','source_sha256':hashes,'definitions':{'rng':'NumPy default_rng PCG64','source_order':'saved CSV order, restricted to MobileViT-S','ece':'10 stable-sorted, equal-count array_split groups, rebuilt inside each replicate','logit_clip':1e-6,'intercept':'logistic intercept with slope fixed to one','slope':'logistic slope with an estimated intercept','percentile':'2.5 and 97.5 percentiles, linear interpolation','log_loss':'unweighted image mean, natural logs, probability clip [1e-15,1-1e-15]'},'scope':'NEW independent arithmetic bootstrap from retained predictions. Not training, inference, or recovery of historical source code. Existing manuscript intervals have not been replaced by this script.','results':details}
(OUT/'independent_bootstrap_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(pd.DataFrame(allrows).to_string(index=False))
