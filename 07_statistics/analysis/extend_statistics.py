"""New analysis of retained predictions; no training, inference or recalibration.

Preserves the verified 2026-10-14 probability, calibration and threshold definitions.
Source-selected thresholds are held fixed in every uncertainty calculation.
Figure 2 uses the original upper panels, original bin point estimates and a new
paired-examination interval for the external observed fraction in each fixed bin.
"""
from pathlib import Path
import base64, hashlib, io, json, math, platform, sys, time
from collections import Counter
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '07_statistics/recomputed'
OUT.mkdir(parents=True,exist_ok=True)
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
PAPER = ROOT
SOURCE = ROOT / '05_verification/inputs'
PAIR = SOURCE / 'messidor2_pairs.csv'
OLD = ROOT / '08_figures/Figure2_original_20261015.png'
OLD_BOOT = ROOT / '05_verification/results/independent_calibration_bootstrap_summary.csv'
MODELS = ['EfficientNet-Lite0','MobileNetV3-Large','MobileViT-S','ResNet-50']
ANDROID = MODELS[:2]
METRICS = ['brier','ece_equal_frequency','intercept_slope_fixed_1','slope_intercept_free','binary_log_loss']
TARGETS = [.90,.95,.98,.99]
NBOOT = 2000

def provenance(p):
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}

def expit(x):
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

def diagnostics(p, y):
    # Identical definitions to recompute_bootstrap.py, 2026-10-14.
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
    return np.array([np.mean((p-y)**2), ece, a, beta[1], ll])

def auc(y,p):
    ranks=pd.Series(p).rank(method='average').to_numpy()
    n=y.sum()
    return float((ranks[y].sum()-n*(n+1)/2)/(n*(len(y)-n)))

def index_draws(n, groups, seed):
    rng=np.random.default_rng(seed)
    for _ in range(NBOOT):
        if groups is None:
            yield rng.integers(0,n,size=n)
        else:
            yield np.concatenate([groups[k] for k in rng.integers(0,len(groups),size=len(groups))])

pair=pd.read_csv(PAIR,sep=';')
pair.columns=pair.columns.str.strip()
assert len(pair)==874
lookup={Path(str(row[c]).strip()).stem:i for i,row in pair.iterrows() for c in ['left','right']}
assert len(lookup)==1748
data={}; provenance_list=[provenance(PAIR),provenance(OLD),provenance(OLD_BOOT)]
for dataset,file in [('APTOS','aptos_val_all_predictions_combined.csv'),('MESSIDOR2','messidor2_all_predictions_combined.csv')]:
    path=SOURCE/file
    df=pd.read_csv(path)
    provenance_list.append(provenance(path))
    # Verify original predictions are byte-identical to the previously audited copy.
    # Packaged input hashes are checked by run_statistics.py before execution.
    for model in MODELS:
        d=df.loc[df.model.eq(model)].reset_index(drop=True)
        n=733 if dataset=='APTOS' else 1744
        assert len(d)==n and d.image_id.nunique()==n
        probs=d[[f'prob_{i}' for i in range(5)]].to_numpy(float)
        assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
        assert np.max(abs(probs.sum(axis=1)-1))<1e-5
        assert np.array_equal(probs.argmax(axis=1),d.y_pred.to_numpy())
        y=d.y_true.to_numpy(int)>=2
        p=probs[:,2:].sum(axis=1)
        assert y.sum()==(298 if dataset=='APTOS' else 457)
        assert np.max(abs(p-d.referable_score.to_numpy()))<1e-6
        groups=None;cluster=None
        if dataset=='MESSIDOR2':
            cluster=d.image_id.map(lambda v:lookup[Path(str(v)).stem]).to_numpy(int)
            groups=[np.where(cluster==k)[0] for k in sorted(set(cluster))]
            assert len(groups)==874 and sum(map(len,groups))==1744
            assert Counter(map(len,groups))=={2:870,1:4}
        data[dataset,model]={'d':d,'y':y,'p':p,'groups':groups,'cluster':cluster}

start=time.perf_counter()
operating=[];calib=[];brief=[]
for model in MODELS:
    src=data['APTOS',model]
    positives=np.sort(src['p'][src['y']])[::-1]
    thresholds=np.array([positives[int(np.ceil(t*len(positives)))-1] for t in TARGETS])
    for dataset in ['APTOS','MESSIDOR2']:
        item=data[dataset,model];y=item['y'];p=item['p']
        point=diagnostics(p,y)
        brief.append({'dataset':dataset,'model':model,'n':len(y),'positive_n':int(y.sum()),'negative_n':int((~y).sum()),'prevalence':float(y.mean()),'mean_predicted_risk':float(p.mean()),'referable_auc':auc(y,p),**dict(zip(METRICS,map(float,point)))})
        preds=p[:,None]>=thresholds[None,:]
        tp=np.sum(preds&y[:,None],axis=0);fp=np.sum(preds&~y[:,None],axis=0)
        fn=y.sum()-tp;tn=(~y).sum()-fp
        ci=None
        if model in ANDROID:
            draws=[]
            for idx in index_draws(len(y),item['groups'],42 if dataset=='APTOS' else 43):
                yy=y[idx];pp=preds[idx]
                draws.append(np.r_[pp[yy].mean(axis=0),(~pp[~yy]).mean(axis=0)])
            draws=np.array(draws)
            assert draws.shape==(NBOOT,8) and np.isfinite(draws).all()
            pd.DataFrame(draws,columns=[f'{m}_{int(t*100)}' for m in ['sensitivity','specificity'] for t in TARGETS]).to_csv(OUT/f'{dataset}_{model}_threshold_bootstrap_draws.csv',index=False)
            ci=np.percentile(draws,[2.5,97.5],axis=0,method='linear')
            cdraw=np.array([diagnostics(p[idx],y[idx]) for idx in index_draws(len(y),item['groups'],44 if dataset=='APTOS' else 45)])
            assert cdraw.shape==(NBOOT,5) and np.isfinite(cdraw).all()
            cci=np.percentile(cdraw,[2.5,97.5],axis=0,method='linear')
            pd.DataFrame(cdraw,columns=METRICS).to_csv(OUT/f'{dataset}_{model}_calibration_bootstrap_draws.csv',index=False)
            for j,metric in enumerate(METRICS):
                calib.append({'dataset':dataset,'model':model,'metric':metric,'point_estimate':float(point[j]),'ci_low':float(cci[0,j]),'ci_high':float(cci[1,j])})
        for j,target in enumerate(TARGETS):
            row={'model':model,'dataset':dataset,'target_APTOS_sensitivity':target,'full_precision_threshold':float(thresholds[j]),'TP':int(tp[j]),'FP':int(fp[j]),'FN':int(fn[j]),'TN':int(tn[j]),'sensitivity':float(tp[j]/y.sum()),'specificity':float(tn[j]/(~y).sum()),'n':len(y)}
            if ci is not None:
                row.update({'sensitivity_ci_low':float(ci[0,j]),'sensitivity_ci_high':float(ci[1,j]),'specificity_ci_low':float(ci[0,j+4]),'specificity_ci_high':float(ci[1,j+4])})
            operating.append(row)
        print(dataset,model,'complete',round(time.perf_counter()-start,1),'seconds',flush=True)

pd.DataFrame(operating).to_csv(OUT/'source_selected_thresholds_all_models.csv',index=False)
pd.DataFrame(calib).to_csv(OUT/'android_models_calibration_with_intervals.csv',index=False)
pd.DataFrame(brief).to_csv(OUT/'all_models_point_estimates.csv',index=False)

# Reliability intervals: keep original fixed-bin members, resample whole exams.
reliability=[];members=[]
for dataset in ['APTOS','MESSIDOR2']:
    item=data[dataset,'MobileViT-S'];p=item['p'];y=item['y']
    bins=np.array_split(np.argsort(p,kind='stable'),10)
    previous=pd.read_csv(SOURCE/f'{"APTOS_val" if dataset=="APTOS" else "MESSIDOR-2"}_mobilevit_bins.csv')
    for b,idx in enumerate(bins):
        assert len(idx)==int(previous.iloc[b]['n'])
        assert abs(p[idx].mean()-previous.iloc[b]['mean_predicted'])<1e-14
        assert abs(y[idx].mean()-previous.iloc[b]['observed_fraction'])<1e-14
        for i in idx:
            members.append({'dataset':dataset,'image_id':item['d'].iloc[i].image_id,'bin':b+1,'referable_risk':float(p[i]),'referable_true':int(y[i]),'pair_row_index':None if dataset=='APTOS' else int(item['cluster'][i])})
    if dataset=='MESSIDOR2':
        binid=np.empty(len(y),int)
        for b,idx in enumerate(bins):binid[idx]=b
        numer=np.zeros((874,10));denom=np.zeros((874,10))
        for i in range(len(y)):
            numer[item['cluster'][i],binid[i]]+=int(y[i])
            denom[item['cluster'][i],binid[i]]+=1
        rng=np.random.default_rng(45);rvals=[];rsizes=[]
        for _ in range(NBOOT):
            ids=rng.integers(0,874,size=874)
            dd=denom[ids].sum(axis=0)
            assert (dd>0).all()
            rvals.append(numer[ids].sum(axis=0)/dd)
            rsizes.append(int(dd.sum()))
        ci=np.percentile(np.array(rvals),[2.5,97.5],axis=0,method='linear')
        pd.DataFrame(rvals,columns=[f'bin_{i}' for i in range(1,11)]).to_csv(OUT/'Figure2_MESSIDOR2_fixed_bin_cluster_bootstrap_draws.csv',index=False)
    for b,idx in enumerate(bins):
        rate=float(y[idx].mean());n=len(idx)
        if dataset=='APTOS':
            z=1.959963984540054
            mid=(rate+z*z/(2*n))/(1+z*z/n)
            half=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
            lo,hi=max(0,mid-half),min(1,mid+half)
        else:lo,hi=map(float,ci[:,b])
        reliability.append({'dataset':dataset,'bin':b+1,'n':n,'events':int(y[idx].sum()),'mean_predicted':float(p[idx].mean()),'observed_fraction':rate,'ci_low':lo,'ci_high':hi,'interval':'Wilson 95%' if dataset=='APTOS' else 'paired-examination percentile bootstrap 95% (fixed bin membership)'})
pd.DataFrame(reliability).to_csv(OUT/'Figure2_reliability_bins.csv',index=False)
pd.DataFrame(members).to_csv(OUT/'Figure2_fixed_bin_membership.csv',index=False)

# Replace the calibration portion only; original A/B pixels are retained.
original=Image.open(OLD).convert('RGB')
assert original.size==(2400,2520)
im=original.copy();draw=ImageDraw.Draw(im)
CUT=1275
draw.rectangle((0,CUT,2399,2519),fill='white')
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="2520" viewBox="0 0 2400 2520">','<rect width="2400" height="2520" fill="white"/>']
top=original.crop((0,0,2400,CUT));bio=io.BytesIO();top.save(bio,format='PNG')
svg.append(f'<image x="0" y="0" width="2400" height="{CUT}" href="data:image/png;base64,{base64.b64encode(bio.getvalue()).decode()}"/>')
fonts={}
def font(size,bold=False):
    key=size,bold
    if key not in fonts:
        candidates=[r'C:\Windows\Fonts\timesbd.ttf' if bold else r'C:\Windows\Fonts\times.ttf','DejaVuSerif-Bold.ttf' if bold else 'DejaVuSerif.ttf']
        for candidate in candidates:
            try:fonts[key]=ImageFont.truetype(candidate,size);break
            except OSError:pass
        else:fonts[key]=ImageFont.load_default(size=size)
    return fonts[key]
def line(x1,y1,x2,y2,fill='#000000',width=2):
    draw.line((round(x1),round(y1),round(x2),round(y2)),fill=fill,width=width)
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{fill}" stroke-width="{width}"/>')
def text(x,y,s,size=26,bold=False,anchor='mm',color='#000000',rotation=0):
    f=font(size,bold)
    if rotation:
        box=f.getbbox(s);layer=Image.new('RGBA',(box[2]-box[0]+20,size*2),(255,255,255,0));ld=ImageDraw.Draw(layer)
        ld.text((10,size),s,font=f,fill=color,anchor='lm');layer=layer.rotate(rotation,expand=True)
        im.paste(layer,(round(x-layer.width/2),round(y-layer.height/2)),layer)
    else:draw.text((round(x),round(y)),s,font=f,fill=color,anchor=anchor)
    ta='middle' if anchor.startswith('m') else ('start' if anchor.startswith('l') else 'end')
    transform=f' transform="rotate({-rotation} {x} {y})"' if rotation else ''
    svg.append(f'<text x="{x}" y="{y}" font-family="Times New Roman,serif" font-size="{size}" font-weight="{"bold" if bold else "normal"}" text-anchor="{ta}" dominant-baseline="middle" fill="{color}"{transform}>{escape(s)}</text>')
def circle(x,y,color):
    draw.ellipse((round(x-7),round(y-7),round(x+7),round(y+7)),fill=color,outline='white',width=2)
    svg.append(f'<circle cx="{x}" cy="{y}" r="7" fill="{color}" stroke="white" stroke-width="2"/>')
for dataset,x0,x1,panel,color,title in [('APTOS',205,1128,'C','#0066a7','APTOS referable-DR calibration'),('MESSIDOR2',1365,2288,'D','#e66b00','MESSIDOR-2 referable-DR calibration')]:
    y0,y1=1407,2311
    text(x0-125,1320,panel,44,True)
    text((x0+x1)/2,1336,title,36,True)
    px=lambda v:x0+v*(x1-x0)
    py=lambda v:y1-v*(y1-y0)
    for tick in np.linspace(0,1,6):
        line(px(tick),y0,px(tick),y1,'#dddddd',2)
        line(x0,py(tick),x1,py(tick),'#dddddd',2)
        text(px(tick),y1+32,f'{tick:.1f}',24)
        text(x0-34,py(tick),f'{tick:.1f}',24,anchor='rm')
    line(x0,y1,x1,y1,width=3);line(x0,y0,x0,y1,width=3)
    line(x0,y1,x1,y0,'#777777',3)
    rows=[r for r in reliability if r['dataset']==dataset]
    last=None
    for r in rows:
        x,y=px(r['mean_predicted']),py(r['observed_fraction'])
        if last is not None:line(*last,x,y,color,4)
        last=x,y
        lo,hi=py(r['ci_low']),py(r['ci_high'])
        line(x,lo,x,hi,color,3);line(x-7,lo,x+7,lo,color,3);line(x-7,hi,x+7,hi,color,3)
        circle(x,y,color)
    text((x0+x1)/2,2380,'Mean predicted referable-DR probability',29,True)
    text(x0-100,(y0+y1)/2,'Observed referable-DR fraction',28,True,rotation=90)
    metrics=[r for r in brief if r['dataset']==dataset and r['model']=='MobileViT-S'][0]
    text((x0+x1)/2,2425,f"Brier {metrics['brier']:.3f}  |  ECE {metrics['ece_equal_frequency']:.3f}",24)
    text((x0+x1)/2,2453,f"Intercept {metrics['intercept_slope_fixed_1']:.2f}  |  Slope {metrics['slope_intercept_free']:.2f}",24)
    text((x0+x1)/2,2485,'Wilson 95% intervals' if dataset=='APTOS' else 'Paired-examination bootstrap 95% intervals',22)
svg.append('</svg>')
im.save(FIG/'Figure2_errors_and_calibration_20261016.png',dpi=(300,300))
(FIG/'Figure2_errors_and_calibration_20261016.svg').write_text('\n'.join(svg),encoding='utf-8')
assert Image.open(FIG/'Figure2_errors_and_calibration_20261016.png').crop((0,0,2400,CUT)).tobytes()==original.crop((0,0,2400,CUT)).tobytes()

audit={'date':'2026-10-16','runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform()},'source_inputs':provenance_list,'scope':'New calculations from original retained probabilities. No training, inference, recalibration or changes to original inputs. MobileViT-S S3 intervals are not recomputed or replaced.','methods':{'referable_target':'grade >= 2','referable_risk':'sum(prob_2,prob_3,prob_4)','threshold_rule':'Largest source positive score whose p>=threshold rule attains empirical sensitivity >=target. Four targets 0.90/0.95/0.98/0.99. MESSIDOR2 labels never used to choose thresholds.','threshold_ci':'Fixed thresholds in 2000 resamples; source images seed42, external paired-eye exams seed43. Conditional on retained model and source-selected threshold.','calibration_ci':'Android models only,2000 resamples; source images seed44, external paired-eye exams seed45. Same definitions as independent 2026-10-14 S3 computation.','calibration':'Logistic intercept with slope fixed at1; slope with estimated intercept; diagnostic probabilities clipped before logit transformation [1e-6,1-1e-6]. 10 stable-sorted equal-count array_split bins; ECE bins rebuilt within calibration replicate. Logloss natural logs,clip1e-15. Diagnostic fits are not applied as recalibration.','rng':'NumPy default_rng PCG64','ci_percentile':'2.5/97.5 percentile with linear interpolation','Figure2_interval':'APTOS Wilson intervals unchanged in method. MESSIDOR2: fixed original stable-equal-count bin members; resample 874 examination IDs with replacement,retain all available eyes. Calculate observed event fraction within each original bin;2000 draws,seed45. Bin x-coordinates and original point estimates fixed.','Figure2_upper_panels':'Original top 1275 rows preserved pixel for pixel; A/B confusion matrices unchanged.','external_clusters':{'examinations':874,'two_image_examinations':870,'single_image_examinations':4,'images':1744},'bootstrap_external_sample_image_count_range': [min(rsizes),max(rsizes)]},'results':{'operating_points':operating,'calibration':calib,'all_models_point_estimates':brief,'Figure2_bins':reliability},'seconds':time.perf_counter()-start}
(OUT/'statistics_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('DONE',round(audit['seconds'],1),'seconds',flush=True)
