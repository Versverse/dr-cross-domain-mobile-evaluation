"""Recover full-precision source thresholds under an explicit rule.

This new calculation verifies table-5 point estimates. It is not a claim that
the missing historical selection source code has been found.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
B=Path(__file__).resolve().parent
S=B.parent/'inputs'
O=B.parent/'results';O.mkdir(exist_ok=True)
data={}
hashes={}
for name,file in [('APTOS','aptos_val_all_predictions_combined.csv'),('MESSIDOR2','messidor2_all_predictions_combined.csv')]:
 path=S/file;hashes[file]=hashlib.sha256(path.read_bytes()).hexdigest()
 d=pd.read_csv(path);d=d.loc[d.model=='MobileViT-S'].copy()
 data[name]=((d.y_true.to_numpy()>=2),d[['prob_2','prob_3','prob_4']].sum(axis=1).to_numpy())
old=[[0.5399,90.27,95.63,38.73,97.67],[0.3802,95.30,92.64,45.73,95.34],[0.2538,98.32,91.26,55.14,92.62],[0.1378,99.33,87.13,66.52,88.19]]
rows=[]
ya,pa=data['APTOS'];positive_scores=np.sort(pa[ya])[::-1]
for i,target in enumerate([.90,.95,.98,.99]):
 # The largest source threshold attaining the target; predictions p>=t.
 threshold=float(positive_scores[int(np.ceil(target*ya.sum()))-1])
 row={'target_APTOS_sensitivity':target,'full_precision_threshold':threshold}
 v=[]
 for name,(y,p) in data.items():
  pred=p>=threshold
  sensitivity=float(np.mean(pred[y]));specificity=float(np.mean(~pred[~y]))
  row[name+'_TP']=int(np.sum(pred&y));row[name+'_positive_N']=int(y.sum())
  row[name+'_TN']=int(np.sum(~pred&~y));row[name+'_negative_N']=int((~y).sum())
  row[name+'_sensitivity']=sensitivity;row[name+'_specificity']=specificity
  v.extend([round(100*sensitivity,2),round(100*specificity,2)])
 assert round(threshold,4)==old[i][0] and v==old[i][1:],(target,threshold,v)
 rows.append(row)
pd.DataFrame(rows).to_csv(O/'verified_full_precision_operating_points.csv',index=False)
out={'date':'2026-10-14','scope':'New arithmetic verification, no inference or training. All four Table 5 thresholds (4 decimals) and all 16 sensitivity/specificity point estimates (2 decimals) match. Original historical code and bootstrap intervals are not verified by this script.','selection_rule':'Choose largest APTOS referable risk threshold with empirical sensitivity at least target; include p>=threshold. Selection uses source labels only; apply unchanged to MESSIDOR-2.','risk':'sum(prob_2,prob_3,prob_4)','source_sha256':hashes,'results':rows}
(O/'operating_point_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
