"""New, paired inference only; original checkpoints and final TFLite stay read-only.
Portable adaptation of the program executed on 2026-10-16; paths and output location are configurable.
Final canonical-threshold postprocessing is integrated here; no training or threshold fitting occurs.
"""
from pathlib import Path
import argparse,csv,json,hashlib,sys,time,platform,importlib.metadata,os
BASE=Path(__file__).resolve().parent
from runtime_paths import load_config, load_manifest, frozen_records
DEPS=BASE/'deps'
if DEPS.is_dir():sys.path.insert(0,str(DEPS))
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import timm
try:
 from ai_edge_litert.interpreter import Interpreter
 RUNTIME='ai-edge-litert'
except ImportError:
 import tensorflow as tf
 Interpreter=tf.lite.Interpreter
 RUNTIME='tensorflow'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
class Wrapper(nn.Module):
 def __init__(self,model):
  super().__init__();self.model=model
  self.register_buffer('mean',torch.tensor([.485,.456,.406],dtype=torch.float32).view(1,3,1,1))
  self.register_buffer('std',torch.tensor([.229,.224,.225],dtype=torch.float32).view(1,3,1,1))
 def forward(self,x):return self.model((x.permute(0,3,1,2)-self.mean)/self.std)
def state(raw):
 for k in ['state_dict','model_state_dict','model']:
  if k in raw and isinstance(raw[k],dict):raw=raw[k];break
 return {str(k).removeprefix('module.'):v for k,v in raw.items() if hasattr(v,'shape')}
def softmax(logits):
 x=np.asarray(logits,dtype=np.float64);v=np.exp(x-x.max());return v/v.sum()
def versions():
 out={}
 for key in ['torch','torchvision','timm','numpy','Pillow','tensorflow','tensorflow-cpu','ai-edge-litert']:
  try:out[key]=importlib.metadata.version(key)
  except importlib.metadata.PackageNotFoundError:pass
 return out
def raw_rgb(path):
 import cv2
 bgr=cv2.imdecode(np.fromfile(Path(path),dtype=np.uint8),cv2.IMREAD_COLOR)
 assert bgr is not None
 gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);_,mask=cv2.threshold(gray,15,255,cv2.THRESH_BINARY)
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 cropped=bgr
 if contours:
  x,y,w,h=cv2.boundingRect(max(contours,key=cv2.contourArea));cx=x+w//2;cy=y+h//2;s=max(w,h)
  x1=max(0,cx-s//2);x2=min(bgr.shape[1],cx+s//2);y1=max(0,cy-s//2);y2=min(bgr.shape[0],cy+s//2)
  if x2>x1 and y2>y1:cropped=bgr[y1:y2,x1:x2]
 resized=cv2.resize(cropped,(224,224),interpolation=cv2.INTER_AREA)
 lab=cv2.cvtColor(resized,cv2.COLOR_BGR2LAB);lab[:,:,0]=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(lab[:,:,0])
 return cv2.cvtColor(cv2.cvtColor(lab,cv2.COLOR_LAB2BGR),cv2.COLOR_BGR2RGB)
def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--raw',action='store_true');p.add_argument('--selected',action='store_true');p.add_argument('--models',nargs='+');p.add_argument('--threads',type=int,default=4);p.add_argument('--config',type=Path,default=BASE/'local_config.json');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 cfg=load_config(a.config);thresholds,canonical_risks=frozen_records(cfg)
 items,manifest_sha=load_manifest(cfg,selected=a.raw or a.selected)
 if a.smoke:
  items=[next(r for r in items if r['dataset']==ds and int(r['label'])==k) for ds in ['APTOS','MESSIDOR2'] for k in range(5)]
 out=a.output.resolve()
 if out.exists():raise FileExistsError('Use a new output directory; completed evidence is never overwritten: '+str(out))
 out.mkdir(parents=True)
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(20261016)
 retained={}
 for ds,entry in cfg['source_prediction_csv'].items():
  assert sha(entry['path'])==entry['sha256']
  with Path(entry['path']).open(encoding='utf-8-sig',newline='') as f:
   for row in csv.DictReader(f):retained[(ds,row['model'],row['image_id'])]=row
 run={'new_experiment':True,'training_performed':False,'process_id':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'python':sys.version,'platform':platform.platform(),'versions':versions(),'runtime':RUNTIME,'threads':a.threads,'input_manifest_sha256':manifest_sha,'canonical_decisions_applied':cfg['frozen_thresholds_sha256'],'canonical_risks_sha256':cfg['canonical_retained_risks_sha256'],'sharing_adaptation':'Relative data-root configuration; canonical retained-risk postprocessing integrated; inference/preprocessing retained','N_per_model':len(items),'mode':'smoke' if a.smoke else ('selected_frozen' if a.selected or a.raw else 'full'),'input_mode':'raw_online_no_offline_encoding' if a.raw else 'retained_preprocessed_file_decoded_RGB','completed_models':{},'status':'running'}
 (out/'run_status.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
 for key in (a.models or list(cfg['models'])):
  spec=cfg['models'][key];assert sha(spec['checkpoint'])==spec['checkpoint_sha256'];assert sha(spec['tflite'])==spec['tflite_sha256']
  model=timm.create_model(spec['timm_name'],pretrained=False,num_classes=5)
  model.load_state_dict(state(torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)),strict=True)
  wrapped=Wrapper(model).eval()
  # Native Windows runtime file loader can misdecode CJK paths; bytes preserve the exact artifact.
  interpreter=Interpreter(model_content=Path(spec['tflite']).read_bytes(),num_threads=a.threads);interpreter.allocate_tensors()
  inp=interpreter.get_input_details()[0];op=interpreter.get_output_details()[0]
  assert list(inp['shape'])==[1,224,224,3] and inp['dtype']==np.float32
  assert int(np.prod(op['shape']))==5 and op['dtype']==np.float32
  fields=['dataset','image_id','label','processed_sha256','input_tensor_sha256']+[f'{side}_{kind}_{k}' for side in ['pytorch','tflite'] for kind in ['logit','prob'] for k in range(5)]+['pytorch_top1','tflite_top1','top1_agree','pytorch_risk','tflite_risk','max_abs_logit_diff','max_abs_prob_diff']+[f'retained_prob_{k}' for k in range(5)]+['retained_top1','retained_risk_sum_float64','retained_referable_score_column','pytorch_retained_top1_agree','max_abs_prob_diff_pytorch_retained','pytorch_minus_retained_risk']
  for rule in cfg['thresholds'][key]:fields.extend([f'{rule}_threshold',f'{rule}_pytorch_decision',f'{rule}_tflite_decision',f'{rule}_agree',f'{rule}_pytorch_margin',f'{rule}_retained_decision',f'{rule}_pytorch_retained_agree'])
  summaries={ds:{'N':0,'top1_matches':0,'max_abs_logit_diff':0.,'sum_abs_logit_diff':0.,'max_abs_prob_diff':0.,'sum_abs_prob_diff':0.,'pytorch_retained_top1_matches':0,'max_abs_prob_diff_pytorch_retained':0.,'decision_matches':{r:0 for r in cfg['thresholds'][key]},'pytorch_retained_decision_matches':{r:0 for r in cfg['thresholds'][key]}} for ds in ['APTOS','MESSIDOR2']}
  started=time.monotonic()
  with (out/f'{key}_paired.csv').open('w',encoding='utf-8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   for i,r in enumerate(items):
    assert sha(r['processed_path'])==r['processed_sha256']
    if a.raw:
     assert sha(r['raw_path'])==r['raw_sha256'];arr=raw_rgb(r['raw_path'])
     png=out/'python_raw_processed'/r['dataset']/(r['image_id']+'.png');png.parent.mkdir(exist_ok=True,parents=True)
     if not png.exists():Image.fromarray(arr).save(png)
    else:
     with Image.open(r['processed_path']) as im:arr=np.array(im.convert('RGB'),dtype=np.uint8)
    assert arr.shape==(224,224,3)
    x=np.ascontiguousarray(arr.astype(np.float32)[None]/np.float32(255.0))
    with torch.inference_mode():pt=wrapped(torch.from_numpy(x)).numpy().reshape(-1).astype(np.float64)
    interpreter.set_tensor(inp['index'],x);interpreter.invoke();lt=interpreter.get_tensor(op['index']).reshape(-1).astype(np.float64)
    assert np.isfinite(pt).all() and np.isfinite(lt).all()
    pp=softmax(pt);lp=softmax(lt);pr=float(pp[2:].sum());lr=float(lp[2:].sum())
    old=retained[(r['dataset'],spec['display'],r['image_id'])];oldp=np.array([float(old[f'prob_{k}']) for k in range(5)]);oldrisk=canonical_risks[(r['dataset'],key,r['image_id'])]
    row={k:r[k] for k in ['dataset','image_id','label','processed_sha256']};row['input_tensor_sha256']=hashlib.sha256(x.tobytes()).hexdigest()
    for side,logits,probs in [('pytorch',pt,pp),('tflite',lt,lp)]:
     for j in range(5):row[f'{side}_logit_{j}']=float(logits[j]);row[f'{side}_prob_{j}']=float(probs[j])
    row.update(pytorch_top1=int(pt.argmax()),tflite_top1=int(lt.argmax()),top1_agree=int(pt.argmax()==lt.argmax()),pytorch_risk=pr,tflite_risk=lr,max_abs_logit_diff=float(np.abs(pt-lt).max()),max_abs_prob_diff=float(np.abs(pp-lp).max()))
    for j in range(5):row[f'retained_prob_{j}']=float(oldp[j])
    row.update(retained_top1=int(old['y_pred']),retained_risk_sum_float64=oldrisk,retained_referable_score_column=float(old['referable_score']),pytorch_retained_top1_agree=int(pt.argmax()==int(old['y_pred'])),max_abs_prob_diff_pytorch_retained=float(np.abs(pp-oldp).max()),pytorch_minus_retained_risk=pr-oldrisk)
    s=summaries[r['dataset']];s['N']+=1;s['top1_matches']+=row['top1_agree'];s['max_abs_logit_diff']=max(s['max_abs_logit_diff'],row['max_abs_logit_diff']);s['max_abs_prob_diff']=max(s['max_abs_prob_diff'],row['max_abs_prob_diff']);s['sum_abs_logit_diff']+=float(np.abs(pt-lt).sum());s['sum_abs_prob_diff']+=float(np.abs(pp-lp).sum())
    s['pytorch_retained_top1_matches']+=row['pytorch_retained_top1_agree'];s['max_abs_prob_diff_pytorch_retained']=max(s['max_abs_prob_diff_pytorch_retained'],row['max_abs_prob_diff_pytorch_retained'])
    for rule,t in cfg['thresholds'][key].items():
     agree=int((pr>=t)==(lr>=t));old_agree=int((pr>=t)==(oldrisk>=t));row.update({f'{rule}_threshold':t,f'{rule}_pytorch_decision':int(pr>=t),f'{rule}_tflite_decision':int(lr>=t),f'{rule}_agree':agree,f'{rule}_pytorch_margin':pr-t,f'{rule}_retained_decision':int(oldrisk>=t),f'{rule}_pytorch_retained_agree':old_agree});s['decision_matches'][rule]+=agree;s['pytorch_retained_decision_matches'][rule]+=old_agree
    w.writerow(row)
    if i%100==0:f.flush();print(f'{key} {i+1}/{len(items)}',flush=True)
  for s in summaries.values():
   s['mean_abs_logit_diff']=s.pop('sum_abs_logit_diff')/(5*s['N']);s['mean_abs_prob_diff']=s.pop('sum_abs_prob_diff')/(5*s['N'])
  run['completed_models'][key]={'spec':spec,'elapsed_s':time.monotonic()-started,'by_dataset':summaries,'csv_sha256':sha(out/f'{key}_paired.csv')}
  (out/'run_status.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
  print(json.dumps({key:summaries},indent=2),flush=True)
  del wrapped,model,interpreter
 run['status']='complete';run['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 (out/'run_status.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
if __name__=='__main__':main()
