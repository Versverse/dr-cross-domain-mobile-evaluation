"""Separate raw online preprocessing from historical offline image encoding."""
from pathlib import Path
import argparse,csv,json,sys,io
from runtime_paths import load_config,load_manifest
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'deps'))
import numpy as np
from PIL import Image
import cv2
def readcsv(p):
 with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=BASE/'local_config.json');p.add_argument('--raw-results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 cfg=load_config(a.config);items,_=load_manifest(cfg,selected=True);raw_results=a.raw_results.resolve();output_dir=a.output.resolve()
 if output_dir.exists():raise FileExistsError('Use a new encoding-analysis output directory: '+str(output_dir))
 assert json.loads((raw_results/'run_status.json').read_text(encoding='utf-8'))['status']=='complete'
 output_dir.mkdir(parents=True)
 results={key:{(r['dataset'],r['image_id']):r for r in readcsv(raw_results/f'{key}_paired.csv')} for key in ['efficientnet_lite0','mobilenetv3_large']}
 output=[]
 for r in items:
  ds=r['dataset'];name=r['image_id'];rawpng=raw_results/'python_raw_processed'/ds/(name+'.png')
  with Image.open(rawpng) as im:online=np.array(im.convert('RGB'),dtype=np.uint8)
  with Image.open(r['processed_path']) as im:retained=np.array(im.convert('RGB'),dtype=np.uint8)
  diff=np.abs(online.astype(np.int16)-retained.astype(np.int16))
  suffix=Path(r['processed_path']).suffix.lower()
  ok,encoded=cv2.imencode(suffix,cv2.cvtColor(online,cv2.COLOR_RGB2BGR));assert ok
  with Image.open(io.BytesIO(encoded.tobytes())) as im:roundtrip=np.array(im.convert('RGB'),dtype=np.uint8)
  rd=np.abs(roundtrip.astype(np.int16)-retained.astype(np.int16))
  row={'dataset':ds,'image_id':name,'label':int(r['label']),'stored_extension':suffix,'selection_reason':r['selection_reason'],'online_vs_retained_max_RGB_diff':int(diff.max()),'online_vs_retained_mean_RGB_diff':float(diff.mean()),'online_vs_retained_equal_channels_fraction':float((diff==0).mean()),'after_same_extension_default_encode_decode_max_RGB_diff':int(rd.max()),'after_same_extension_default_encode_decode_identical':int(np.array_equal(roundtrip,retained))}
  for key,data in results.items():
   result=data[(ds,name)];row[f'{key}_retained_top1']=int(result['retained_top1']);row[f'{key}_raw_pytorch_top1']=int(result['pytorch_top1']);row[f'{key}_raw_tflite_top1']=int(result['tflite_top1']);row[f'{key}_raw_vs_retained_top1_changed']=int(result['pytorch_top1']!=result['retained_top1']);row[f'{key}_raw_vs_retained_max_prob_diff']=float(result['max_abs_prob_diff_pytorch_retained'])
  output.append(row)
 with (output_dir/'raw_vs_retained_preprocessing.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
 groups={}
 for ds in ['APTOS','MESSIDOR2']:
  for ext in sorted({r['stored_extension'] for r in output if r['dataset']==ds}):
   rr=[r for r in output if r['dataset']==ds and r['stored_extension']==ext]
   groups[f'{ds}_{ext}']={'N':len(rr),'identical_online_to_retained_N':sum(r['online_vs_retained_max_RGB_diff']==0 for r in rr),'max_online_to_retained_RGB_diff':max(r['online_vs_retained_max_RGB_diff'] for r in rr),'identical_after_original_extension_encode_decode_N':sum(r['after_same_extension_default_encode_decode_identical'] for r in rr),'raw_vs_retained_top1_changes':{key:sum(r[f'{key}_raw_vs_retained_top1_changed'] for r in rr) for key in results}}
 summary={'new_analysis':True,'N_frozen_images':len(output),'groups':groups,'meaning':'Raw online preprocessing omits the historical offline file-saving stage. Reproduce that stage using the original extension/default OpenCV encoding then PIL decode; identical recovered pixels identify the encoding stage. Direct raw results and encoded-file results stay separate. This subset is for paired consistency, not external population accuracy.'}
 (output_dir/'raw_encoding_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
