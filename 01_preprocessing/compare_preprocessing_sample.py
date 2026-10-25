"""Compare a supplied preprocessing function against retained image outputs.

No model training/inference. Without --run, select deterministic samples only.
For JPG, encode using OpenCV's default JPEG setting and decode before comparing.
Matching samples support this candidate pipeline on those images; they do not
establish which historical program created the archive.
"""
from pathlib import Path,PurePosixPath
from zipfile import ZipFile
import argparse,ast,csv,json,hashlib
from collections import defaultdict

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-script',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--processed-zip',type=Path,required=True)
    p.add_argument('--root-token',required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--per-class',type=int,default=5)
    p.add_argument('--all',action='store_true')
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    raw=defaultdict(list)
    for f in a.raw_root.rglob('*'):
        if f.is_file() and f.suffix.lower() in ['.png','.jpg','.jpeg']:raw[f.name.casefold()].append(f)
    grouped=defaultdict(list)
    for row in csv.DictReader(a.manifest.open(encoding='utf-8-sig',newline='')):grouped[int(row['label'])].append(row)
    selected=[]
    for grade,rows in sorted(grouped.items()):
        ordered=sorted(rows,key=lambda r:PurePosixPath(r['path']).name)
        for row in ordered if a.all else ordered[:a.per_class]:
            parts=PurePosixPath(row['path']).parts;name='/'.join(parts[parts.index(a.root_token):])
            hits=raw[PurePosixPath(name).name.casefold()]
            if len(hits)!=1:raise ValueError('Expected one raw image for '+name)
            selected.append({'label':grade,'raw_path':str(hits[0]),'archive_path':name})
    result={'status':'sample_selected_not_executed','sample_size':len(selected),
        'selection':'first sorted filename per class, or all when --all',
        'source_script_sha256':hashlib.sha256(a.source_script.read_bytes()).hexdigest(),'records':selected}
    if a.run:
        import cv2,numpy as np
        cv2.setNumThreads(4)
        tree=ast.parse(a.source_script.read_text(encoding='utf-8-sig'))
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='crop_and_preprocess_image']
        if len(functions)!=1:raise ValueError('Expected the reviewed original preprocessing function')
        # Load only the reviewed function, never the source file's top-level batch loop.
        namespace={'cv2':cv2,'np':np,'CROP_THRESHOLD':15}
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(a.source_script),'exec'),namespace)
        fn=namespace['crop_and_preprocess_image']
        with ZipFile(a.processed_zip) as z:
            for index,row in enumerate(selected):
                source_path=Path(row['raw_path'])
                staged=None
                # OpenCV on Windows may reject Unicode paths. Stage identical bytes
                # in the task directory; do not change the reviewed algorithm.
                if not str(source_path).isascii():
                    stage=a.report.parent/'pixel_check_inputs'
                    stage.mkdir(parents=True,exist_ok=True)
                    staged=stage/(str(index)+source_path.suffix)
                    assert str(staged.resolve()).isascii()
                    staged.write_bytes(source_path.read_bytes())
                candidate=fn(str(staged if staged else source_path),(224,224),threshold=15)
                if staged:staged.unlink()
                if candidate is None:raise ValueError('Processing failed: '+row['raw_path'])
                extension=Path(row['archive_path']).suffix.lower()
                ok,encoded=cv2.imencode(extension,candidate)
                if not ok:raise ValueError('Encoding failed')
                roundtrip=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
                retained=cv2.imdecode(np.frombuffer(z.read(row['archive_path']),dtype=np.uint8),cv2.IMREAD_COLOR)
                if retained is None:raise ValueError('Retained image decode failed')
                same_shape=roundtrip.shape==retained.shape
                difference=np.abs(roundtrip.astype(np.int16)-retained.astype(np.int16)) if same_shape else None
                row.update({'same_shape':same_shape,'roundtrip_pixel_equal':bool(same_shape and np.array_equal(roundtrip,retained)),
                    'mean_abs_pixel_difference':float(difference.mean()) if same_shape else None,
                    'max_abs_pixel_difference':int(difference.max()) if same_shape else None})
                if (index+1)%100==0:print(f'Compared {index+1}/{len(selected)} images',flush=True)
        result.update({'status':'candidate_preprocessing_compared','opencv_version':cv2.__version__,
            'numpy_version':np.__version__,'opencv_threads':cv2.getNumThreads(),
            'exact_pixel_matches':sum(r['roundtrip_pixel_equal'] for r in selected)})
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))

if __name__=='__main__':main()
