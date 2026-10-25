"""Rebuild archive membership from saved manifests, never re-randomize a split.

This is a new curation utility, not the recovered historical packaging script.
With no --output it checks archive membership and produces no image archive.
"""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZIP_STORED
import argparse, csv, json, hashlib
from collections import Counter

def expected_path(value,root):
    parts=PurePosixPath(value.replace('\\','/')).parts
    if root not in parts:raise ValueError(f'Manifest path lacks root token {root}')
    parts=parts[parts.index(root):]
    if any(p in ('','..','.') for p in parts):raise ValueError('Unsafe archive member path')
    return '/'.join(parts)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-zip',type=Path,required=True)
    p.add_argument('--manifest',type=Path,action='append',required=True)
    p.add_argument('--root-token',required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.output and (a.output.exists() or a.output.resolve()==a.input_zip.resolve()):
        raise ValueError('Output must be a new file; never replace the input archive')
    expected={};counts={}
    for m in a.manifest:
        rows=list(csv.DictReader(m.open(encoding='utf-8-sig',newline='')))
        for row in rows:
            name=expected_path(row['path'],a.root_token)
            if name in expected:raise ValueError(f'Duplicate manifest member: {name}')
            folder=PurePosixPath(name).parent.name
            if not folder.startswith(str(int(row['label']))+'_'):raise ValueError('Class folder/label mismatch')
            expected[name]=int(row['label'])
        counts[m.name]={'n':len(rows),'by_label':dict(Counter(int(r['label']) for r in rows)),
                        'sha256':hashlib.sha256(m.read_bytes()).hexdigest()}
    with ZipFile(a.input_zip) as z:
        actual={i.filename.replace('\\','/'):i for i in z.infolist() if not i.is_dir() and Path(i.filename).suffix.lower() in ['.png','.jpg','.jpeg']}
        missing=sorted(set(expected)-set(actual));extra=sorted(set(actual)-set(expected))
        if missing or extra:raise ValueError(f'Membership differs: {len(missing)} missing, {len(extra)} extra')
        # Validate all selected members' CRC by streaming them, without extracting originals.
        members=[]
        for name in sorted(expected):
            h=hashlib.sha256()
            with z.open(actual[name]) as f:
                while True:
                    block=f.read(1024*1024)
                    if not block:break
                    h.update(block)
            members.append({'path':name,'label':expected[name],'sha256':h.hexdigest()})
        if a.output:
            a.output.parent.mkdir(parents=True,exist_ok=True)
            with ZipFile(a.output,'x',compression=ZIP_STORED) as out:
                for row in members:out.writestr(row['path'],z.read(actual[row['path']]))
            with ZipFile(a.output) as out:
                assert out.testzip() is None
                assert set(out.namelist())==set(expected)
                for row in members:assert hashlib.sha256(out.read(row['path'])).hexdigest()==row['sha256']
    report={'utility':'New manifest-based archive reconstruction','re_randomized':False,
            'preprocessing_performed':False,'manifests':counts,'images':len(members),
            'membership_exact_match':True,'all_member_crcs_checked':True,'members':members,
            'new_archive_written':str(a.output) if a.output else None}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['members','manifests']},ensure_ascii=False))

if __name__=='__main__':main()
