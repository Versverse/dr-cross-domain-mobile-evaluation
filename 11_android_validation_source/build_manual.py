"""Portable Windows SDK-only rebuild adaptation; creates a new local debug key.
This sharing copy was syntax-checked, not rebuilt during packaging.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import os, subprocess, shutil, json, hashlib, time

SOURCE=Path(__file__).resolve().parent
BASE=Path(os.environ.get('FIDELITY_BUILD_ROOT',str(SOURCE/'rebuild_local'))).resolve()
if BASE.exists():raise FileExistsError('Use a fresh FIDELITY_BUILD_ROOT; previous builds are not overwritten.')
BASE.mkdir(parents=True)
SDK=Path(os.environ['ANDROID_SDK_ROOT']).resolve()
BUILD=SDK/'build-tools/36.0.0'
JBR=Path(os.environ['JAVA_HOME']).resolve()
AAR_ROOT=Path(os.environ['FIDELITY_AAR_ROOT']).resolve()
RECORDED=json.loads((SOURCE/'build_integrity.json').read_text(encoding='utf8'))
ANDROID=SDK/'platforms/android-36.1/android.jar'
PACKAGE='com.example.aptosedgetest.fidelity20261016'
for name in ['deps','classes','dex','assets','native','logs']:(BASE/name).mkdir(exist_ok=True)
events=[]
def long(p):
    value=str(p.resolve());return Path(value if value.startswith('\\\\?\\') else '\\\\?\\'+value)
def sha(p):return hashlib.sha256(long(p).read_bytes()).hexdigest()
def run(args,name):
    start=time.time()
    result=subprocess.run([str(x) for x in args],cwd=BASE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=200)
    text=result.stdout.decode('utf8',errors='replace');(BASE/'logs'/f'{name}.txt').write_text(text,encoding='utf8')
    events.append({'step':name,'exit_code':result.returncode,'seconds':round(time.time()-start,3)})
    print(name,result.returncode,text[-3000:],flush=True)
    if result.returncode:raise RuntimeError(f'{name} failed; see logs/{name}.txt')
sources=[]
for name in ['opencv-4.10.0.aar','tensorflow-lite-api-2.16.1.aar','tensorflow-lite-2.16.1.aar']:
    cache=long(AAR_ROOT)
    candidates=list(cache.rglob(name));assert len(candidates)==1,(name,candidates)
    archive=candidates[0]; assert hashlib.sha256(archive.read_bytes()).hexdigest()==next(x['sha256'] for x in RECORDED['aar_sources'] if x['name']==name); sources.append({'name':name,'source':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
    with ZipFile(archive) as z:
        (BASE/'deps'/name.replace('.aar','.jar')).write_bytes(z.read('classes.jar'))
        for item in z.namelist():
            if item.startswith('jni/arm64-v8a/') and item.endswith('.so'):
                dest=BASE/'native/lib/arm64-v8a'/Path(item).name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(item))
asset_source=SOURCE.parent/'03_android/AptosEdgeTest/app/src/main/assets'
config=SOURCE/'runtime_config.json';shutil.copy2(config,BASE/'assets/runtime_config.json')
spec=json.loads(config.read_text(encoding='utf8'))
for name,s in spec.items():
    f=asset_source/s['asset'];assert sha(f)==s['sha256'];shutil.copy2(f,BASE/'assets'/f.name)
class_path=';'.join(str(p) for p in [ANDROID,*sorted((BASE/'deps').glob('*.jar'))])
run([JBR/'bin/javac.exe','-encoding','UTF-8','-source','8','-target','8','-classpath',class_path,'-d',BASE/'classes',SOURCE/'FidelityActivity.java'],'01_javac')
jar=BASE/'activity.jar'
with ZipFile(jar,'w',ZIP_DEFLATED) as z:
    for f in (BASE/'classes').rglob('*.class'):z.write(f,f.relative_to(BASE/'classes').as_posix())
run([JBR/'bin/java.exe','-cp',BUILD/'lib/d8.jar','com.android.tools.r8.D8','--min-api','26','--lib',ANDROID,'--output',BASE/'dex',jar,*sorted((BASE/'deps').glob('*.jar'))],'02_d8')
unsigned=BASE/'fidelity-unsigned.apk'
run([BUILD/'aapt2.exe','link','-I',ANDROID,'--manifest',SOURCE/'AndroidManifest.xml','--version-code','1','--version-name','0.1-experiment','-A',BASE/'assets','-o',unsigned],'03_aapt2')
with ZipFile(unsigned,'a',ZIP_DEFLATED) as z:
    for f in sorted((BASE/'dex').glob('*.dex')):z.write(f,f.name)
    for f in sorted((BASE/'native').rglob('*.so')):z.write(f,f.relative_to(BASE/'native').as_posix())
aligned=BASE/'fidelity-aligned.apk'
run([BUILD/'zipalign.exe','-f','-p','4',unsigned,aligned],'04_zipalign')
keystore=BASE/'local_debug.keystore'
if not keystore.exists():
    run([JBR/'bin/keytool.exe','-genkeypair','-keystore',keystore,'-storepass','android','-keypass','android','-alias','localdebug','-dname','CN=Local Fidelity Debug,O=Local Research,C=XX','-keyalg','RSA','-keysize','2048','-validity','3650','-noprompt'],'05_local_debug_key')
apk=BASE/'DR_Fidelity_native_debug_20261016.apk'
run([JBR/'bin/java.exe','-jar',BUILD/'lib/apksigner.jar','sign','--ks',keystore,'--ks-key-alias','localdebug','--ks-pass','pass:android','--key-pass','pass:android','--out',apk,aligned],'06_apksigner')
run([JBR/'bin/java.exe','-jar',BUILD/'lib/apksigner.jar','verify','--verbose','--print-certs',apk],'07_verify_signature')
run([BUILD/'aapt2.exe','dump','badging',apk],'08_dump_badging')
with ZipFile(apk) as z:
    assert z.testzip() is None
    for name,s in spec.items():assert hashlib.sha256(z.read('assets/'+s['asset'])).hexdigest()==s['sha256']
    entries=z.namelist();assert 'classes.dex' in entries and 'lib/arm64-v8a/libopencv_java4.so' in entries
report={'status':'BUILT_AND_SIGNATURE_VERIFIED_NOT_DEVICE_RUN','package':PACKAGE,'apk':str(apk),'apk_sha256':sha(apk),'bytes':apk.stat().st_size,'ABI':['arm64-v8a'],'models':spec,'input_manifest':'Prepared separately with prepare_device_inputs.py; see ../12_android_fidelity/completed_records/manifest.tsv for recorded identity','original_source_sha256':RECORDED['original_source_sha256'],'java_source_sha256':sha(SOURCE/'FidelityActivity.java'),'aar_sources':sources,'SDK':str(SDK),'build_tools':'36.0.0','java':str(JBR),'steps':events,'device_run':'This rebuild program does not execute a device. Original execution is recorded in the source build_integrity.json/actual_device_run.','timing':'Diagnostic debug observations only; no new performance claims.','private_key':'New local debug key remains in build directory only; never include local_debug.keystore in distribution archives.'}
(BASE/'build_integrity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'apk':str(apk),'sha256':report['apk_sha256'],'status':report['status'],'bytes':report['bytes']},ensure_ascii=False),flush=True)
