"""Build Option A submission with cleared notebook outputs and atomic ZIP replacement."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib, json
ROOT=Path(__file__).resolve().parents[1]
PREFIX='lab21_2A202602428/'
OUT=ROOT/'exports';OUT.mkdir(exist_ok=True)
DEST=OUT/'lab21_2A202602428.zip'
TEMP=OUT/'lab21_2A202602428.tmp.zip'
files=[ROOT/'submission'/f for f in ['REPORT.md','REFLECTION.md','QUALITATIVE_EVIDENCE.md','LINKS.md']]
files+=sorted((ROOT/'results').glob('*.json'))+[ROOT/'results/runs.csv']
files+=[ROOT/'adapters/correct/adapter_config.json',ROOT/'adapters/correct/adapter_model.safetensors']
files+=sorted((ROOT/'notebooks').glob('*.py'))
files+=[ROOT/'scripts/collect_qualitative.py',ROOT/'scripts/collect_regression_examples.py']
assert all(f.is_file() for f in files)
with ZipFile(TEMP,'w',ZIP_DEFLATED,compresslevel=1) as z:
 for f in files:z.write(f,PREFIX+f.relative_to(ROOT).as_posix())
 for name in ['Lab21_06_merge_and_serve.ipynb','Lab21_07_qualitative_evidence.ipynb','Lab21_09_regression_evidence.ipynb']:
  nb=json.loads((ROOT/'colab'/name).read_text(encoding='utf-8'))
  nb['metadata'].pop('widgets',None)
  for c in nb['cells']:
   if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
  z.writestr(PREFIX+'notebooks/'+name,json.dumps(nb,ensure_ascii=False,indent=1))
with ZipFile(TEMP) as z:
 assert z.testzip() is None
 for name in ['results/regression_selected_losses.json','results/merge_check.json','results/hf_hub.json','submission/REPORT.md']:
  assert PREFIX+name in z.namelist()
 for name in z.namelist():
  if name.endswith('.ipynb'):
   nb=json.loads(z.read(name))
   assert all(not c['outputs'] and c['execution_count'] is None for c in nb['cells'] if c['cell_type']=='code')
 count=len(z.namelist())
TEMP.replace(DEST)
with DEST.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
(OUT/'lab21_2A202602428.zip.sha256').write_text(digest+'  '+DEST.name+'\n',encoding='utf-8')
print(f'Package verified: {count} files; {DEST.stat().st_size/1024**2:.2f} MiB')
print(DEST)
