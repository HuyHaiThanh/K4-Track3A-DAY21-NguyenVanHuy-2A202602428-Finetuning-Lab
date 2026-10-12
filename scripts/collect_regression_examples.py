import pathlib,sys,json,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from labkit import generate,evaluate as ev,report
from labkit.config import get_tier
from peft import PeftModel
frozen_path=ROOT/'results/baselines_frozen.json'
verdict_path=ROOT/'results/verdict.json'
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [frozen_path,verdict_path]}
frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
verdict=json.loads(verdict_path.read_text(encoding='utf-8'))
tier=get_tier(frozen['tier'])
assert tier.model_id==frozen['model'] and not frozen.get('smoke_mode')
rows=[json.loads(l) for l in (ROOT/'data/eval_regression.jsonl').read_text(encoding='utf-8').splitlines() if l]
assert len(rows)==15
model,tok=generate.load_base(tier)
base,_=generate.generate_batch(model,tok,[r['instruction'] for r in rows],system=None,max_new_tokens=96,label='regression/base')
del model;generate.free_memory()
model,tok=generate.load_base(tier)
model=PeftModel.from_pretrained(model,str(ROOT/'adapters/correct'));model.eval()
ft,_=generate.generate_batch(model,tok,[r['instruction'] for r in rows],system=None,max_new_tokens=96,label='regression/ft')
del model;generate.free_memory()
examples=[]
for i,(r,b,p) in enumerate(zip(rows,base,ft)):
 sb=ev.keyword_recall(b,r['keywords']);sf=ev.keyword_recall(p,r['keywords'])
 examples.append({'i':i,'question':r['instruction'],'keywords':r['keywords'],'pred_b':b,'pred_ft':p,'score_b':sb,'score_ft':sf,'delta':sf-sb})
bmean=sum(r['score_b'] for r in examples)/15
fmean=sum(r['score_ft'] for r in examples)/15
expected=next(r['regression'] for r in verdict['comparison'] if '(c)' in r['run'])
matched=abs(bmean-frozen['baseline_b']['regression'])<1e-6 and abs(fmean-expected)<0.0001
losing=sorted([r for r in examples if r['delta']<0],key=lambda r:r['delta'])
report.write_json(examples,'regression_comparison.json',results_dir=ROOT/'results')
report.write_json({'matches_frozen_scores':matched,'baseline_regression':bmean,'ft_regression':fmean,'loss_count':len(losing)},'regression_reproduction_check.json',results_dir=ROOT/'results')
report.write_json(losing[:2],'regression_selected_losses.json',results_dir=ROOT/'results')
for p,h in hashes.items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h
print(f'Baseline={bmean}; FT={fmean}; khớp={matched}; ca thua={len(losing)}')
assert matched,'Điểm bổ sung không khớp; kiểm tra môi trường trước khi dùng trong report.'
assert len(losing)>=2,'Không đủ hai ca regression thua trên tập cố định.'
