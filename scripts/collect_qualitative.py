"""Collect full qualitative evidence without training or changing frozen metrics."""
import hashlib, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from labkit import evaluate as ev, generate, report
from labkit.config import get_tier

def main():
    import torch
    from peft import PeftModel
    assert torch.cuda.is_available(), "Chạy trên kernel Colab GPU."
    frozen_path = ROOT / "results/baselines_frozen.json"
    verdict_path = ROOT / "results/verdict.json"
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [frozen_path,verdict_path]}
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    tier = get_tier(frozen["tier"])
    assert tier.model_id == frozen["model"]
    assert not frozen.get("smoke_mode")
    assert hashlib.sha256(generate.OPTIMIZED_PROMPT.encode()).hexdigest()[:16] == frozen["optimized_prompt_sha"]
    target = [json.loads(l) for l in (ROOT / "data/eval_target.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(target) == frozen["n_target"] == 50
    references = json.loads((ROOT / "data/checksums.json").read_text(encoding="utf-8"))
    raw = (ROOT / "data/eval_target.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == references["eval_target.jsonl"], "Eval checksum khác."
    model,tok = generate.load_base(tier)
    preds_b,_ = generate.generate_batch(model,tok,[r["input"] for r in target],system=generate.OPTIMIZED_PROMPT,label="qualitative/base-b")
    del model
    generate.free_memory()
    model,tok = generate.load_base(tier)
    model = PeftModel.from_pretrained(model,str(ROOT / "adapters/correct"))
    model.eval()
    preds_ft,_ = generate.generate_batch(model,tok,[r["input"] for r in target],system=generate.NAIVE_PROMPT,label="qualitative/ft")
    del model
    generate.free_memory()
    examples=[]
    for i,(row,b,ft) in enumerate(zip(target,preds_b,preds_ft)):
        sb=ev.triage_field_accuracy(b,row["label"])
        sf=ev.triage_field_accuracy(ft,row["label"])
        examples.append({"i":i,"ticket":row["input"],"label":row["label"],"pred_b":b,"pred_ft":ft,"score_b":sb,"score_ft":sf,"delta":sf-sb})
    observed_b=sum(r["score_b"] for r in examples)/len(examples)
    observed_ft=sum(r["score_ft"] for r in examples)/len(examples)
    expected_ft=next(r["target"] for r in verdict["comparison"] if "(c)" in r["run"])
    matches=abs(observed_b-frozen["baseline_b"]["target"])<1e-6 and abs(observed_ft-expected_ft)<1e-6
    report.write_json({"supplementary_inference":True,"baseline_target":observed_b,"ft_target":observed_ft,"matches_frozen_scores":matches},"qualitative_reproduction_check.json",results_dir=ROOT / "results")
    report.write_json(examples,"qualitative_comparison.json",results_dir=ROOT / "results")
    losses=sorted([r for r in examples if r["delta"]<0],key=lambda r:r["delta"])
    wins=sorted([r for r in examples if r["delta"]>0],key=lambda r:-r["delta"])
    selected=losses[:2]+wins[:2]
    for r in sorted(examples,key=lambda r:r["score_ft"]):
        if len(selected)>=5: break
        if r not in selected:selected.append(r)
    report.write_json(selected,"qualitative_selected.json",results_dir=ROOT / "results")
    lines=["# Bằng chứng định tính bổ sung", "", f"FT thắng: {len(wins)}; thua: {len(losses)}; hòa: {len(examples)-len(wins)-len(losses)}.", f"Điểm sinh lại khớp kết quả đóng băng: {matches}.", ""]
    for r in selected:
        lines += [f"## Ticket {r['i']}",r['ticket'],"", "Nhãn: "+json.dumps(r['label'],ensure_ascii=False),"", "Baseline (b):", "~~~json",r['pred_b'],"~~~", "Fine-tune:","~~~json",r['pred_ft'],"~~~",f"Điểm (b)={r['score_b']}; FT={r['score_ft']}; delta={r['delta']}.", ""]
    if len(losses)<2:lines += ["Không đủ hai ca FT thua trên tập eval cố định. Khai báo hạn chế này; không tạo ca thua giả."]
    if not matches:lines += ["Điểm sinh lại không khớp: đây là kết quả bổ sung, chưa thể dùng thay cho đầu ra lần đo chính. Kiểm tra phiên bản/generation trước khi kết luận."]
    (ROOT / "submission/QUALITATIVE_EVIDENCE.md").write_text("\n".join(lines),encoding="utf-8")
    for p,h in before.items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h
    print(f"Đã lưu đủ đầu ra. Baseline={observed_b}, FT={observed_ft}, khớp={matches}; ca thua={len(losses)}.")

if __name__ == "__main__":main()
