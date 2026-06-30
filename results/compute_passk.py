"""Reproduce the pass@1 / pass@2 table in RESULTS.md from the archived Harbor logs.

    python results/compute_passk.py

Run logs: results/jobs/<Model>/sample{1,2}/<task>/{trajectory.json,reward.json}. Each reward.json
is the verifier's grade under that family's declared metric (cosine for quantized / top-k-selected
outputs, allclose otherwise). pass@1 = single-sample pass rate (20 samples/model); pass@2 = ≥1 of
the 2 samples passes.
"""
import os
import json
import glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NICE = {"gpt55": "GPT-5.5", "opus": "Opus-4.8", "gemini": "Gemini-3.5-flash"}
FAMS = ["quant_w4a16_pa", "fp8_microscale", "moe_top2", "ssm_scan", "mla_decode",
        "w8a8_smoothquant", "fused_silu_fp8", "fused_qmoe", "multi_lora_qgemm", "nsa_block_sparse"]


def samples(short, fam):
    out = []
    for n in (1, 2):
        for j in glob.glob(os.path.join(REPO, "results", "jobs", NICE[short], f"sample{n}", fam + "*", "reward.json")):
            try:
                out.append(int(json.load(open(j)).get("reward") or 0))
            except Exception:
                pass
    return out[:2]


print(f"{'family':<20}{'GPT':<12}{'Opus':<12}{'flash':<12}")
agg = {m: [0, 0, 0] for m in NICE}     # pass@1 num, denom, pass@2
for fam in FAMS:
    row = f"{fam:<20}"
    for m in NICE:
        s = (samples(m, fam) + [0, 0])[:2]
        agg[m][0] += sum(s); agg[m][1] += 2; agg[m][2] += (1 if any(s) else 0)
        row += f"{str(s):<12}"
    print(row)
print()
for m in NICE:
    n, d, p2 = agg[m]
    print(f"  {NICE[m]:<17} pass@1 = {n}/{d} = {100*n//d}%    pass@2 = {p2}/10 = {10*p2}%")
