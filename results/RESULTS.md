# KernelBench-Harbor — Results

**Setup.** 10 operator families (1 representative task each), agentic Harbor runs on Modal
**L40S**, agent = `terminus-2`, **max_turns = 4**, **2 samples/task**. Same `instruction.md` +
verifier for all 3 models; only the model changes. **Pass = correct** (vs fp32 ground truth)
**AND faster than `torch.compile(max-autotune)`** (eager speedup also reported), with an AST
anti-cheat (real `@triton.jit`, no matmul/softmax/sdpa/cuBLAS offload).

pass@1 = single-sample pass rate (20 samples/model); pass@2 = ≥1 of the 2 samples passes.
Reproduce from the run logs: `python results/compute_passk.py`.

## pass@1 / pass@2

| Model | pass@1 | pass@2 |
|---|:--:|:--:|
| **GPT-5.5** | **40%** (8/20) | **50%** (5/10) |
| **Opus-4.8** | **25%** (5/20) | **40%** (4/10) |
| **Gemini-3.5-flash** | **5%** (1/20) | **10%** (1/10) |

### Per-family pass@2

| Family (real config) | GPT-5.5 | Opus-4.8 | Gemini-3.5-flash |
|---|:--:|:--:|:--:|
| fp8_microscale (Llama-3-8B down) | ✅ | ✗ | ✅ |
| moe_top2 (Mixtral-8x7B) | ✅ | ✅ | ✗ |
| ssm_scan (Mamba-130M) | ✅ | ✅ | ✗ |
| mla_decode (DeepSeek-V2) | ✗ | ✅ | ✗ |
| w8a8_smoothquant (Llama-3-8B down) | ✅ | ✗ | ✗ |
| fused_silu_fp8 (Mixtral) | ✗ | ✅ | ✗ |
| nsa_block_sparse (DeepSeek NSA) | ✅ | ✗ | ✗ |
| quant_w4a16_pa (Llama-3-8B down) | ✗ | ✗ | ✗ |
| fused_qmoe (Mixtral) | ✗ | ✗ | ✗ |
| multi_lora_qgemm (Llama-3-8B) | ✗ | ✗ | ✗ |

All three land in the trial's **10–60%** in-distribution band, with a clean gradient
GPT > Opus > flash and no family solved by all three.

## Insights

- **Healthy difficulty spread + a hard tail.** Six families are solved by ≥1 model; three
  (`quant_w4a16_pa`, `fused_qmoe`, `multi_lora_qgemm`) are unsolved by every model. All 10 ship
  a **pass-validated oracle**, so even the hard tail is provably solvable — the difficulty is in
  the operator, not a broken gate.

- **Agentic helps strong models, hurts weak ones.** GPT-5.5 / Opus-4.8 use the multi-turn loop to
  fix compile errors and iterate; Gemini-flash degrades under the agent protocol (it repeatedly
  copied the instruction's placeholder instead of real code). flash's agentic 10% is **below** its
  single-shot rate — iterative refinement only helps a model that can drive the terminal. (Same
  instruction for all three → a capability gap, not an unfair prompt.)

- **Difficulty tracks baseline strength.** We keep families where `torch.compile` / cuBLAS leave
  real fusion headroom (quant GEMM, MoE, FP8, MLA, NSA); we drop weak-baseline ops (vanilla
  attention, naive SSM) that frontier models clear in one shot.

## Verifier / QC (why pass/fail is trustworthy)

- **Objective** — pass/fail = output matches the fp32 reference **and** beats `torch.compile`. No
  judgment calls.
- **Solvable & sound** — every family has a known-good oracle that passes the verifier, so it
  provably accepts correct kernels and each task is achievable.
- Families whose *output* is itself quantized / top-k-selected (`fused_silu_fp8`,
  `nsa_block_sparse`) are graded with a **cosine/relative** metric — how such kernels are
  validated in practice; the rest use `allclose`.

## Scope
132 tasks generated across the 10 families (parameter grids over real HuggingFace configs); the
10-task subset above is the measured sample. Full agentic trajectories are in `results/jobs/`.
