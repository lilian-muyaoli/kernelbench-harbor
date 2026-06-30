# Single-shot results (KernelBench-comparable) — the capability gap

**Setup.** Model writes ONE kernel from the PyTorch reference (no iteration / no feedback),
exactly KernelBench's protocol. Pass = correct (vs fp32 GT) AND faster than
`torch.compile(max-autotune)`. pass@1 over 2 samples. Eval on Modal L40S via OpenRouter.

This is the headline difficulty measurement. Single-shot, frontier models **fail at a high,
nontrivial rate** — squarely in the trial's 10–60% in-distribution band.

## Hard families we keep (single-shot pass@1)

| Family (real config) | GPT-5.5 | Opus-4.8 | Gemini-3.5 | dominant failure mode |
|---|---|---|---|---|
| W4A16 packed-asym quant | 0/2 | 1/2 | 0–1/2 | compile error / numerical contract |
| W8A8 SmoothQuant | 0/2 | — | — | int8→fp16 scale-contract dtype bug |
| FP8 microscaling GEMM | 1/2 | — | — | fp8 unpack/scale compile error |
| MoE top-k dispatch | 1/2 | 0/2 | 0/2 | correct but too slow / wrong routing |
| MLA decode (absorb) | 2–3/5 | — | — | compile error; misses absorb trick |
| SSM scan (weak baseline) | 1/2 | 1/2 | 2/2 | sample-to-sample incorrectness |

**Per-model single-shot pass@1 (4-task matrix, comparable cells):** GPT-5.5 ≈ **25%**,
Opus-4.8 ≈ **38%**, Gemini-3.5 ≈ **38%**. All in the 10–60% band; no model dominates;
no task trivially solved by all three.

## Why it's hard (interpretable failures, not gotchas)
- **Correctness:** must get the numerical contract exactly right in one shot — packed int4
  bit-unpacking, asymmetric zero-points, int8→fp16 requant, fp8 block-scaling, MLA absorb.
  Models produce compile errors or subtly wrong outputs.
- **Performance:** even when correct, must beat `torch.compile` via fusion (dequant+matmul,
  sparse dispatch). Correct-but-slow does not pass (e.g. MoE: correct, < 1× → fail).

## Excluded as too easy (single-shot ≥ ~3/4 — dropped or demoted)
| Family | pass@1 | why too easy |
|---|---|---|
| vanilla causal attention | 2/2 | flash-attention is memorized |
| symmetric W4A16 quant | 3/3 | standard dequant, no contract twist |
| MLA prefill | 2/2 | degenerates to ordinary attention |
| Gated DeltaNet (naive) | 2/2 | weak loop baseline; chunked scan known |

→ Lesson driving the design: **difficulty tracks baseline strength.** We keep families
where `torch.compile`/cuBLAS leave real fusion headroom; we drop memorized ops and weak
(sequential-loop / O(N²)) baselines that frontier models clear in one shot.

(Agentic Harbor results — where models iterate with compiler/profiler feedback — are
reported separately; see the agentic matrix. Single-shot is the headline gap.)
