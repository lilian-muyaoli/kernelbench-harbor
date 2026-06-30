# Scaling pipeline: 10s → 100s → 1K → 10K → 100K tasks

The benchmark is built so that **difficulty lives in the operator template, and volume
comes from parameter grids** over real model configs. A confirmed-hard template scales to
hundreds of tasks with zero new human design; the shared verifier is written once.

## The 3-stage pipeline (already implemented)

1. **Template authoring (human, the only creative step).**
   A template = a naive PyTorch reference (`Model` + `get_inputs`) for one operator family
   where `torch.compile` leaves real headroom (quant GEMM, MoE, FP8, MLA-decode, …).
   Validated once with the cheap probe (`validate/measure_pass1.py`): OpenRouter generates a
   kernel → Modal grades it → we confirm pass@1 ∈ [10%, 60%].

2. **Programmatic expansion (`generate_tasks.py`, no GPU).**
   `template × parameter grid` → self-contained Harbor task dirs. The grid is **mined from
   real HuggingFace configs** (layer dims, group sizes, #experts, dtypes), so every task is a
   real deployment shape, not a contrived one. One template → 15–30 tasks in seconds.

3. **Evaluation on Modal (Harbor-native).**
   Each task's `tests/test.sh` runs the shared `kb_verifier.py` (correctness vs fp32 GT +
   speedup vs `torch.compile` + anti-cheat + roofline). Harbor's `-e modal` runtime gives GPU
   sandboxes; difficulty at scale is estimated by **sampling** (a few tasks per template) — we
   do not run every model on every task.

## Volume vs effort

| Scale | How | Human effort | Wall-clock (gen) | GPU eval cost* |
|---|---|---|---|---|
| **10s** | hand a few templates | hours | minutes | ~$5 |
| **100s** | expand grids over HF configs | ~1 day | minutes (gen) | ~$30–60 |
| **1K** | +more templates (~15 families) + denser grids + more models | ~1 week, 1 eng | ~1 hr gen, hrs eval | ~$300–600 |
| **10K** | mine HF configs at scale + auto-difficulty filtering | ~3–4 weeks, 2 eng + 1 ops | gen trivial; eval parallel on Modal | ~$3K–6K |
| **100K** | config-mining crawler + LLM-assisted template synthesis + RL-style auto-curation | ~2–3 months, 3–4 eng + experts | continuous | ~$30K–60K, dominated by GPU |

*OpenRouter (generation) + Modal (GPU eval); rough, L40S-class.

## Staffing

- **10s–100s:** 1 engineer (the SPA/SPL).
- **1K:** 1 engineer full-time + occasional GPU-kernel expert review for new templates.
- **10K:** 2 engineers (pipeline + infra) + 1 ops (config mining, QC triage) + 1 part-time
  GPU expert to vet new operator families.
- **100K:** 3–4 engineers + 1–2 GPU/ML-systems experts + ops. The bottleneck shifts from
  authoring to **GPU throughput and QC**, both parallelizable on Modal.

## How fast can we produce each scale

Generation is near-instant (templates are code). The real clock is **(a) template validation**
(one probe run per new template, minutes) and **(b) GPU eval** (parallel on Modal, ~1–3 min
per task per model). With Modal's concurrency: 100s in an afternoon, 1K in a day, 10K in a
few days of mostly-unattended eval.

## What scales without re-doing work
- **Verifier:** written once, shared by every task (`common/kb_verifier.py`).
- **Difficulty:** inherited by every grid point of a validated template; spot-checked by
  sampling, not exhaustively re-measured.
- **Realism:** guaranteed by sourcing grid params from real model configs.

## Bottlenecks & mitigations
- *Too-easy drift* (weak-baseline ops): gate templates on strong `torch.compile` baselines;
  tune the speedup threshold per family.
- *GPU cost at 10K+:* pre-compile/cache kernels on CPU; batch eval; reserve GPUs.
- *Verifier reward-hacking at scale:* the "beat torch.compile" gate + AST anti-cheat are
  template-agnostic, so they protect new families for free.
