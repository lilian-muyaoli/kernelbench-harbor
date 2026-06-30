"""
generate_tasks.py — the scaling production line.

Expands a CONFIRMED hard operator template across a parameter grid of REAL
HuggingFace layer dimensions into self-contained Harbor task directories.

Run (no GPU needed, just writes files):
    python generate_tasks.py

Each task dir:
    tasks/<id>/
        task.toml                 # Harbor metadata
        instruction.md            # what the agent must do
        environment/Dockerfile    # torch+triton image; copies reference + verifier
        environment/reference.py  # naive PyTorch Model + get_inputs (the "question")
        environment/kb_verifier.py
        environment/config.json   # tolerances + speedup threshold (difficulty knob)
        tests/test.sh             # runs verifier -> /logs/verifier/reward.json
        solution/solve.sh         # oracle placeholder (sanity)
"""
import os
import json
import shutil
import base64

ROOT = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(ROOT, "tasks")
VERIFIER = os.path.join(ROOT, "common", "kb_verifier.py")
ORACLES_DIR = os.path.join(ROOT, "common", "oracles")
# family prefix -> known-good Triton solution (validated PASS); used by `harbor run -a oracle`
ORACLES = {
    "quant_w4a16_pa": "quant_w4a16_pa.py",
    "fp8_microscale": "fp8_microscale.py",
    "moe": "moe.py",
    "ssm_scan": "ssm_scan.py",
    "mla_decode": "mla_decode.py",
    "w8a8_smoothquant": "w8a8_smoothquant.py",   # promoted from a reward=1 agentic run (5.5x)
    "multi_lora_qgemm": "multi_lora_qgemm.py",    # promoted from a reward=1 single-shot run (1.48x)
    "fused_silu_fp8": "fused_silu_fp8.py",        # adapted from vLLM deep_gemm kernel (1.4x, cosine)
    "fused_qmoe": "fused_qmoe.py",                 # fused int8-dequant tiled MoE GEMM (1.44x)
    "nsa_block_sparse": "nsa_block_sparse.py",     # flash sparse attn + efficient block scoring (1.4x @ n2048, cosine)
}

# ---- CONFIRMED hard template: packed-asymmetric W4A16 group-quant GEMM ----
QUANT_REF = '''import torch
import torch.nn as nn

GROUP = {group}

class Model(nn.Module):
    """Packed asymmetric W4A16 (GPTQ-style) linear, decode shape.
    x:[M,K] {dt}; qweight:[K,N//8] int32 (8 uint4 nibbles each; nibble i -> col 8*j+i);
    scales,zeros:[K//GROUP,N] {dt}. w[k,n]=(unpack(qweight)[k,n]-zeros[k//GROUP,n])*scales[k//GROUP,n]."""
    def forward(self, x, qweight, scales, zeros):
        K = qweight.shape[0]; N = qweight.shape[1] * 8
        shifts = torch.arange(0, 32, 4, device=x.device, dtype=torch.int32)
        w = ((qweight.unsqueeze(-1) >> shifts) & 0xF).reshape(K, N).to(x.dtype)
        g = K // GROUP
        se = scales.reshape(g, 1, N).expand(g, GROUP, N).reshape(K, N).to(x.dtype)
        ze = zeros.reshape(g, 1, N).expand(g, GROUP, N).reshape(K, N).to(x.dtype)
        return torch.matmul(x, (w - ze) * se)

def get_inputs():
    M, K, N = {M}, {K}, {N}
    nib = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int64)
    packed = torch.zeros(K, N // 8, device="cuda", dtype=torch.int64)
    for i in range(8):
        packed |= (nib[:, i::8] << (4 * i))
    qweight = packed.to(torch.int32)
    x = torch.randn(M, K, device="cuda", dtype=torch.{dt})
    scales = (torch.randn(K // GROUP, N, device="cuda") * 0.01).to(torch.{dt})
    zeros = torch.randint(0, 16, (K // GROUP, N), device="cuda").to(torch.{dt})
    return [x, qweight, scales, zeros]
'''

INSTRUCTION = '''# Optimize a GPU kernel

You are given a naive PyTorch reference module `Model` (in `reference.py`). Write an
optimized replacement `ModelNew` with the **same `forward` signature and outputs**.

## CRITICAL — how to submit
Your FINAL action MUST write the complete code (imports + Triton kernel + `class ModelNew`)
to **`/app/solution.py`**. Use a single shell command, e.g.:
```
cat > /app/solution.py << 'EOF'
<your full python code here>
EOF
```
The grader reads ONLY `/app/solution.py`. If you do not write this file, you score 0.

## Rules (your submission is rejected otherwise)
- `ModelNew()` must construct with **no arguments**; same `forward(...)` signature as `Model`.
- `ModelNew.forward` must compute the core work with at least one real `@triton.jit` kernel.
- Do NOT offload core compute to torch (`torch.matmul`, `torch.softmax`, `einsum`, `F.linear`, …)
  or vendor libs (`cublas`, `cudnn`, `flash_attn`, `scaled_dot_product_attention`).
- Must be numerically correct vs the reference (fp32 ground truth, dtype-aware tolerance).
- You may use torch only for tensor allocation, shape handling, and launching the kernel.

## You pass iff
correct **AND** faster than `torch.compile(Model, mode="max-autotune")` by the configured
threshold. Beating the compiler is the goal — a correct-but-slow kernel does not pass.

## The reference
```python
{ref}
```
'''

DOCKERFILE = '''FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
RUN pip install --no-cache-dir numpy
COPY reference.py /app/reference.py
COPY kb_verifier.py /app/kb_verifier.py
COPY config.json /app/config.json
WORKDIR /app
'''

TEST_SH = '''#!/bin/bash
mkdir -p /logs/verifier
python /app/kb_verifier.py /app/reference.py /app/solution.py /app/config.json /logs/verifier/reward.json
# also emit reward.txt fallback (1/0) from reward.json
python -c "import json; print(int(json.load(open('/logs/verifier/reward.json'))['reward']))" > /logs/verifier/reward.txt 2>/dev/null || echo 0 > /logs/verifier/reward.txt
'''

# Oracle: writes a known-good Triton solution to /app/solution.py so `harbor run -a oracle`
# exercises the full chain and confirms a correct solution PASSES the verifier.
SOLVE_SH = '''#!/bin/bash
set -e
if [ -f /app/solution/oracle_solution.py ]; then
  cp /app/solution/oracle_solution.py /app/solution.py
  echo "oracle solution installed"
else
  echo "no oracle provided for this task" >&2
  exit 1
fi
'''

TASK_TOML = '''schema_version = "1.3"

[task]
name = "kernelbench-harbor/{id}"
description = "{desc}"
authors = [{{ name = "AfterQuery trial", email = "muyaosh18@gmail.com" }}]
keywords = {keywords}

[environment]
os = "linux"
cpus = 4
memory_mb = 16384
gpus = 1
gpu_types = ["L40S"]
network_mode = "public"
build_timeout_sec = 1800.0

[agent]
timeout_sec = 1200.0

[verifier]
timeout_sec = 1200.0
'''

# ---- CONFIRMED hard template #2: top-k MoE FFN (sparse dispatch) ----
MOE_REF = '''import torch
import torch.nn as nn

TOPK = {topk}

class Model(nn.Module):
    """Top-{topk} mixture-of-experts FFN. x:[T,D]; w:[E,D,H]; router_logits:[T,E].
    out[t] = sum over its top-{topk} experts e of gate[t,e] * (x[t] @ w[e]).
    A fast kernel must do sparse dispatch instead of computing all E experts."""
    def forward(self, x, w, router_logits):
        probs = torch.softmax(router_logits.float(), dim=-1)
        topv, topi = probs.topk(TOPK, dim=-1)
        topv = (topv / topv.sum(dim=-1, keepdim=True)).to(x.dtype)
        all_out = torch.einsum('td,edh->teh', x, w)
        H = w.shape[2]
        sel = torch.gather(all_out, 1, topi.unsqueeze(-1).expand(-1, -1, H))
        return (sel * topv.unsqueeze(-1)).sum(dim=1)

def get_inputs():
    T, D, E, H = {T}, {D}, {E}, {H}
    x = torch.randn(T, D, device="cuda", dtype=torch.{dt})
    w = (torch.randn(E, D, H, device="cuda", dtype=torch.{dt}) * 0.02)
    router_logits = torch.randn(T, E, device="cuda", dtype=torch.{dt})
    return [x, w, router_logits]
'''

# (name, D, E, H, topk) — real MoE configs (Mixtral / Qwen-MoE / DeepSeek-MoE / Grok)
MOE_LAYERS = [
    ("mixtral_8x7b", 4096, 8, 14336, 2),
    ("grok1", 6144, 8, 8192, 2),
    ("qwen2_57b", 3584, 64, 2560, 8),
    ("deepseek_v2_lite", 2048, 64, 1408, 6),
    ("qwen3_30b", 2048, 128, 768, 8),
]

# ---- hard template #3: W8A8 SmoothQuant GEMM ----
W8A8_REF = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """W8A8 SmoothQuant linear. x_int8:[M,K] int8; w_int8:[K,N] int8;
    x_scale:[M,1] fp16 (per-token); w_scale:[1,N] fp16 (per-channel). out fp16."""
    def forward(self, x_int8, w_int8, x_scale, w_scale):
        acc = torch.matmul(x_int8.float(), w_int8.float())
        return (acc * x_scale.float() * w_scale.float()).to(torch.float16)

def get_inputs():
    M, K, N = {M}, {K}, {N}
    x_int8 = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    w_int8 = torch.randint(-127, 128, (K, N), device="cuda", dtype=torch.int8)
    x_scale = (torch.rand(M, 1, device="cuda") * 0.01 + 0.001).to(torch.float16)
    w_scale = (torch.rand(1, N, device="cuda") * 0.01 + 0.001).to(torch.float16)
    return [x_int8, w_int8, x_scale, w_scale]
'''

# ---- hard template #4: FP8 (e4m3) block-microscaled GEMM ----
FP8_REF = '''import torch
import torch.nn as nn

BLOCK = {block}

class Model(nn.Module):
    """FP8 block-microscaled weight GEMM. x:[M,K] {dt}; w_fp8:[K,N] float8_e4m3fn;
    scales:[K//BLOCK,N] fp16.  w[k,n]=w_fp8[k,n]*scales[k//BLOCK,n] ; out=x@w."""
    def forward(self, x, w_fp8, scales):
        K, N = w_fp8.shape
        w = w_fp8.float().reshape(K // BLOCK, BLOCK, N) * scales[:, None, :].float()
        return torch.matmul(x, w.reshape(K, N).to(x.dtype))

def get_inputs():
    M, K, N = {M}, {K}, {N}
    x = torch.randn(M, K, device="cuda", dtype=torch.{dt})
    w_fp8 = (torch.randn(K, N, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    scales = (torch.rand(K // BLOCK, N, device="cuda") * 0.05 + 0.05).to(torch.float16)
    return [x, w_fp8, scales]
'''

# ---- hard template #5 (supplementary, weak baseline): Mamba selective scan ----
SSM_REF = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """Mamba selective scan. u,delta:[B,L,D]; A:[D,S]; Bm,C:[B,L,S]. y:[B,L,D]."""
    def forward(self, u, delta, A, Bm, C):
        Bb, L, D = u.shape
        S = A.shape[1]
        dA = torch.exp(delta.unsqueeze(-1) * A)
        dBu = (delta.unsqueeze(-1) * Bm.unsqueeze(2)) * u.unsqueeze(-1)
        h = torch.zeros(Bb, D, S, device=u.device, dtype=torch.float32)
        ys = []
        for t in range(L):
            h = dA[:, t].float() * h + dBu[:, t].float()
            ys.append((h * C[:, t].unsqueeze(1).float()).sum(-1))
        return torch.stack(ys, dim=1).to(u.dtype)

def get_inputs():
    B, L, D, S = {B}, {L}, {D}, {S}
    u = torch.randn(B, L, D, device="cuda", dtype=torch.float16)
    delta = torch.nn.functional.softplus(torch.randn(B, L, D, device="cuda", dtype=torch.float16))
    A = (-torch.rand(D, S, device="cuda", dtype=torch.float16) - 0.5)
    Bm = torch.randn(B, L, S, device="cuda", dtype=torch.float16)
    C = torch.randn(B, L, S, device="cuda", dtype=torch.float16)
    return [u, delta, A, Bm, C]
'''

# linear (K,N) shapes for W8A8 / FP8 — real attn+MLP projections
LINEAR_SHAPES = [
    ("llama3_8b_qkv", 4096, 6144),
    ("llama3_8b_o", 4096, 4096),
    ("llama3_8b_gate", 4096, 14336),
    ("llama3_8b_down", 14336, 4096),
    ("qwen2_7b_gate", 3584, 18944),
    ("qwen2_72b_down", 29568, 8192),
]
# (name, B, D, S) for SSM — real Mamba/Mamba2 configs
SSM_CFG = [
    ("mamba_130m", 4, 1536, 16),
    ("mamba_790m", 4, 3072, 16),
    ("mamba2_2k_s64", 4, 2048, 64),
    ("mamba2_2k_s128", 2, 2048, 128),
    ("codestral_mamba", 2, 4096, 16),
]

# ---- hard template #6: MLA decode (absorb), the SOTA non-quant attention family ----
MLA_REF = '''import math
import torch
import torch.nn as nn

H, DH, DC = {H}, {DH}, {DC}

class Model(nn.Module):
    """MLA decode, naive decompress-then-attend. q:[B,H,DH]; c_kv:[B,L,DC] cached latent;
    W_uk:[DC,H*DH]; W_uv:[DC,H*DH]. Fast kernel must use the 'absorb' trick (attend in
    latent space; never materialize per-head K/V)."""
    def forward(self, q, c_kv, W_uk, W_uv):
        B, L, _ = c_kv.shape
        K = (c_kv @ W_uk).view(B, L, H, DH).permute(0, 2, 1, 3)
        V = (c_kv @ W_uv).view(B, L, H, DH).permute(0, 2, 1, 3)
        att = torch.einsum('bhd,bhld->bhl', q, K) / math.sqrt(DH)
        att = torch.softmax(att, dim=-1)
        return torch.einsum('bhl,bhld->bhd', att, V)

def get_inputs():
    B, L = {B}, {L}
    q = torch.randn(B, H, DH, device="cuda", dtype=torch.float16)
    c_kv = torch.randn(B, L, DC, device="cuda", dtype=torch.float16) * 0.02
    W_uk = torch.randn(DC, H * DH, device="cuda", dtype=torch.float16) * 0.02
    W_uv = torch.randn(DC, H * DH, device="cuda", dtype=torch.float16) * 0.02
    return [q, c_kv, W_uk, W_uv]
'''

# (name, H, DH, DC) — real MLA configs (DeepSeek-V2/V3, Kimi)
MLA_CFG = [
    ("deepseek_v2", 16, 128, 512),
    ("deepseek_v3", 32, 128, 512),
    ("kimi_k2", 16, 64, 256),
]

# ---- hard template #7: FUSED composite (SiLU+mul+per-group dynamic FP8 quant), vLLM DeepGEMM MoE act ----
FUSED_SILU_REF = '''import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = {group}
FP8_MAX = 448.0  # e4m3fn max

class Model(nn.Module):
    """Fused SiLU + multiply + per-group dynamic FP8 quant (vLLM batched_deep_gemm_moe).
    y:[T,2H]=concat(gate,up). act=silu(gate)*up; per-group (GROUP along H) dynamic fp8:
    scale=amax(|act|)/FP8_MAX; q=round_fp8(clamp(act/scale)). Returns dequant q*scale.
    A fast kernel fuses silu+mul+reduction+quant in one pass."""
    def forward(self, y):
        T, H2 = y.shape
        H = H2 // 2
        gate, up = y[:, :H], y[:, H:]
        act = (F.silu(gate.float()) * up.float())
        G = H // GROUP
        a = act.reshape(T, G, GROUP)
        scale = (a.abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
        q = (a / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
        return (q * scale).reshape(T, H)

def get_inputs():
    T, H = {T}, {H}
    y = torch.randn(T, 2 * H, device="cuda", dtype=torch.{dt})
    return [y]
'''

# (name, H) — real MoE intermediate sizes (H divisible by GROUP)
FUSED_CFG = [
    ("deepseek_moe", 1408),
    ("qwen_moe", 2560),
    ("mixtral", 14336),
]

# ---- hard template #8: FUSED quantized MoE (vLLM fused_moe_gptq_awq): routing + dequant + grouped GEMM ----
FUSED_QMOE_REF = '''import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 128

class Model(nn.Module):
    """Quantized top-2 MoE. x:[T,D]; qweight:[E,D,N] int8 (symmetric int4 in [-8,7]);
    scales:[E,D//GROUP,N] fp16; router_logits:[T,E]. Per-expert dequant (per-group along D)
    + grouped GEMM + top-2 gated combine, fused in one kernel."""
    def forward(self, x, qweight, scales, router_logits):
        T, D = x.shape
        E, _, N = qweight.shape
        probs = torch.softmax(router_logits.float(), dim=-1)
        topv, topi = probs.topk(2, dim=-1)
        topv = (topv / topv.sum(dim=-1, keepdim=True)).to(x.dtype)
        G = D // GROUP
        w = (qweight.reshape(E, G, GROUP, N).to(x.dtype) * scales[:, :, None, :].to(x.dtype)).reshape(E, D, N)
        all_out = torch.einsum('td,edn->ten', x, w)
        sel = torch.gather(all_out, 1, topi.unsqueeze(-1).expand(-1, -1, N))
        return (sel * topv.unsqueeze(-1)).sum(dim=1)

def get_inputs():
    T, D, E, N = {T}, {D}, {E}, {N}
    x = torch.randn(T, D, device="cuda", dtype=torch.{dt})
    qweight = torch.randint(-8, 8, (E, D, N), device="cuda", dtype=torch.int8)
    scales = (torch.randn(E, D // GROUP, N, device="cuda") * 0.01).to(torch.{dt})
    router_logits = torch.randn(T, E, device="cuda", dtype=torch.{dt})
    return [x, qweight, scales, router_logits]
'''

# (name, D, E, N) — real quantized-MoE configs
FUSED_QMOE_CFG = [
    ("deepseek_moe", 2048, 64, 1408),
    ("mixtral", 4096, 8, 14336),
    ("qwen_moe", 2048, 60, 1408),
]

# ---- hard template #9: multi-LoRA quantized GEMM (vLLM lora_expand): int4 base + per-token gathered LoRA ----
LORA_REF = '''import torch
import torch.nn as nn

GROUP = 128

class Model(nn.Module):
    """Multi-LoRA quantized linear (batched adapters). x:[T,D]; qweight:[D,N] int8 (int4 base);
    scales:[D//GROUP,N] fp16; A:[L,D,r], B:[L,r,N]; lora_ids:[T].
    y[t] = x[t]@dequant(W) + (x[t]@A[lora_ids[t]])@B[lora_ids[t]] (fused, per-token adapter gather)."""
    def forward(self, x, qweight, scales, A, B, lora_ids):
        D, N = qweight.shape
        w = (qweight.reshape(D // GROUP, GROUP, N).to(x.dtype) * scales[:, None, :].to(x.dtype)).reshape(D, N)
        base = x @ w
        Ai, Bi = A[lora_ids], B[lora_ids]
        lo = torch.einsum('td,tdr->tr', x, Ai)
        lo = torch.einsum('tr,trn->tn', lo, Bi)
        return base + lo

def get_inputs():
    T, D, N, r, L = {T}, {D}, {N}, {r}, {L}
    x = torch.randn(T, D, device="cuda", dtype=torch.{dt})
    qweight = torch.randint(-8, 8, (D, N), device="cuda", dtype=torch.int8)
    scales = (torch.randn(D // GROUP, N, device="cuda") * 0.01).to(torch.{dt})
    A = (torch.randn(L, D, r, device="cuda", dtype=torch.{dt}) * 0.02)
    B = (torch.randn(L, r, N, device="cuda", dtype=torch.{dt}) * 0.02)
    lora_ids = torch.randint(0, L, (T,), device="cuda")
    return [x, qweight, scales, A, B, lora_ids]
'''

# (name, D, N, r, L) — real multi-LoRA configs
LORA_CFG = [
    ("llama3_8b_r16", 4096, 4096, 16, 8),
    ("qwen_7b_r32", 3584, 3584, 32, 8),
    ("llama3_8b_r64", 4096, 4096, 64, 4),
]

# ---- hard template #10: Native Sparse Attention (DeepSeek NSA): block top-k selection + sparse flash ----
NSA_REF = '''import math
import torch
import torch.nn as nn

BS = {bs}
TOPB = {topb}

class Model(nn.Module):
    """Block-sparse causal attention (NSA). q,k,v:[B,H,N,D]. Each query scores KV blocks by
    mean key, selects top-TOPB causal blocks, attends only to keys in those blocks."""
    def forward(self, q, k, v):
        B, H, N, D = q.shape
        nb = N // BS
        scale = 1.0 / math.sqrt(D)
        kb = k.view(B, H, nb, BS, D).mean(3)
        bscore = torch.einsum('bhnd,bhcd->bhnc', q, kb) * scale
        qpos = torch.arange(N, device=q.device) // BS
        cblk = qpos[:, None] >= torch.arange(nb, device=q.device)[None, :]
        bscore = bscore.masked_fill(~cblk, float('-inf'))
        topb = bscore.topk(min(TOPB, nb), dim=-1).indices
        sel = torch.zeros(B, H, N, nb, dtype=torch.bool, device=q.device)
        sel.scatter_(-1, topb, True)
        keyblk = (torch.arange(N, device=q.device) // BS)
        keep = sel.gather(-1, keyblk[None, None, None, :].expand(B, H, N, N))
        att = torch.einsum('bhnd,bhmd->bhnm', q, k) * scale
        causal = torch.arange(N, device=q.device)[:, None] >= torch.arange(N, device=q.device)[None, :]
        att = att.masked_fill(~(causal & keep), float('-inf'))
        att = torch.softmax(att, dim=-1)
        return torch.matmul(att, v)

def get_inputs():
    B, H, N, D = {B}, {H}, {N}, {D}
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    return [q, k, v]
'''

# (name, B, H, N, D, bs, topb) — NSA configs
NSA_CFG = [
    ("n1024_h8", 2, 8, 1024, 64, 64, 4),
    ("n2048_h16", 2, 16, 2048, 64, 64, 4),
    ("n4096_h8", 2, 8, 4096, 64, 64, 4),     # long context, D=64: sparsity 4/64 -> perf gate fair
]

# (name, K, N) — real HuggingFace layer dims; N divisible by 8, K divisible by 64/128
LAYERS = [
    ("llama3_8b_gate", 4096, 14336),
    ("llama3_8b_down", 14336, 4096),
    ("llama3_70b_gate", 8192, 28672),
    ("llama3_70b_down", 28672, 8192),
    ("qwen2_7b_gate", 3584, 18944),
    ("qwen2_7b_down", 18944, 3584),
    ("qwen2_72b_gate", 8192, 29568),
    ("qwen2_72b_down", 29568, 8192),
]
DT_SHORT = {"float16": "fp16", "bfloat16": "bf16"}


def family_dir(tid):
    """Group tasks into per-family folders (moe_top2/6/8 collapse to 'moe')."""
    fam = tid.split("__")[0]
    return "moe" if fam.startswith("moe_top") else fam


def emit(tid, ref, desc, keywords):
    d = os.path.join(TASKS_DIR, family_dir(tid), tid)
    os.makedirs(os.path.join(d, "environment"), exist_ok=True)
    os.makedirs(os.path.join(d, "tests"), exist_ok=True)
    os.makedirs(os.path.join(d, "solution"), exist_ok=True)
    open(os.path.join(d, "environment", "reference.py"), "w").write(ref)
    shutil.copy(VERIFIER, os.path.join(d, "environment", "kb_verifier.py"))
    cfg = {"atol": 2e-2, "rtol": 2e-2, "speedup_threshold": 1.0,
           "num_correctness_trials": 5, "warmup": 25, "perf_trials": 100}
    # Families whose output is itself quantized (fp8/int) or top-k-selected are graded with a
    # cosine/relative metric -- the standard way such kernels are validated, since the output is a
    # discrete value. Non-discrete-output families use allclose.
    fam0 = tid.split("__")[0]
    if fam0 in ("fused_silu_fp8", "nsa_block_sparse"):
        cfg.update({"metric": "cosine", "cosine_min": 0.99, "rel_l1_max": 0.05})
    json.dump(cfg, open(os.path.join(d, "environment", "config.json"), "w"), indent=2)
    open(os.path.join(d, "instruction.md"), "w").write(INSTRUCTION.format(ref=ref))
    open(os.path.join(d, "environment", "Dockerfile"), "w").write(DOCKERFILE)
    open(os.path.join(d, "tests", "test.sh"), "w").write(TEST_SH)
    # Self-contained oracle: embed the known-good Triton solution (base64) directly in
    # solve.sh so it writes /app/solution.py without depending on the solution/ dir being
    # mounted into the container. Used by `harbor run -a oracle` for sanity-checking.
    fam = tid.split("__")[0]
    oracle_name = next((v for k, v in ORACLES.items() if fam.startswith(k)), None)
    oracle_path = os.path.join(ORACLES_DIR, oracle_name) if oracle_name else None
    if oracle_path and os.path.exists(oracle_path):
        b64 = base64.b64encode(open(oracle_path, "rb").read()).decode()
        solve = ("#!/bin/bash\nset -e\n"
                 "base64 -d > /app/solution.py <<'B64EOF'\n" + b64 + "\nB64EOF\n"
                 'echo "oracle solution installed at /app/solution.py"\n')
        shutil.copy(oracle_path, os.path.join(d, "solution", "oracle_solution.py"))  # human-readable copy
    else:
        solve = SOLVE_SH
    open(os.path.join(d, "solution", "solve.sh"), "w").write(solve)
    open(os.path.join(d, "task.toml"), "w").write(
        TASK_TOML.format(id=tid, desc=desc, keywords=json.dumps(keywords)))
    return tid


def gen_quant():
    ids = []
    combos = [(n, K, N, g, "float16") for (n, K, N) in LAYERS for g in (64, 128)]
    combos += [(n, K, N, 128, "bfloat16") for (n, K, N) in LAYERS]  # bf16 @ g128
    for (name, K, N, group, dt) in combos:
        tid = f"quant_w4a16_pa__{name}__g{group}__{DT_SHORT[dt]}"
        ref = QUANT_REF.format(group=group, M=16, K=K, N=N, dt=dt)
        desc = f"Packed-asym W4A16 group-quant GEMM, {name} (K={K},N={N}), group={group}, {DT_SHORT[dt]}, decode M=16."
        ids.append(emit(tid, ref, desc, ["gpu", "triton", "quantization", "performance"]))
    return ids


def gen_moe():
    ids = []
    combos = [(n, D, E, H, tk, T, "float16") for (n, D, E, H, tk) in MOE_LAYERS for T in (16, 128, 512)]
    combos += [(n, D, E, H, tk, 512, "bfloat16") for (n, D, E, H, tk) in MOE_LAYERS]
    for (name, D, E, H, tk, T, dt) in combos:
        tid = f"moe_top{tk}__{name}__T{T}__{DT_SHORT[dt]}"
        ref = MOE_REF.format(topk=tk, T=T, D=D, E=E, H=H, dt=dt)
        desc = f"Top-{tk} MoE FFN, {name} (D={D},E={E},H={H}), T={T}, {DT_SHORT[dt]}."
        ids.append(emit(tid, ref, desc, ["gpu", "triton", "moe", "performance"]))
    return ids


def gen_w8a8():
    ids = []
    for (name, K, N) in LINEAR_SHAPES:
        for M in (16, 64, 256):
            tid = f"w8a8_smoothquant__{name}__M{M}"
            ref = W8A8_REF.format(M=M, K=K, N=N)
            desc = f"W8A8 SmoothQuant GEMM, {name} (K={K},N={N}), M={M}."
            ids.append(emit(tid, ref, desc, ["gpu", "triton", "quantization", "int8", "performance"]))
    return ids


def gen_fp8():
    ids = []
    combos = [(n, K, N, b, "float16") for (n, K, N) in LINEAR_SHAPES for b in (32, 128)]
    combos += [(n, K, N, 32, "bfloat16") for (n, K, N) in LINEAR_SHAPES]
    for (name, K, N, block, dt) in combos:
        tid = f"fp8_microscale__{name}__b{block}__{DT_SHORT[dt]}"
        ref = FP8_REF.format(block=block, M=16, K=K, N=N, dt=dt)
        desc = f"FP8 e4m3 block-microscaled GEMM, {name} (K={K},N={N}), block={block}, {DT_SHORT[dt]}."
        ids.append(emit(tid, ref, desc, ["gpu", "triton", "fp8", "microscaling", "performance"]))
    return ids


def gen_ssm():
    ids = []
    for (name, B, D, S) in SSM_CFG:
        for L in (256, 512):
            tid = f"ssm_scan__{name}__L{L}"
            ref = SSM_REF.format(B=B, L=L, D=D, S=S)
            desc = f"Mamba selective scan, {name} (D={D},S={S}), L={L}, B={B}."
            ids.append(emit(tid, ref, desc, ["gpu", "triton", "ssm", "mamba", "performance"]))
    return ids


def gen_mla():
    ids = []
    for (name, H, DH, DC) in MLA_CFG:
        for L in (2048, 4096, 8192):
            for B in (8, 16):
                tid = f"mla_decode__{name}__L{L}__B{B}"
                ref = MLA_REF.format(H=H, DH=DH, DC=DC, B=B, L=L)
                desc = f"MLA decode (absorb), {name} (H={H},DH={DH},DC={DC}), KV-len L={L}, B={B}."
                ids.append(emit(tid, ref, desc, ["gpu", "triton", "attention", "mla", "performance"]))
    return ids


def gen_fused():
    ids = []
    combos = [(n, H, T, "float16") for (n, H) in FUSED_CFG for T in (256, 512)]
    combos += [(n, H, 512, "bfloat16") for (n, H) in FUSED_CFG]
    for (name, H, T, dt) in combos:
        tid = f"fused_silu_fp8__{name}__T{T}__{DT_SHORT[dt]}"
        ref = FUSED_SILU_REF.format(group=128, T=T, H=H, dt=dt)
        desc = f"Fused SiLU+mul+per-group FP8 quant, {name} (H={H}), T={T}, {DT_SHORT[dt]}."
        ids.append(emit(tid, ref, desc, ["gpu", "triton", "fused", "fp8", "moe", "performance"]))
    return ids


def gen_qmoe():
    ids = []
    combos = [(n, D, E, N, "float16") for (n, D, E, N) in FUSED_QMOE_CFG]
    combos += [(n, D, E, N, "bfloat16") for (n, D, E, N) in FUSED_QMOE_CFG]
    for (name, D, E, N, dt) in combos:
        tid = f"fused_qmoe__{name}__{DT_SHORT[dt]}"
        ref = FUSED_QMOE_REF.format(T=256, D=D, E=E, N=N, dt=dt)
        desc = f"Fused quantized top-2 MoE, {name} (D={D},E={E},N={N}), {DT_SHORT[dt]}."
        ids.append(emit(tid, ref, desc, ["gpu", "triton", "fused", "moe", "quantization", "performance"]))
    return ids


def gen_lora():
    ids = []
    combos = [(n, D, N, r, L, "float16") for (n, D, N, r, L) in LORA_CFG]
    combos += [(n, D, N, r, L, "bfloat16") for (n, D, N, r, L) in LORA_CFG]
    for (name, D, N, r, L, dt) in combos:
        tid = f"multi_lora_qgemm__{name}__{DT_SHORT[dt]}"
        ref = LORA_REF.format(T=64, D=D, N=N, r=r, L=L, dt=dt)
        desc = f"Multi-LoRA quantized GEMM, {name} (D={D},N={N},r={r},L={L}), {DT_SHORT[dt]}."
        ids.append(emit(tid, ref, desc, ["gpu", "triton", "fused", "lora", "quantization", "performance"]))
    return ids


def gen_nsa():
    ids = []
    for (name, B, H, N, D, bs, topb) in NSA_CFG:
        tid = f"nsa_block_sparse__{name}"
        ref = NSA_REF.format(bs=bs, topb=topb, B=B, H=H, N=N, D=D)
        desc = f"Native Sparse Attention (block top-k), {name} (B={B},H={H},N={N},D={D},BS={bs},TOPB={topb})."
        ids.append(emit(tid, ref, desc, ["gpu", "triton", "attention", "sparse", "nsa", "performance"]))
    return ids


def main():
    ids = (gen_quant() + gen_moe() + gen_w8a8() + gen_fp8() + gen_ssm()
           + gen_mla() + gen_fused() + gen_qmoe() + gen_lora() + gen_nsa())
    print(f"generated {len(ids)} tasks into {TASKS_DIR}")
    from collections import Counter
    fams = Counter(i.split("__")[0] for i in ids)
    for fam, n in fams.items():
        print(f"  {fam}: {n}")


if __name__ == "__main__":
    main()
