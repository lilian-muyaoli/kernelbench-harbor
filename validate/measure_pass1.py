"""
Measure pass@1 of a frontier model on ONE task, using our verifier
(correct vs fp32 GT  AND  beats torch.compile  AND  anti-reward-hack).

This is the difficulty-calibration tool: it tells us whether a task lands in the
10-60% in-distribution band for frontier models, with DATA instead of guesswork.

Local step : call the model via OpenRouter -> Triton ModelNew.
Remote step: compile+run on Modal L40S, grade with our verifier.

Run:
  cd /Users/lilianli/Downloads/bench/KernelBench
  uv run modal run /Users/lilianli/Downloads/bench/kernelbench-harbor/validate/measure_pass1.py \
      --model openrouter/openai/gpt-5.5 --task attention_causal --samples 2
"""
import os
import re
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "triton==3.1.0", "numpy")
)
app = modal.App("measure-pass1", image=image)

# ----------------------------- TASK LIBRARY -----------------------------
# Each task = a naive PyTorch reference module `Model` + get_inputs().
# The baseline is torch.compile(Model, max-autotune). The agent must produce
# `ModelNew` with a real Triton kernel that is correct AND faster than baseline.
TASKS = {
    "attention_causal": {
        "desc": "Causal self-attention (fp16). Reference is naive matmul+softmax+matmul.",
        "ref": '''
import math
import torch
import torch.nn as nn

class Model(nn.Module):
    """Naive causal scaled-dot-product attention. q,k,v: [B, H, N, D]."""
    def forward(self, q, k, v):
        D = q.shape[-1]
        scale = 1.0 / math.sqrt(D)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        N = q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        return torch.matmul(att, v)

def get_inputs():
    B, H, N, D = 8, 16, 2048, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    return [q, k, v]
''',
    },
    "quant_gemm_w4a16": {
        "desc": "W4A16 group-quantized matmul (decode shape M=16). Fuse dequant+matmul.",
        "ref": '''
import torch
import torch.nn as nn

GROUP = 128

class Model(nn.Module):
    """W4A16 symmetric group-quantized linear.
    x: [M, K] fp16 ; qweight: [K, N] int8 with values in [-8, 7] ;
    scales: [K//GROUP, N] fp16. Output = x @ dequant(qweight)."""
    def forward(self, x, qweight, scales):
        K, N = qweight.shape
        w = qweight.reshape(K // GROUP, GROUP, N).to(x.dtype) * scales[:, None, :].to(x.dtype)
        w = w.reshape(K, N)
        return torch.matmul(x, w)

def get_inputs():
    M, K, N = 16, 4096, 4096
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    qweight = torch.randint(-8, 8, (K, N), device="cuda", dtype=torch.int8)
    scales = (torch.randn(K // GROUP, N, device="cuda") * 0.01).to(torch.float16)
    return [x, qweight, scales]
''',
    },
    "quant_gemm_w4a16_packed_asym": {
        "desc": "Packed int4 + asymmetric (GPTQ-style) W4A16. Unpack nibbles + (q-zero)*scale + matmul, fused.",
        "ref": '''
import torch
import torch.nn as nn

GROUP = 128

class Model(nn.Module):
    """Packed asymmetric W4A16 (GPTQ-style).
    x:[M,K] fp16 ;
    qweight:[K, N//8] int32 (8 unsigned int4 nibbles per int32; nibble i = column 8*j+i) ;
    scales,zeros:[K//GROUP, N] fp16.
    dequant: w[k,n] = ( unpack(qweight)[k,n] - zeros[k//GROUP,n] ) * scales[k//GROUP,n].
    output = x @ w."""
    def forward(self, x, qweight, scales, zeros):
        K = qweight.shape[0]
        N = qweight.shape[1] * 8
        shifts = torch.arange(0, 32, 4, device=x.device, dtype=torch.int32)
        w = (qweight.unsqueeze(-1) >> shifts) & 0xF          # [K, N//8, 8]
        w = w.reshape(K, N).to(x.dtype)
        g = K // GROUP
        scales_e = scales.reshape(g, 1, N).expand(g, GROUP, N).reshape(K, N).to(x.dtype)
        zeros_e = zeros.reshape(g, 1, N).expand(g, GROUP, N).reshape(K, N).to(x.dtype)
        w = (w - zeros_e) * scales_e
        return torch.matmul(x, w)

def get_inputs():
    M, K, N = 16, 4096, 4096
    nib = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int64)
    packed = torch.zeros(K, N // 8, device="cuda", dtype=torch.int64)
    for i in range(8):
        packed |= (nib[:, i::8] << (4 * i))
    qweight = packed.to(torch.int32)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    scales = (torch.randn(K // GROUP, N, device="cuda") * 0.01).to(torch.float16)
    zeros = torch.randint(0, 16, (K // GROUP, N), device="cuda").to(torch.float16)
    return [x, qweight, scales, zeros]
''',
    },
    # --- scaled variants of the CONFIRMED hard template (packed-asym W4A16),
    #     real HuggingFace layer dims (decode shape M=16). Same difficulty mechanism. ---
    "quant_w4a16_packed_asym_llama3_8b_gate": {
        "desc": "packed-asym W4A16, Llama-3-8B gate/up proj (K=4096,N=14336,G=128).",
        "ref": '''
import torch
import torch.nn as nn
GROUP = 128
class Model(nn.Module):
    def forward(self, x, qweight, scales, zeros):
        K = qweight.shape[0]; N = qweight.shape[1] * 8
        shifts = torch.arange(0, 32, 4, device=x.device, dtype=torch.int32)
        w = ((qweight.unsqueeze(-1) >> shifts) & 0xF).reshape(K, N).to(x.dtype)
        g = K // GROUP
        se = scales.reshape(g, 1, N).expand(g, GROUP, N).reshape(K, N).to(x.dtype)
        ze = zeros.reshape(g, 1, N).expand(g, GROUP, N).reshape(K, N).to(x.dtype)
        return torch.matmul(x, (w - ze) * se)
def get_inputs():
    M, K, N = 16, 4096, 14336
    nib = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int64)
    packed = torch.zeros(K, N // 8, device="cuda", dtype=torch.int64)
    for i in range(8):
        packed |= (nib[:, i::8] << (4 * i))
    qweight = packed.to(torch.int32)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    scales = (torch.randn(K // GROUP, N, device="cuda") * 0.01).to(torch.float16)
    zeros = torch.randint(0, 16, (K // GROUP, N), device="cuda").to(torch.float16)
    return [x, qweight, scales, zeros]
''',
    },
    "quant_w4a16_packed_asym_llama3_8b_down": {
        "desc": "packed-asym W4A16, Llama-3-8B down proj (K=14336,N=4096,G=128).",
        "ref": '''
import torch
import torch.nn as nn
GROUP = 128
class Model(nn.Module):
    def forward(self, x, qweight, scales, zeros):
        K = qweight.shape[0]; N = qweight.shape[1] * 8
        shifts = torch.arange(0, 32, 4, device=x.device, dtype=torch.int32)
        w = ((qweight.unsqueeze(-1) >> shifts) & 0xF).reshape(K, N).to(x.dtype)
        g = K // GROUP
        se = scales.reshape(g, 1, N).expand(g, GROUP, N).reshape(K, N).to(x.dtype)
        ze = zeros.reshape(g, 1, N).expand(g, GROUP, N).reshape(K, N).to(x.dtype)
        return torch.matmul(x, (w - ze) * se)
def get_inputs():
    M, K, N = 16, 14336, 4096
    nib = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int64)
    packed = torch.zeros(K, N // 8, device="cuda", dtype=torch.int64)
    for i in range(8):
        packed |= (nib[:, i::8] << (4 * i))
    qweight = packed.to(torch.int32)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    scales = (torch.randn(K // GROUP, N, device="cuda") * 0.01).to(torch.float16)
    zeros = torch.randint(0, 16, (K // GROUP, N), device="cuda").to(torch.float16)
    return [x, qweight, scales, zeros]
''',
    },
    "moe_topk": {
        "desc": "Top-2 MoE FFN: route each token to 2 of E experts, gated sum. Fast kernel must do sparse dispatch.",
        "ref": '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """Top-2 mixture-of-experts. x:[T,D]; w:[E,D,H]; router_logits:[T,E].
    out[t] = sum over its top-2 experts e of gate[t,e] * (x[t] @ w[e])."""
    def forward(self, x, w, router_logits):
        probs = torch.softmax(router_logits.float(), dim=-1)
        topv, topi = probs.topk(2, dim=-1)
        topv = (topv / topv.sum(dim=-1, keepdim=True)).to(x.dtype)
        all_out = torch.einsum('td,edh->teh', x, w)          # dense: all experts
        H = w.shape[2]
        sel = torch.gather(all_out, 1, topi.unsqueeze(-1).expand(-1, -1, H))
        return (sel * topv.unsqueeze(-1)).sum(dim=1)

def get_inputs():
    T, D, E, H = 512, 2048, 8, 2048
    x = torch.randn(T, D, device="cuda", dtype=torch.float16)
    w = (torch.randn(E, D, H, device="cuda", dtype=torch.float16) * 0.02)
    router_logits = torch.randn(T, E, device="cuda", dtype=torch.float16)
    return [x, w, router_logits]
''',
    },
    "ssm_scan": {
        "desc": "Mamba-style selective scan (sequential recurrence). Fast kernel needs a parallel scan.",
        "ref": '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """Selective state-space scan (Mamba S6 core).
    u,delta:[B,L,D]; A:[D,S]; Bm,C:[B,L,S].  h_t = exp(delta*A)*h_{t-1} + (delta*Bm)*u ;
    y_t = sum_s h_t * C_t.  Output y:[B,L,D]."""
    def forward(self, u, delta, A, Bm, C):
        Bb, L, D = u.shape
        S = A.shape[1]
        dA = torch.exp(delta.unsqueeze(-1) * A)                                   # [B,L,D,S]
        dBu = (delta.unsqueeze(-1) * Bm.unsqueeze(2)) * u.unsqueeze(-1)           # [B,L,D,S]
        h = torch.zeros(Bb, D, S, device=u.device, dtype=torch.float32)
        ys = []
        for t in range(L):
            h = dA[:, t].float() * h + dBu[:, t].float()
            ys.append((h * C[:, t].unsqueeze(1).float()).sum(-1))                 # [B,D]
        return torch.stack(ys, dim=1).to(u.dtype)                                 # [B,L,D]

def get_inputs():
    B, L, D, S = 4, 256, 1024, 16
    u = torch.randn(B, L, D, device="cuda", dtype=torch.float16)
    delta = torch.nn.functional.softplus(torch.randn(B, L, D, device="cuda", dtype=torch.float16))
    A = (-torch.rand(D, S, device="cuda", dtype=torch.float16) - 0.5)
    Bm = torch.randn(B, L, S, device="cuda", dtype=torch.float16)
    C = torch.randn(B, L, S, device="cuda", dtype=torch.float16)
    return [u, delta, A, Bm, C]
''',
    },
    "w8a8_smoothquant": {
        "desc": "W8A8 SmoothQuant GEMM: int8 act x int8 weight, per-token & per-channel scales, fused requant.",
        "ref": '''
import torch
import torch.nn as nn
class Model(nn.Module):
    """x_int8:[M,K] int8; w_int8:[K,N] int8; x_scale:[M,1] fp16 (per-token);
    w_scale:[1,N] fp16 (per-channel). out = (x_int8 @ w_int8) * x_scale * w_scale, fp16."""
    def forward(self, x_int8, w_int8, x_scale, w_scale):
        acc = torch.matmul(x_int8.float(), w_int8.float())
        return (acc * x_scale.float() * w_scale.float()).to(torch.float16)
def get_inputs():
    M, K, N = 16, 4096, 4096
    x_int8 = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    w_int8 = torch.randint(-127, 128, (K, N), device="cuda", dtype=torch.int8)
    x_scale = (torch.rand(M, 1, device="cuda") * 0.01 + 0.001).to(torch.float16)
    w_scale = (torch.rand(1, N, device="cuda") * 0.01 + 0.001).to(torch.float16)
    return [x_int8, w_int8, x_scale, w_scale]
''',
    },
    "fp8_microscaling_gemm": {
        "desc": "FP8 (e4m3) block-microscaled weight GEMM: dequant fp8 with per-32-block scales + matmul, fused.",
        "ref": '''
import torch
import torch.nn as nn
BLOCK = 32
class Model(nn.Module):
    """x:[M,K] fp16; w_fp8:[K,N] float8_e4m3fn; scales:[K//BLOCK,N] fp16.
    w[k,n] = w_fp8[k,n] * scales[k//BLOCK,n] ; out = x @ w."""
    def forward(self, x, w_fp8, scales):
        K, N = w_fp8.shape
        w = w_fp8.float().reshape(K // BLOCK, BLOCK, N) * scales[:, None, :].float()
        return torch.matmul(x, w.reshape(K, N).to(x.dtype))
def get_inputs():
    M, K, N = 16, 4096, 4096
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_fp8 = (torch.randn(K, N, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    scales = (torch.rand(K // BLOCK, N, device="cuda") * 0.05 + 0.05).to(torch.float16)
    return [x, w_fp8, scales]
''',
    },
    "mla_decode": {
        "desc": "MLA DECODE (DeepSeek/Kimi): 1 query attends to a long cached KV latent. Naive decompresses KV; the fast kernel must use the 'absorb' trick (attend in latent space, never materialize K/V).",
        "ref": '''
import math
import torch
import torch.nn as nn

H, DH, DC = 16, 128, 512  # heads, head_dim, KV latent dim

class Model(nn.Module):
    """MLA decode, naive decompress-then-attend. One query token per batch attends to a
    cached KV latent of length L.
    q:[B,H,DH]; c_kv:[B,L,DC] cached latent; W_uk:[DC,H*DH]; W_uv:[DC,H*DH]."""
    def forward(self, q, c_kv, W_uk, W_uv):
        B, L, _ = c_kv.shape
        K = (c_kv @ W_uk).view(B, L, H, DH).permute(0, 2, 1, 3)   # [B,H,L,DH] decompress (heavy)
        V = (c_kv @ W_uv).view(B, L, H, DH).permute(0, 2, 1, 3)
        att = torch.einsum('bhd,bhld->bhl', q, K) / math.sqrt(DH) # [B,H,L]
        att = torch.softmax(att, dim=-1)
        return torch.einsum('bhl,bhld->bhd', att, V)              # [B,H,DH]

def get_inputs():
    B, L = 8, 4096  # decode: batch 8, KV-cache length 4096
    q = torch.randn(B, H, DH, device="cuda", dtype=torch.float16)
    c_kv = torch.randn(B, L, DC, device="cuda", dtype=torch.float16) * 0.02
    W_uk = torch.randn(DC, H * DH, device="cuda", dtype=torch.float16) * 0.02
    W_uv = torch.randn(DC, H * DH, device="cuda", dtype=torch.float16) * 0.02
    return [q, c_kv, W_uk, W_uv]
''',
    },
    "gated_deltanet": {
        "desc": "Gated DeltaNet (Qwen3-Next), naive sequential delta rule. Fast kernel must write a chunked parallel scan.",
        "ref": '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """Gated DeltaNet (naive sequential). q,k:[B,H,L,Dk]; v:[B,H,L,Dv];
    beta,alpha:[B,H,L]. S_t = alpha_t*S_{t-1} + beta_t*k_t^T(v_t - k_t@S_{t-1}); o_t=q_t@S_t."""
    def forward(self, q, k, v, beta, alpha):
        B, H, L, Dk = q.shape
        Dv = v.shape[-1]
        S = torch.zeros(B, H, Dk, Dv, device=q.device, dtype=torch.float32)
        outs = []
        for t in range(L):
            kt, vt, qt = k[:, :, t].float(), v[:, :, t].float(), q[:, :, t].float()
            a = alpha[:, :, t].float().unsqueeze(-1).unsqueeze(-1)
            b = beta[:, :, t].float().unsqueeze(-1).unsqueeze(-1)
            kS = torch.einsum('bhk,bhkv->bhv', kt, S)
            S = a * S + b * torch.einsum('bhk,bhv->bhkv', kt, vt - kS)
            outs.append(torch.einsum('bhk,bhkv->bhv', qt, S))
        return torch.stack(outs, dim=2).to(q.dtype)

def get_inputs():
    B, H, L, Dk, Dv = 2, 8, 256, 128, 128
    q = torch.randn(B, H, L, Dk, device="cuda", dtype=torch.float16) * 0.5
    k = torch.nn.functional.normalize(torch.randn(B, H, L, Dk, device="cuda", dtype=torch.float16), dim=-1)
    v = torch.randn(B, H, L, Dv, device="cuda", dtype=torch.float16) * 0.5
    beta = torch.rand(B, H, L, device="cuda", dtype=torch.float16)
    alpha = torch.rand(B, H, L, device="cuda", dtype=torch.float16) * 0.1 + 0.9
    return [q, k, v, beta, alpha]
''',
    },
    "nsa_block_sparse_attn": {
        "desc": "Native Sparse Attention (DeepSeek NSA): per-query top-k block selection + sparse causal flash. Different mechanism (block routing in attention).",
        "ref": '''
import math
import torch
import torch.nn as nn

BS = 64   # KV block size
TOPB = 4  # blocks selected per query

class Model(nn.Module):
    """Block-sparse causal attention. q,k,v:[B,H,N,D]. Each query scores KV blocks by its
    mean key, selects top-TOPB causal blocks, and attends only to keys in those blocks."""
    def forward(self, q, k, v):
        B, H, N, D = q.shape
        nb = N // BS
        scale = 1.0 / math.sqrt(D)
        kb = k.view(B, H, nb, BS, D).mean(3)                          # [B,H,nb,D] block reps
        bscore = torch.einsum('bhnd,bhcd->bhnc', q, kb) * scale       # [B,H,N,nb]
        qpos = torch.arange(N, device=q.device) // BS
        cblk = qpos[:, None] >= torch.arange(nb, device=q.device)[None, :]   # causal block mask
        bscore = bscore.masked_fill(~cblk, float('-inf'))
        topb = bscore.topk(min(TOPB, nb), dim=-1).indices            # [B,H,N,TOPB]
        sel = torch.zeros(B, H, N, nb, dtype=torch.bool, device=q.device)
        sel.scatter_(-1, topb, True)
        keyblk = (torch.arange(N, device=q.device) // BS)
        keep = sel.gather(-1, keyblk[None, None, None, :].expand(B, H, N, N))  # [B,H,N,N]
        att = torch.einsum('bhnd,bhmd->bhnm', q, k) * scale
        causal = torch.arange(N, device=q.device)[:, None] >= torch.arange(N, device=q.device)[None, :]
        att = att.masked_fill(~(causal & keep), float('-inf'))
        att = torch.softmax(att, dim=-1)
        return torch.matmul(att, v)

def get_inputs():
    B, H, N, D = 2, 8, 1024, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    return [q, k, v]
''',
    },
    "multi_lora_quant_gemm": {
        "desc": "FUSED multi-LoRA quantized GEMM (vLLM lora_expand, harder): int4 base GEMM + per-token gathered low-rank LoRA, fused.",
        "ref": '''
import torch
import torch.nn as nn

GROUP = 128

class Model(nn.Module):
    """Multi-LoRA quantized linear (batched adapters, vLLM-style). x:[T,D];
    qweight:[D,N] int8 (int4 base [-8,7]); scales:[D//GROUP,N] fp16;
    A:[L,D,r], B:[L,r,N] (L LoRA adapters); lora_ids:[T] int (adapter per token).
    y[t] = x[t] @ dequant(W) + (x[t] @ A[lora_ids[t]]) @ B[lora_ids[t]]."""
    def forward(self, x, qweight, scales, A, B, lora_ids):
        D, N = qweight.shape
        w = (qweight.reshape(D // GROUP, GROUP, N).to(x.dtype) * scales[:, None, :].to(x.dtype)).reshape(D, N)
        base = x @ w
        Ai = A[lora_ids]                              # [T,D,r] per-token adapter gather
        Bi = B[lora_ids]                              # [T,r,N]
        lo = torch.einsum('td,tdr->tr', x, Ai)
        lo = torch.einsum('tr,trn->tn', lo, Bi)
        return base + lo

def get_inputs():
    T, D, N, r, L = 64, 4096, 4096, 16, 8
    x = torch.randn(T, D, device="cuda", dtype=torch.float16)
    qweight = torch.randint(-8, 8, (D, N), device="cuda", dtype=torch.int8)
    scales = (torch.randn(D // GROUP, N, device="cuda") * 0.01).to(torch.float16)
    A = (torch.randn(L, D, r, device="cuda", dtype=torch.float16) * 0.02)
    B = (torch.randn(L, r, N, device="cuda", dtype=torch.float16) * 0.02)
    lora_ids = torch.randint(0, L, (T,), device="cuda")
    return [x, qweight, scales, A, B, lora_ids]
''',
    },
    "fused_quant_mlp": {
        "desc": "FUSED composite (harder than AMD gemm_fusion): 2-layer AWQ MLP — x@W1(int4) -> SiLU -> @W2(int4), all fused, decode shape.",
        "ref": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 128

class Model(nn.Module):
    """Quantized fused 2-layer MLP (AWQ-style). x:[T,D];
    qw1:[D,H] int8 (int4 [-8,7]), s1:[D//GROUP,H] fp16;
    qw2:[H,D] int8, s2:[H//GROUP,D] fp16.  y = silu(x @ deq(qw1)) @ deq(qw2)."""
    def forward(self, x, qw1, s1, qw2, s2):
        D, H = qw1.shape
        w1 = (qw1.reshape(D // GROUP, GROUP, H).to(x.dtype) * s1[:, None, :].to(x.dtype)).reshape(D, H)
        h = F.silu((x @ w1).float()).to(x.dtype)
        w2 = (qw2.reshape(H // GROUP, GROUP, D).to(x.dtype) * s2[:, None, :].to(x.dtype)).reshape(H, D)
        return h @ w2

def get_inputs():
    T, D, H = 16, 4096, 14336
    x = torch.randn(T, D, device="cuda", dtype=torch.float16)
    qw1 = torch.randint(-8, 8, (D, H), device="cuda", dtype=torch.int8)
    s1 = (torch.randn(D // GROUP, H, device="cuda") * 0.01).to(torch.float16)
    qw2 = torch.randint(-8, 8, (H, D), device="cuda", dtype=torch.int8)
    s2 = (torch.randn(H // GROUP, D, device="cuda") * 0.01).to(torch.float16)
    return [x, qw1, s1, qw2, s2]
''',
    },
    "fused_quant_moe": {
        "desc": "FUSED composite (vLLM fused_moe_gptq_awq): top-2 MoE + per-group int4 dequant + grouped GEMM, one kernel.",
        "ref": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 128

class Model(nn.Module):
    """Quantized top-2 MoE. x:[T,D]; qweight:[E,D,N] int8 (symmetric int4 vals in [-8,7]);
    scales:[E, D//GROUP, N] fp16; router_logits:[T,E].
    Per-expert dequant w=q*scale(per-group along D), grouped GEMM, top-2 gated combine."""
    def forward(self, x, qweight, scales, router_logits):
        T, D = x.shape
        E, _, N = qweight.shape
        probs = torch.softmax(router_logits.float(), dim=-1)
        topv, topi = probs.topk(2, dim=-1)
        topv = (topv / topv.sum(dim=-1, keepdim=True)).to(x.dtype)
        G = D // GROUP
        w = qweight.reshape(E, G, GROUP, N).to(x.dtype) * scales[:, :, None, :].to(x.dtype)
        w = w.reshape(E, D, N)
        all_out = torch.einsum('td,edn->ten', x, w)
        sel = torch.gather(all_out, 1, topi.unsqueeze(-1).expand(-1, -1, N))
        return (sel * topv.unsqueeze(-1)).sum(dim=1)

def get_inputs():
    T, D, E, N = 256, 2048, 8, 1408
    x = torch.randn(T, D, device="cuda", dtype=torch.float16)
    qweight = torch.randint(-8, 8, (E, D, N), device="cuda", dtype=torch.int8)
    scales = (torch.randn(E, D // GROUP, N, device="cuda") * 0.01).to(torch.float16)
    router_logits = torch.randn(T, E, device="cuda", dtype=torch.float16)
    return [x, qweight, scales, router_logits]
''',
    },
    "fused_silu_mul_fp8": {
        "desc": "FUSED composite (vLLM DeepGEMM MoE act): SiLU(gate)*up + per-group dynamic FP8 quant, one kernel.",
        "ref": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 128
FP8_MAX = 448.0  # e4m3fn max

class Model(nn.Module):
    """Fused SiLU + multiply + per-group dynamic FP8 quantization (vLLM batched_deep_gemm_moe).
    y:[T, 2H] = concat(gate, up). act = silu(gate)*up ; then per-group (size GROUP along H)
    dynamic fp8: scale_g = amax(|act_g|)/FP8_MAX ; q = round_fp8(clamp(act/scale)).
    Returns dequantized q.float()*scale (comparable). A fast kernel fuses all stages."""
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
    T, H = 512, 4096
    y = torch.randn(T, 2 * H, device="cuda", dtype=torch.float16)
    return [y]
''',
    },
}

PROMPT = '''You are an expert GPU kernel engineer. You are given a PyTorch module `Model`.
Write an optimized replacement `ModelNew` (same forward signature and outputs) whose
forward uses a custom **Triton** kernel (`@triton.jit`) to run faster.

HARD RULES (your answer is rejected otherwise):
- Must define and call at least one real `@triton.jit` kernel that does the core compute.
- Do NOT call torch.nn.functional.scaled_dot_product_attention, any flash_attn / xformers
  library, or otherwise offload the whole computation to a prebuilt fused operator.
- Must be numerically correct vs the reference and must run on the given input shapes.
- Output ONLY one Python code block with the full code for `ModelNew` (and its Triton
  kernel + imports). No prose, no tests.

Reference module:
```python
{ref}
```
'''


def extract_code(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m[0] if m else text).strip()


def static_check(src):
    """Anti-reward-hack gate. Returns (ok, reason)."""
    if "@triton.jit" not in src and "triton.jit" not in src:
        return False, "no @triton.jit kernel"
    banned = ["scaled_dot_product_attention", "flash_attn", "xformers", "F.sdpa"]
    for b in banned:
        if b in src:
            return False, f"uses banned op: {b}"
    return True, "ok"


@app.function(gpu="L40S", timeout=1200)
def evaluate(ref_src: str, gen_src: str, samples: list):
    import math, torch, triton  # noqa
    import importlib.util, sys

    def load_module(src, name):
        # Triton's @triton.jit needs real source on disk (inspect.getsource),
        # so we import from a file rather than exec() in a dict.
        path = f"/tmp/{name}.py"
        with open(path, "w") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # build reference + inputs
    refmod = load_module(ref_src, "kb_ref")
    Model = refmod.Model
    get_inputs = refmod.get_inputs

    def bench(fn, *a, warmup=25, iters=100):
        for _ in range(warmup):
            fn(*a)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            s.record(); fn(*a); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return sorted(ts)[iters // 2]

    out = {"correct": False, "speedup_vs_compile": None, "speedup_vs_eager": None,
           "max_err": None, "error": None}
    try:
        inputs = get_inputs()
        ref = Model().cuda()
        # fp32 ground truth: upcast only floating tensors; keep int tensors (e.g. packed
        # quant weights) intact so bit-ops still work.
        gt_inputs = [t.float() if t.is_floating_point() else t for t in inputs]
        gt = Model().cuda()(*gt_inputs)

        genmod = load_module(gen_src, "kb_gen")
        ModelNew = genmod.ModelNew
        mnew = ModelNew().cuda()
        y = mnew(*inputs)

        max_err = (y.float() - gt).abs().max().item()
        out["max_err"] = max_err
        out["correct"] = bool(torch.allclose(y.float(), gt, atol=2e-2, rtol=2e-2))

        t_eager = bench(lambda: ref(*inputs))
        try:
            compiled = torch.compile(lambda *a: ref(*a), mode="max-autotune")
            compiled(*inputs)
            t_compile = bench(lambda: compiled(*inputs))
        except Exception:
            t_compile = t_eager  # fall back if compile fails on this op
        t_new = bench(lambda: mnew(*inputs))
        out["speedup_vs_compile"] = t_compile / t_new
        out["speedup_vs_eager"] = t_eager / t_new
    except Exception as exc:
        import traceback
        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
    return out


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO, "results", "runs")


@app.local_entrypoint()
def main(model: str = "openrouter/openai/gpt-5.5", task: str = "attention_causal", samples: int = 2):
    import json
    from datetime import datetime, timezone
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv("/Users/lilianli/Downloads/bench/KernelBench/.env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    t = TASKS[task]
    prompt = PROMPT.format(ref=t["ref"].strip())
    model_slug = model.replace("openrouter/", "")

    # persistent run dir: results/runs/<timestamp>__<model>__<task>/
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"{ts}__{model_slug.replace('/', '-')}__{task}")
    os.makedirs(run_dir, exist_ok=True)

    records, n_pass = [], 0
    for i in range(samples):
        print(f"\n===== sample {i+1}/{samples}  model={model_slug}  task={task} =====")
        rec = {"sample": i, "static_ok": False, "correct": None, "max_err": None,
               "speedup_vs_compile": None, "speedup_vs_eager": None, "pass": False, "error": None}
        try:
            resp = client.chat.completions.create(
                model=model_slug, messages=[{"role": "user", "content": prompt}],
                max_tokens=16384, temperature=0.7)
            gen = extract_code(resp.choices[0].message.content or "")
        except Exception as e:
            rec["error"] = f"gen: {str(e)[:300]}"
            records.append(rec); print("  gen error:", rec["error"]); continue
        open(os.path.join(run_dir, f"sample{i}_kernel.py"), "w").write(gen)  # save trajectory
        ok, reason = static_check(gen)
        rec["static_ok"] = ok
        print(f"static_check: {ok} ({reason})  | gen {len(gen)} chars")
        if not ok:
            rec["error"] = f"static: {reason}"; records.append(rec); print("  -> FAIL (static)"); continue
        r = evaluate.remote(t["ref"], gen, [])
        passed = bool(r["correct"]) and (r["speedup_vs_compile"] or 0) >= 1.0
        rec.update({"correct": r["correct"], "max_err": r["max_err"],
                    "speedup_vs_compile": r["speedup_vs_compile"],
                    "speedup_vs_eager": r["speedup_vs_eager"], "pass": passed, "error": r["error"]})
        n_pass += int(passed)
        records.append(rec)
        print(f"  correct={r['correct']}  max_err={r['max_err']}  "
              f"vs_compile={r['speedup_vs_compile']}  vs_eager={r['speedup_vs_eager']}  PASS={passed}")
        if r["error"]:
            print("  error:", r["error"][:600])

    summary = {"timestamp": ts, "model": model_slug, "task": task, "samples": samples,
               "n_pass": n_pass, "pass_at_1": n_pass / samples if samples else 0,
               "records": records}
    # per-run json + append one line to the master log
    json.dump(summary, open(os.path.join(run_dir, "result.json"), "w"), indent=2)
    with open(os.path.join(RUNS_DIR, "all_results.jsonl"), "a") as f:
        f.write(json.dumps({k: summary[k] for k in
                            ("timestamp", "model", "task", "samples", "n_pass", "pass_at_1")}) + "\n")
    print(f"\n==== pass@1 over {samples}: {n_pass}/{samples}  |  saved -> {run_dir} ====")
