import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

H, DH, DC = 16, 128, 512


@triton.jit
def _causal_attn_kernel(
    Q, K, V, O,
    S: tl.constexpr,
    H_: tl.constexpr,
    DH_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE_LOG2E: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H_
    hidx = pid_bh - b * H_

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + ((b * S + offs_m[:, None]) * H_ + hidx) * DH_ + offs_d[None, :]
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < S) & (offs_d[None, :] < DH_), other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K + ((b * S + offs_n[None, :]) * H_ + hidx) * DH_ + offs_d[:, None]
        v_ptrs = V + ((b * S + offs_n[:, None]) * H_ + hidx) * DH_ + offs_d[None, :]

        k = tl.load(k_ptrs, mask=(offs_n[None, :] < S) & (offs_d[:, None] < DH_), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < S) & (offs_d[None, :] < DH_), other=0.0)

        qk = tl.dot(q, k) * SCALE_LOG2E
        causal_mask = offs_n[None, :] <= offs_m[:, None]
        qk = tl.where(causal_mask & (offs_n[None, :] < S) & (offs_m[:, None] < S), qk, -1.0e20)

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])

        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)

        m_i = m_new
        l_i = l_new

    out = acc / l_i[:, None]
    o_ptrs = O + ((b * S + offs_m[:, None]) * H_ + hidx) * DH_ + offs_d[None, :]
    tl.store(o_ptrs, out, mask=(offs_m[:, None] < S) & (offs_d[None, :] < DH_))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h, W_dkv, W_uk, W_uv, W_q):
        B, S, D = h.shape

        c = h @ W_dkv
        q = (h @ W_q).view(B, S, H, DH).contiguous()
        k = (c @ W_uk).view(B, S, H, DH).contiguous()
        v = (c @ W_uv).view(B, S, H, DH).contiguous()

        out = torch.empty((B, S, H, DH), device=h.device, dtype=h.dtype)

        block_m = 16
        block_n = 64
        grid = (triton.cdiv(S, block_m), B * H)

        _causal_attn_kernel[grid](
            q, k, v, out,
            S, H, DH,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=DH,
            SCALE_LOG2E=(1.0 / math.sqrt(DH)) * 1.4426950408889634,
            num_warps=4,
            num_stages=3,
        )

        return out.reshape(B, S, H * DH)