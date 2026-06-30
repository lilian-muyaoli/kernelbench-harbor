import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

H, DH, DC = 16, 128, 512


@triton.jit
def _mla_causal_attn_fwd_kernel(
    Q, K, V, O,
    S: tl.constexpr,
    NH: tl.constexpr,
    HD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // NH
    h = pid_bh - b * NH

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    stride_b = S * NH * HD
    stride_s = NH * HD
    stride_h = HD

    base = b * stride_b + h * stride_h

    q = tl.load(
        Q + base + offs_m[:, None] * stride_s + offs_d[None, :],
        mask=(offs_m[:, None] < S) & (offs_d[None, :] < HD),
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -3.4028234663852886e38, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    scale = 0.08838834764831845  # 1 / sqrt(128)

    hi = tl.minimum(S, (pid_m + 1) * BLOCK_M)

    for start_n in tl.range(0, hi, BLOCK_N):
        cur_n = start_n + offs_n

        k = tl.load(
            K + base + cur_n[None, :] * stride_s + offs_d[:, None],
            mask=(cur_n[None, :] < S) & (offs_d[:, None] < HD),
            other=0.0,
        )

        qk = tl.dot(q, k) * scale
        causal = offs_m[:, None] >= cur_n[None, :]
        qk = tl.where(causal & (cur_n[None, :] < S), qk, -3.4028234663852886e38)

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)

        v = tl.load(
            V + base + cur_n[:, None] * stride_s + offs_d[None, :],
            mask=(cur_n[:, None] < S) & (offs_d[None, :] < HD),
            other=0.0,
        )

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    acc = acc / l_i[:, None]

    tl.store(
        O + base + offs_m[:, None] * stride_s + offs_d[None, :],
        acc,
        mask=(offs_m[:, None] < S) & (offs_d[None, :] < HD),
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h, W_dkv, W_uk, W_uv, W_q):
        B, S, D = h.shape

        c = h @ W_dkv
        k = c @ W_uk
        v = c @ W_uv
        q = h @ W_q

        out = torch.empty_like(q)

        grid = (triton.cdiv(S, 16), B * H)

        _mla_causal_attn_fwd_kernel[grid](
            q, k, v, out,
            S,
            H,
            DH,
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_D=128,
            num_warps=4,
            num_stages=3,
        )

        return out