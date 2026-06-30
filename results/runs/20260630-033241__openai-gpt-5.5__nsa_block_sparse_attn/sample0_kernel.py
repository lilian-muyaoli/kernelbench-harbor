import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

BS = 64
TOPB = 4


@triton.jit
def _blocksparse_causal_attn_kernel(
    q, k, v, kb, out,
    sqb: tl.constexpr, sqh: tl.constexpr, sqn: tl.constexpr, sqd: tl.constexpr,
    skb: tl.constexpr, skh: tl.constexpr, skn: tl.constexpr, skd: tl.constexpr,
    svb: tl.constexpr, svh: tl.constexpr, svn: tl.constexpr, svd: tl.constexpr,
    skbb: tl.constexpr, skbh: tl.constexpr, skbc: tl.constexpr, skbd: tl.constexpr,
    sob: tl.constexpr, soh: tl.constexpr, son: tl.constexpr, sod: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    NB: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_BS: tl.constexpr,
):
    pid = tl.program_id(0)

    qn = pid % N
    tmp = pid // N
    h = tmp % H
    b = tmp // H

    offs_d = tl.arange(0, BLOCK_D)
    offs_m = tl.arange(0, BLOCK_BS)

    q_ptrs = q + b * sqb + h * sqh + qn * sqn + offs_d * sqd
    q_vec = tl.load(q_ptrs, mask=offs_d < D, other=0.0).to(tl.float32)

    cblk = qn // BLOCK_BS

    s0 = tl.full((), -float("inf"), tl.float32)
    s1 = tl.full((), -float("inf"), tl.float32)
    s2 = tl.full((), -float("inf"), tl.float32)
    s3 = tl.full((), -float("inf"), tl.float32)

    i0 = tl.full((), -1, tl.int32)
    i1 = tl.full((), -1, tl.int32)
    i2 = tl.full((), -1, tl.int32)
    i3 = tl.full((), -1, tl.int32)

    for bid in tl.static_range(0, NB):
        kb_ptrs = kb + b * skbb + h * skbh + bid * skbc + offs_d * skbd
        kb_vec = tl.load(kb_ptrs, mask=offs_d < D, other=0.0).to(tl.float32)

        dot = tl.sum(q_vec * kb_vec, axis=0)
        dot = dot.to(tl.float16).to(tl.float32)
        sc = (dot * SCALE).to(tl.float16).to(tl.float32)
        sc = tl.where(bid <= cblk, sc, -float("inf"))

        os0, os1, os2, os3 = s0, s1, s2, s3
        oi0, oi1, oi2, oi3 = i0, i1, i2, i3

        gt0 = sc > os0
        gt1 = (sc > os1) & (~gt0)
        gt2 = (sc > os2) & (~gt0) & (~gt1)
        gt3 = (sc > os3) & (~gt0) & (~gt1) & (~gt2)

        s0 = tl.where(gt0, sc, os0)
        i0 = tl.where(gt0, bid, oi0)

        s1 = tl.where(gt0, os0, tl.where(gt1, sc, os1))
        i1 = tl.where(gt0, oi0, tl.where(gt1, bid, oi1))

        s2 = tl.where(gt0, os1, tl.where(gt1, os1, tl.where(gt2, sc, os2)))
        i2 = tl.where(gt0, oi1, tl.where(gt1, oi1, tl.where(gt2, bid, oi2)))

        s3 = tl.where(gt0, os2, tl.where(gt1, os2, tl.where(gt2, os2, tl.where(gt3, sc, os3))))
        i3 = tl.where(gt0, oi2, tl.where(gt1, oi2, tl.where(gt2, oi2, tl.where(gt3, bid, oi3))))

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)

    for ti in tl.static_range(0, TOPB):
        if ti == 0:
            blk = i0
        elif ti == 1:
            blk = i1
        elif ti == 2:
            blk = i2
        else:
            blk = i3

        valid_blk = blk >= 0
        key_start = blk * BLOCK_BS

        k_ptrs = (
            k
            + b * skb
            + h * skh
            + (key_start + offs_m[:, None]) * skn
            + offs_d[None, :] * skd
        )
        v_ptrs = (
            v
            + b * svb
            + h * svh
            + (key_start + offs_m[:, None]) * svn
            + offs_d[None, :] * svd
        )

        mat_mask = valid_blk & (offs_m[:, None] < BLOCK_BS) & (offs_d[None, :] < D)
        k_mat = tl.load(k_ptrs, mask=mat_mask, other=0.0).to(tl.float32)
        v_mat = tl.load(v_ptrs, mask=mat_mask, other=0.0).to(tl.float32)

        logits = tl.sum(k_mat * q_vec[None, :], axis=1)
        logits = logits.to(tl.float16).to(tl.float32)
        logits = (logits * SCALE).to(tl.float16).to(tl.float32)

        causal_mask = valid_blk & ((key_start + offs_m) <= qn)
        logits = tl.where(causal_mask, logits, -float("inf"))

        b_m = tl.max(logits, axis=0)
        new_m = tl.maximum(m_i, b_m)

        alpha = tl.exp(m_i - new_m)
        p = tl.exp(logits - new_m)

        acc = acc * alpha + tl.sum(p[:, None] * v_mat, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = new_m

    res = acc / l_i
    out_ptrs = out + b * sob + h * soh + qn * son + offs_d * sod
    tl.store(out_ptrs, res, mask=offs_d < D)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        B, H, N, D = q.shape
        nb = N // BS
        scale = 1.0 / math.sqrt(D)

        kb = k.view(B, H, nb, BS, D).mean(3)
        out = torch.empty_like(q)

        grid = (B * H * N,)

        _blocksparse_causal_attn_kernel[grid](
            q, k, v, kb, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            kb.stride(0), kb.stride(1), kb.stride(2), kb.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            H=H,
            N=N,
            D=D,
            SCALE=scale,
            NB=nb,
            BLOCK_D=triton.next_power_of_2(D),
            BLOCK_BS=BS,
            num_warps=8,
        )
        return out