import torch
import torch.nn as nn
import triton
import triton.language as tl

BS = 64
TOPB = 4


@triton.jit
def _block_mean_kernel(
    k, kb,
    N: tl.constexpr,
    NB: tl.constexpr,
    D: tl.constexpr,
    BSZ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NB
    blk = pid - bh * NB

    offs_s = tl.arange(0, BSZ)
    offs_d = tl.arange(0, BLOCK_D)

    ptrs = k + ((bh * N + blk * BSZ + offs_s[:, None]) * D + offs_d[None, :])
    mask = offs_d[None, :] < D
    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    avg = tl.sum(vals, axis=0) / BSZ

    out_ptrs = kb + ((bh * NB + blk) * D + offs_d)
    tl.store(out_ptrs, avg, mask=offs_d < D)


@triton.jit
def _blocksparse_attn_kernel(
    q, k, v, kb, out,
    N: tl.constexpr,
    NB: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BSZ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // N
    nidx = pid - bh * N
    qblk = nidx // BSZ

    offs_d = tl.arange(0, BLOCK_D)
    dmask = offs_d < D

    qv = tl.load(q + (bh * N + nidx) * D + offs_d, mask=dmask, other=0.0).to(tl.float32)

    neg_inf = -float("inf")
    s0 = tl.full((), neg_inf, tl.float32)
    s1 = tl.full((), neg_inf, tl.float32)
    s2 = tl.full((), neg_inf, tl.float32)
    s3 = tl.full((), neg_inf, tl.float32)
    i0 = tl.full((), 0, tl.int32)
    i1 = tl.full((), 0, tl.int32)
    i2 = tl.full((), 0, tl.int32)
    i3 = tl.full((), 0, tl.int32)

    for blk in range(0, NB):
        kbv = tl.load(kb + (bh * NB + blk) * D + offs_d, mask=dmask, other=0.0).to(tl.float32)
        sc = tl.sum(qv * kbv, axis=0)
        sc = tl.where(blk <= qblk, sc, neg_inf)

        c0 = sc > s0
        s3 = tl.where(c0, s2, s3)
        i3 = tl.where(c0, i2, i3)
        s2 = tl.where(c0, s1, s2)
        i2 = tl.where(c0, i1, i2)
        s1 = tl.where(c0, s0, s1)
        i1 = tl.where(c0, i0, i1)
        s0 = tl.where(c0, sc, s0)
        i0 = tl.where(c0, blk, i0)

        c1 = (sc > s1) & (~c0)
        s3 = tl.where(c1, s2, s3)
        i3 = tl.where(c1, i2, i3)
        s2 = tl.where(c1, s1, s2)
        i2 = tl.where(c1, i1, i2)
        s1 = tl.where(c1, sc, s1)
        i1 = tl.where(c1, blk, i1)

        c2 = (sc > s2) & (~c0) & (~c1)
        s3 = tl.where(c2, s2, s3)
        i3 = tl.where(c2, i2, i3)
        s2 = tl.where(c2, sc, s2)
        i2 = tl.where(c2, blk, i2)

        c3 = (sc > s3) & (~c0) & (~c1) & (~c2)
        s3 = tl.where(c3, sc, s3)
        i3 = tl.where(c3, blk, i3)

    valid_slots = tl.minimum(qblk + 1, 4)

    offs_n = tl.arange(0, BSZ)
    m_i = tl.full((), neg_inf, tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)

    for slot in tl.static_range(0, 4):
        sel_blk = i0
        if slot == 1:
            sel_blk = i1
        if slot == 2:
            sel_blk = i2
        if slot == 3:
            sel_blk = i3

        pos = sel_blk * BSZ + offs_n
        nmask = (slot < valid_slots) & (pos <= nidx)

        kptrs = k + ((bh * N + pos[:, None]) * D + offs_d[None, :])
        kt = tl.load(kptrs, mask=nmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(kt * qv[None, :], axis=1) * SCALE
        scores = tl.where(nmask, scores, neg_inf)

        m_tile = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        vptrs = v + ((bh * N + pos[:, None]) * D + offs_d[None, :])
        vt = tl.load(vptrs, mask=nmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * vt, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    res = acc / l_i
    tl.store(out + (bh * N + nidx) * D + offs_d, res, mask=dmask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        B, H, N, D = q.shape
        nb = N // BS
        block_d = 1 << (D - 1).bit_length()

        kb = torch.empty((B, H, nb, D), device=k.device, dtype=k.dtype)
        out = torch.empty_like(q)

        _block_mean_kernel[(B * H * nb,)](
            k, kb,
            N, nb, D,
            BSZ=BS,
            BLOCK_D=block_d,
            num_warps=4,
        )

        _blocksparse_attn_kernel[(B * H * N,)](
            q, k, v, kb, out,
            N, nb, D, D ** -0.5,
            BSZ=BS,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=3,
        )
        return out