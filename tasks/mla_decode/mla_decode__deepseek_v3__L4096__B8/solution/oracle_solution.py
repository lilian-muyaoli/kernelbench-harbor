import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

H, DH, DC = 16, 128, 512


@triton.jit
def _mla_make_r_kernel(q, W_uk, R,
                       BLOCK_DC: tl.constexpr,
                       H_: tl.constexpr, DH_: tl.constexpr, DC_: tl.constexpr):
    bh = tl.program_id(0)
    pid_dc = tl.program_id(1)
    h = bh % H_

    offs_dc = pid_dc * BLOCK_DC + tl.arange(0, BLOCK_DC)
    offs_dh = tl.arange(0, DH_)

    qv = tl.load(q + bh * DH_ + offs_dh).to(tl.float32)
    w = tl.load(
        W_uk + offs_dc[:, None] * (H_ * DH_) + (h * DH_ + offs_dh[None, :]),
        mask=offs_dc[:, None] < DC_,
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(w * qv[None, :], axis=1)
    tl.store(R + bh * DC_ + offs_dc, acc, mask=offs_dc < DC_)


@triton.jit
def _mla_partial_latent_kernel(c_kv, R, partial, m_buf, l_buf,
                               L: tl.constexpr, NB: tl.constexpr,
                               BLOCK_L: tl.constexpr,
                               BLOCK_K: tl.constexpr,
                               BLOCK_D_ALL: tl.constexpr,
                               H_: tl.constexpr, DH_: tl.constexpr, DC_: tl.constexpr,
                               SCALE: tl.constexpr):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H_

    offs_l = pid_n * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = offs_l < L

    scores = tl.zeros((BLOCK_L,), dtype=tl.float32)

    for k0 in tl.static_range(0, DC_, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        c = tl.load(
            c_kv + (b * L + offs_l[:, None]) * DC_ + offs_k[None, :],
            mask=mask_l[:, None],
            other=0.0,
        ).to(tl.float32)
        r = tl.load(R + bh * DC_ + offs_k).to(tl.float32)
        scores += tl.sum(c * r[None, :], axis=1)

    scores = scores * SCALE
    scores = tl.where(mask_l, scores, -float("inf"))

    m = tl.max(scores, axis=0)
    e = tl.exp(scores - m)
    e = tl.where(mask_l, e, 0.0)
    lsum = tl.sum(e, axis=0)

    tl.store(m_buf + bh * NB + pid_n, m)
    tl.store(l_buf + bh * NB + pid_n, lsum)

    offs_d = tl.arange(0, BLOCK_D_ALL)
    c_full = tl.load(
        c_kv + (b * L + offs_l[:, None]) * DC_ + offs_d[None, :],
        mask=mask_l[:, None],
        other=0.0,
    ).to(tl.float32)

    part = tl.sum(c_full * e[:, None], axis=0)
    tl.store(partial + (bh * NB + pid_n) * DC_ + offs_d, part)


@triton.jit
def _mla_reduce_latent_kernel(partial, m_buf, l_buf, Z,
                              NB: tl.constexpr,
                              BLOCK_NB: tl.constexpr,
                              BLOCK_DC: tl.constexpr,
                              DC_: tl.constexpr):
    bh = tl.program_id(0)
    pid_dc = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_NB)
    mask_n = offs_n < NB

    mv = tl.load(m_buf + bh * NB + offs_n, mask=mask_n, other=-float("inf")).to(tl.float32)
    lv = tl.load(l_buf + bh * NB + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    gm = tl.max(mv, axis=0)
    w = tl.exp(mv - gm)
    w = tl.where(mask_n, w, 0.0)
    denom = tl.sum(lv * w, axis=0)

    offs_d = pid_dc * BLOCK_DC + tl.arange(0, BLOCK_DC)
    p = tl.load(
        partial + (bh * NB + offs_n[:, None]) * DC_ + offs_d[None, :],
        mask=mask_n[:, None] & (offs_d[None, :] < DC_),
        other=0.0,
    ).to(tl.float32)

    z = tl.sum(p * w[:, None], axis=0) / denom
    tl.store(Z + bh * DC_ + offs_d, z, mask=offs_d < DC_)


@triton.jit
def _mla_final_kernel(Z, W_uv, out,
                      BLOCK_DH: tl.constexpr,
                      BLOCK_K: tl.constexpr,
                      H_: tl.constexpr, DH_: tl.constexpr, DC_: tl.constexpr):
    bh = tl.program_id(0)
    pid_dh = tl.program_id(1)
    h = bh % H_

    offs_dh = pid_dh * BLOCK_DH + tl.arange(0, BLOCK_DH)
    acc = tl.zeros((BLOCK_DH,), dtype=tl.float32)

    for k0 in tl.static_range(0, DC_, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        z = tl.load(Z + bh * DC_ + offs_k).to(tl.float32)
        w = tl.load(
            W_uv + offs_k[:, None] * (H_ * DH_) + (h * DH_ + offs_dh[None, :]),
            mask=offs_dh[None, :] < DH_,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(z[:, None] * w, axis=0)

    tl.store(out + bh * DH_ + offs_dh, acc, mask=offs_dh < DH_)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, c_kv, W_uk, W_uv):
        B, L, _ = c_kv.shape

        BLOCK_L = 16
        NB = triton.cdiv(L, BLOCK_L)
        BLOCK_NB = triton.next_power_of_2(NB)

        R = torch.empty((B, H, DC), device=q.device, dtype=torch.float32)
        partial = torch.empty((B, H, NB, DC), device=q.device, dtype=torch.float32)
        m_buf = torch.empty((B, H, NB), device=q.device, dtype=torch.float32)
        l_buf = torch.empty((B, H, NB), device=q.device, dtype=torch.float32)
        Z = torch.empty((B, H, DC), device=q.device, dtype=torch.float32)
        out = torch.empty_like(q)

        _mla_make_r_kernel[(B * H, triton.cdiv(DC, 32))](
            q, W_uk, R,
            BLOCK_DC=32,
            H_=H, DH_=DH, DC_=DC,
            num_warps=4,
        )

        _mla_partial_latent_kernel[(NB, B * H)](
            c_kv, R, partial, m_buf, l_buf,
            L=L, NB=NB,
            BLOCK_L=BLOCK_L,
            BLOCK_K=64,
            BLOCK_D_ALL=DC,
            H_=H, DH_=DH, DC_=DC,
            SCALE=1.0 / math.sqrt(DH),
            num_warps=8,
            num_stages=3,
        )

        _mla_reduce_latent_kernel[(B * H, triton.cdiv(DC, 64))](
            partial, m_buf, l_buf, Z,
            NB=NB,
            BLOCK_NB=BLOCK_NB,
            BLOCK_DC=64,
            DC_=DC,
            num_warps=8,
            num_stages=3,
        )

        _mla_final_kernel[(B * H, triton.cdiv(DH, 64))](
            Z, W_uv, out,
            BLOCK_DH=64,
            BLOCK_K=64,
            H_=H, DH_=DH, DC_=DC,
            num_warps=4,
            num_stages=3,
        )

        return out