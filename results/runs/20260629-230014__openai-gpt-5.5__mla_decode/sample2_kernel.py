import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

H, DH, DC = 16, 128, 512


@triton.jit
def _mla_q_proj_kernel(q, W_uk, u, SCALE: tl.constexpr,
                       H_: tl.constexpr, DH_: tl.constexpr, DC_: tl.constexpr,
                       BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_cb = tl.program_id(1)

    h = pid_bh % H_
    offs_c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_d = tl.arange(0, BLOCK_D)

    qv = tl.load(q + pid_bh * DH_ + offs_d, mask=offs_d < DH_, other=0.0).to(tl.float32)
    w = tl.load(
        W_uk + offs_c[:, None] * (H_ * DH_) + h * DH_ + offs_d[None, :],
        mask=(offs_c[:, None] < DC_) & (offs_d[None, :] < DH_),
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(w * qv[None, :], axis=1) * SCALE
    tl.store(u + pid_bh * DC_ + offs_c, acc, mask=offs_c < DC_)


@triton.jit
def _mla_scores_kernel(c_kv, u, scores,
                       L_: tl.constexpr, H_: tl.constexpr, DC_: tl.constexpr,
                       BLOCK_L: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_lb = tl.program_id(1)

    b = pid_bh // H_
    offs_l = pid_lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_c_base = tl.arange(0, BLOCK_C)

    acc = tl.zeros((BLOCK_L,), tl.float32)

    for c0 in tl.range(0, DC_, BLOCK_C):
        offs_c = c0 + offs_c_base
        cv = tl.load(
            c_kv + b * L_ * DC_ + offs_l[:, None] * DC_ + offs_c[None, :],
            mask=(offs_l[:, None] < L_) & (offs_c[None, :] < DC_),
            other=0.0,
        ).to(tl.float32)
        uv = tl.load(u + pid_bh * DC_ + offs_c, mask=offs_c < DC_, other=0.0).to(tl.float32)
        acc += tl.sum(cv * uv[None, :], axis=1)

    tl.store(scores + pid_bh * L_ + offs_l, acc, mask=offs_l < L_)


@triton.jit
def _mla_softmax_kernel(scores, att,
                        L_: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_bh = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)

    x = tl.load(scores + pid_bh * L_ + offs, mask=offs < L_, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(att + pid_bh * L_ + offs, y, mask=offs < L_)


@triton.jit
def _mla_z_kernel(c_kv, att, z,
                  L_: tl.constexpr, H_: tl.constexpr, DC_: tl.constexpr,
                  BLOCK_L: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_cb = tl.program_id(1)

    b = pid_bh // H_
    offs_c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_l_base = tl.arange(0, BLOCK_L)

    acc = tl.zeros((BLOCK_C,), tl.float32)

    for l0 in tl.range(0, L_, BLOCK_L):
        offs_l = l0 + offs_l_base
        av = tl.load(att + pid_bh * L_ + offs_l, mask=offs_l < L_, other=0.0).to(tl.float32)
        cv = tl.load(
            c_kv + b * L_ * DC_ + offs_l[:, None] * DC_ + offs_c[None, :],
            mask=(offs_l[:, None] < L_) & (offs_c[None, :] < DC_),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(cv * av[:, None], axis=0)

    tl.store(z + pid_bh * DC_ + offs_c, acc, mask=offs_c < DC_)


@triton.jit
def _mla_out_kernel(z, W_uv, out,
                    H_: tl.constexpr, DH_: tl.constexpr, DC_: tl.constexpr,
                    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_db = tl.program_id(1)

    h = pid_bh % H_
    offs_d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_c_base = tl.arange(0, BLOCK_C)

    acc = tl.zeros((BLOCK_D,), tl.float32)

    for c0 in tl.range(0, DC_, BLOCK_C):
        offs_c = c0 + offs_c_base
        zv = tl.load(z + pid_bh * DC_ + offs_c, mask=offs_c < DC_, other=0.0).to(tl.float32)
        w = tl.load(
            W_uv + offs_c[:, None] * (H_ * DH_) + h * DH_ + offs_d[None, :],
            mask=(offs_c[:, None] < DC_) & (offs_d[None, :] < DH_),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * zv[:, None], axis=0)

    tl.store(out + pid_bh * DH_ + offs_d, acc, mask=offs_d < DH_)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, c_kv, W_uk, W_uv):
        B, L, _ = c_kv.shape

        u = torch.empty((B, H, DC), device=q.device, dtype=torch.float32)
        scores = torch.empty((B, H, L), device=q.device, dtype=torch.float32)
        att = torch.empty((B, H, L), device=q.device, dtype=torch.float32)
        z = torch.empty((B, H, DC), device=q.device, dtype=torch.float32)
        out = torch.empty((B, H, DH), device=q.device, dtype=q.dtype)

        bc_q = 32
        _mla_q_proj_kernel[(B * H, triton.cdiv(DC, bc_q))](
            q, W_uk, u,
            SCALE=1.0 / math.sqrt(DH),
            H_=H, DH_=DH, DC_=DC,
            BLOCK_C=bc_q, BLOCK_D=DH,
            num_warps=4,
        )

        bl_s = 16
        bc_s = 64
        _mla_scores_kernel[(B * H, triton.cdiv(L, bl_s))](
            c_kv, u, scores,
            L_=L, H_=H, DC_=DC,
            BLOCK_L=bl_s, BLOCK_C=bc_s,
            num_warps=4,
        )

        block_n = triton.next_power_of_2(L)
        _mla_softmax_kernel[(B * H,)](
            scores, att,
            L_=L, BLOCK_N=block_n,
            num_warps=8,
        )

        bl_z = 64
        bc_z = 64
        _mla_z_kernel[(B * H, triton.cdiv(DC, bc_z))](
            c_kv, att, z,
            L_=L, H_=H, DC_=DC,
            BLOCK_L=bl_z, BLOCK_C=bc_z,
            num_warps=4,
        )

        bc_o = 64
        bd_o = 32
        _mla_out_kernel[(B * H, triton.cdiv(DH, bd_o))](
            z, W_uv, out,
            H_=H, DH_=DH, DC_=DC,
            BLOCK_C=bc_o, BLOCK_D=bd_o,
            num_warps=4,
        )

        return out