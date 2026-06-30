import torch
import torch.nn as nn
import triton
import triton.language as tl

H = 16
DH = 128
DC = 512


@triton.jit
def _compute_u_kernel(q, W_uk, U, BLOCK_DC: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_dc = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H

    offs_dc = pid_dc * BLOCK_DC + tl.arange(0, BLOCK_DC)
    offs_d = tl.arange(0, BLOCK_D)

    qv = tl.load(q + (b * H + h) * DH + offs_d).to(tl.float32)
    w = tl.load(
        W_uk + offs_dc[:, None] * (H * DH) + h * DH + offs_d[None, :],
        mask=offs_dc[:, None] < DC,
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(w * qv[None, :], axis=1)
    tl.store(U + pid_bh * DC + offs_dc, acc, mask=offs_dc < DC)


@triton.jit
def _scores_kernel(c_kv, U, scores, L: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_DC: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_l = tl.program_id(1)

    b = pid_bh // H
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_dc = tl.arange(0, BLOCK_DC)

    c = tl.load(
        c_kv + (b * L + offs_l[:, None]) * DC + offs_dc[None, :],
        mask=offs_l[:, None] < L,
        other=0.0,
    ).to(tl.float32)
    u = tl.load(U + pid_bh * DC + offs_dc).to(tl.float32)

    s = tl.sum(c * u[None, :], axis=1) * 0.08838834764831845
    tl.store(scores + pid_bh * L + offs_l, s, mask=offs_l < L)


@triton.jit
def _softmax_z_kernel(c_kv, scores, Z, L: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_DC: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_dc = tl.program_id(1)

    b = pid_bh // H
    offs_dc = pid_dc * BLOCK_DC + tl.arange(0, BLOCK_DC)
    offs_l_base = tl.arange(0, BLOCK_L)

    m = tl.full((), -3.4028234663852886e38, tl.float32)
    denom = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_DC,), tl.float32)

    for start in range(0, L, BLOCK_L):
        offs_l = start + offs_l_base
        s = tl.load(scores + pid_bh * L + offs_l, mask=offs_l < L, other=-3.4028234663852886e38).to(tl.float32)

        blk_m = tl.max(s, axis=0)
        new_m = tl.maximum(m, blk_m)
        alpha = tl.exp(m - new_m)
        p = tl.exp(s - new_m)

        c = tl.load(
            c_kv + (b * L + offs_l[:, None]) * DC + offs_dc[None, :],
            mask=(offs_l[:, None] < L) & (offs_dc[None, :] < DC),
            other=0.0,
        ).to(tl.float32)

        acc = acc * alpha + tl.sum(c * p[:, None], axis=0)
        denom = denom * alpha + tl.sum(p, axis=0)
        m = new_m

    z = acc / denom
    tl.store(Z + pid_bh * DC + offs_dc, z, mask=offs_dc < DC)


@triton.jit
def _out_kernel(Z, W_uv, out, BLOCK_D: tl.constexpr, BLOCK_DC: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_dblk = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H

    offs_d = pid_dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_dc = tl.arange(0, BLOCK_DC)

    z = tl.load(Z + pid_bh * DC + offs_dc).to(tl.float32)
    w = tl.load(
        W_uv + offs_dc[:, None] * (H * DH) + h * DH + offs_d[None, :],
        mask=offs_d[None, :] < DH,
        other=0.0,
    ).to(tl.float32)

    y = tl.sum(z[:, None] * w, axis=0)
    tl.store(out + (b * H + h) * DH + offs_d, y, mask=offs_d < DH)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, c_kv, W_uk, W_uv):
        q = q.contiguous()
        c_kv = c_kv.contiguous()
        W_uk = W_uk.contiguous()
        W_uv = W_uv.contiguous()

        B, L, _ = c_kv.shape
        BH = B * H

        U = torch.empty((BH, DC), device=q.device, dtype=torch.float32)
        scores = torch.empty((BH, L), device=q.device, dtype=torch.float32)
        Z = torch.empty((BH, DC), device=q.device, dtype=torch.float32)
        out = torch.empty((B, H, DH), device=q.device, dtype=q.dtype)

        _compute_u_kernel[(BH, triton.cdiv(DC, 64))](
            q, W_uk, U,
            BLOCK_DC=64,
            BLOCK_D=DH,
            num_warps=4,
        )

        _scores_kernel[(BH, triton.cdiv(L, 16))](
            c_kv, U, scores, L,
            BLOCK_L=16,
            BLOCK_DC=DC,
            num_warps=8,
        )

        _softmax_z_kernel[(BH, triton.cdiv(DC, 64))](
            c_kv, scores, Z, L,
            BLOCK_L=32,
            BLOCK_DC=64,
            num_warps=4,
        )

        _out_kernel[(BH, triton.cdiv(DH, 32))](
            Z, W_uv, out,
            BLOCK_D=32,
            BLOCK_DC=DC,
            num_warps=8,
        )

        return out