import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

H, DH, DC = 16, 128, 512


@triton.jit
def _qk_proj_kernel(q_ptr, w_ptr, g_ptr, scale: tl.constexpr,
                    BLOCK_DC: tl.constexpr, BLOCK_DH: tl.constexpr,
                    H_: tl.constexpr, DH_: tl.constexpr, DC_: tl.constexpr):
    pid_dc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_dc = pid_dc * BLOCK_DC + tl.arange(0, BLOCK_DC)
    offs_d = tl.arange(0, BLOCK_DH)

    q = tl.load(q_ptr + (pid_b * H_ + pid_h) * DH_ + offs_d).to(tl.float32)
    w = tl.load(
        w_ptr + offs_dc[:, None] * (H_ * DH_) + pid_h * DH_ + offs_d[None, :],
        mask=offs_dc[:, None] < DC_,
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(w * q[None, :], axis=1) * scale
    tl.store(g_ptr + (pid_b * H_ + pid_h) * DC_ + offs_dc, acc, mask=offs_dc < DC_)


@triton.jit
def _logits_kernel(c_ptr, g_ptr, logits_ptr,
                   L_: tl.constexpr, BLOCK_L: tl.constexpr,
                   H_: tl.constexpr, DC_: tl.constexpr):
    pid_l = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_dc = tl.arange(0, DC_)

    c = tl.load(
        c_ptr + (pid_b * L_ + offs_l[:, None]) * DC_ + offs_dc[None, :],
        mask=offs_l[:, None] < L_,
        other=0.0,
    ).to(tl.float32)
    g = tl.load(g_ptr + (pid_b * H_ + pid_h) * DC_ + offs_dc).to(tl.float32)

    scores = tl.sum(c * g[None, :], axis=1)
    tl.store(
        logits_ptr + (pid_b * H_ + pid_h) * L_ + offs_l,
        scores,
        mask=offs_l < L_,
    )


@triton.jit
def _softmax_latent_kernel(c_ptr, logits_ptr, s_ptr,
                           L_: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_DC: tl.constexpr,
                           H_: tl.constexpr, DC_: tl.constexpr):
    pid_dc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_dc = pid_dc * BLOCK_DC + tl.arange(0, BLOCK_DC)
    offs_l = tl.arange(0, BLOCK_L)

    log_base = (pid_b * H_ + pid_h) * L_
    c_base = pid_b * L_ * DC_

    m = tl.full((), -3.4028234663852886e38, tl.float32)

    for start in tl.range(0, L_, BLOCK_L):
        ls = start + offs_l
        x = tl.load(logits_ptr + log_base + ls, mask=ls < L_, other=-3.4028234663852886e38)
        m = tl.maximum(m, tl.max(x, axis=0))

    denom = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_DC,), tl.float32)

    for start in tl.range(0, L_, BLOCK_L):
        ls = start + offs_l
        x = tl.load(logits_ptr + log_base + ls, mask=ls < L_, other=-3.4028234663852886e38)
        p = tl.exp(x - m)
        denom += tl.sum(p, axis=0)

        c = tl.load(
            c_ptr + c_base + ls[:, None] * DC_ + offs_dc[None, :],
            mask=(ls[:, None] < L_) & (offs_dc[None, :] < DC_),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(c * p[:, None], axis=0)

    out = acc / denom
    tl.store(s_ptr + (pid_b * H_ + pid_h) * DC_ + offs_dc, out, mask=offs_dc < DC_)


@triton.jit
def _out_proj_kernel(s_ptr, w_ptr, out_ptr,
                     BLOCK_D: tl.constexpr,
                     H_: tl.constexpr, DH_: tl.constexpr, DC_: tl.constexpr):
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_dc = tl.arange(0, DC_)

    s = tl.load(s_ptr + (pid_b * H_ + pid_h) * DC_ + offs_dc).to(tl.float32)
    w = tl.load(
        w_ptr + offs_dc[:, None] * (H_ * DH_) + pid_h * DH_ + offs_d[None, :],
        mask=offs_d[None, :] < DH_,
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(s[:, None] * w, axis=0)
    tl.store(out_ptr + (pid_b * H_ + pid_h) * DH_ + offs_d, acc, mask=offs_d < DH_)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, c_kv, W_uk, W_uv):
        B = q.shape[0]
        L = c_kv.shape[1]

        q = q.contiguous()
        c_kv = c_kv.contiguous()
        W_uk = W_uk.contiguous()
        W_uv = W_uv.contiguous()

        g = torch.empty((B, H, DC), device=q.device, dtype=torch.float32)
        logits = torch.empty((B, H, L), device=q.device, dtype=torch.float32)
        s = torch.empty((B, H, DC), device=q.device, dtype=torch.float32)
        out = torch.empty((B, H, DH), device=q.device, dtype=q.dtype)

        _qk_proj_kernel[(triton.cdiv(DC, 64), H, B)](
            q, W_uk, g, 1.0 / math.sqrt(DH),
            BLOCK_DC=64, BLOCK_DH=DH,
            H_=H, DH_=DH, DC_=DC,
            num_warps=4,
        )

        _logits_kernel[(triton.cdiv(L, 16), H, B)](
            c_kv, g, logits,
            L_=L, BLOCK_L=16,
            H_=H, DC_=DC,
            num_warps=8,
        )

        _softmax_latent_kernel[(triton.cdiv(DC, 64), H, B)](
            c_kv, logits, s,
            L_=L, BLOCK_L=32, BLOCK_DC=64,
            H_=H, DC_=DC,
            num_warps=4,
        )

        _out_proj_kernel[(triton.cdiv(DH, 32), H, B)](
            s, W_uv, out,
            BLOCK_D=32,
            H_=H, DH_=DH, DC_=DC,
            num_warps=8,
        )

        return out