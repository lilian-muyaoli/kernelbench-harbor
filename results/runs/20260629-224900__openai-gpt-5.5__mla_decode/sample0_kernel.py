import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

H, DH, DC = 16, 128, 512


@triton.jit
def _mla_q_transform_kernel(q, W_uk, tq, SCALE: tl.constexpr, BLOCK_DC: tl.constexpr, BLOCK_DH: tl.constexpr):
    pid_dc = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    offs_dc = pid_dc * BLOCK_DC + tl.arange(0, BLOCK_DC)
    offs_d = tl.arange(0, BLOCK_DH)

    qv = tl.load(q + (b * H + h) * DH + offs_d)
    w = tl.load(W_uk + offs_dc[:, None] * (H * DH) + (h * DH + offs_d[None, :]))

    acc = tl.dot(w, qv[:, None], out_dtype=tl.float32)[:, 0] * SCALE
    tl.store(tq + (b * H + h) * DC + offs_dc, acc)


@triton.jit
def _mla_scores_kernel(c_kv, tq, scores, L: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_DC: tl.constexpr):
    pid_l = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_dc = tl.arange(0, BLOCK_DC)

    acc = tl.zeros((BLOCK_L,), tl.float32)

    for dc0 in range(0, DC, BLOCK_DC):
        dc = dc0 + offs_dc
        c = tl.load(
            c_kv + (b * L + offs_l[:, None]) * DC + dc[None, :],
            mask=offs_l[:, None] < L,
            other=0.0,
        )
        t = tl.load(tq + (b * H + h) * DC + dc)
        acc += tl.sum(c.to(tl.float32) * t[None, :], axis=1)

    tl.store(
        scores + (b * H + h) * L + offs_l,
        acc,
        mask=offs_l < L,
    )


@triton.jit
def _mla_softmax_u_kernel(c_kv, scores, u, L: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_DC: tl.constexpr):
    pid_dc = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    offs_dc = pid_dc * BLOCK_DC + tl.arange(0, BLOCK_DC)
    offs_l = tl.arange(0, BLOCK_L)

    m = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0.0, tl.float32)

    for l0 in range(0, L, BLOCK_L):
        l = l0 + offs_l
        s = tl.load(scores + (b * H + h) * L + l, mask=l < L, other=-float("inf"))
        tile_m = tl.max(s, axis=0)
        new_m = tl.maximum(m, tile_m)
        denom = denom * tl.exp(m - new_m) + tl.sum(tl.exp(s - new_m), axis=0)
        m = new_m

    acc = tl.zeros((BLOCK_DC,), tl.float32)

    for l0 in range(0, L, BLOCK_L):
        l = l0 + offs_l
        s = tl.load(scores + (b * H + h) * L + l, mask=l < L, other=-float("inf"))
        p = tl.exp(s - m)
        c = tl.load(
            c_kv + (b * L + l[:, None]) * DC + offs_dc[None, :],
            mask=l[:, None] < L,
            other=0.0,
        )
        acc += tl.sum(p[:, None] * c.to(tl.float32), axis=0)

    acc = acc / denom
    tl.store(u + (b * H + h) * DC + offs_dc, acc)


@triton.jit
def _mla_project_kernel(u, W_uv, out, BLOCK_D: tl.constexpr, BLOCK_DC: tl.constexpr):
    pid_d = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_dc = tl.arange(0, BLOCK_DC)

    acc = tl.zeros((BLOCK_D,), tl.float32)

    for dc0 in range(0, DC, BLOCK_DC):
        dc = dc0 + offs_dc
        uv = tl.load(u + (b * H + h) * DC + dc)
        w = tl.load(
            W_uv + dc[:, None] * (H * DH) + (h * DH + offs_d[None, :]),
            mask=offs_d[None, :] < DH,
            other=0.0,
        )
        acc += tl.sum(uv[:, None] * w.to(tl.float32), axis=0)

    tl.store(out + (b * H + h) * DH + offs_d, acc, mask=offs_d < DH)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, c_kv, W_uk, W_uv):
        q = q.contiguous()
        c_kv = c_kv.contiguous()
        W_uk = W_uk.contiguous()
        W_uv = W_uv.contiguous()

        B = q.shape[0]
        L = c_kv.shape[1]

        tq = torch.empty((B, H, DC), device=q.device, dtype=torch.float32)
        scores = torch.empty((B, H, L), device=q.device, dtype=torch.float32)
        u = torch.empty((B, H, DC), device=q.device, dtype=torch.float32)
        out = torch.empty_like(q)

        _mla_q_transform_kernel[(triton.cdiv(DC, 32), H, B)](
            q, W_uk, tq, 1.0 / math.sqrt(DH), BLOCK_DC=32, BLOCK_DH=128, num_warps=4
        )

        _mla_scores_kernel[(triton.cdiv(L, 64), H, B)](
            c_kv, tq, scores, L, BLOCK_L=64, BLOCK_DC=64, num_warps=4
        )

        _mla_softmax_u_kernel[(triton.cdiv(DC, 32), H, B)](
            c_kv, scores, u, L, BLOCK_L=64, BLOCK_DC=32, num_warps=4
        )

        _mla_project_kernel[(triton.cdiv(DH, 32), H, B)](
            u, W_uv, out, BLOCK_D=32, BLOCK_DC=64, num_warps=4
        )

        return out