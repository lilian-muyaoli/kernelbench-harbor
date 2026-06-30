import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _moe_top2_fused_dense_kernel(
    x_ptr,
    w_ptr,
    topi_ptr,
    topv_ptr,
    out_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < T
    mask_n = offs_n < H

    idx0 = tl.load(topi_ptr + offs_m * 2 + 0, mask=mask_m, other=-1)
    idx1 = tl.load(topi_ptr + offs_m * 2 + 1, mask=mask_m, other=-1)
    val0 = tl.load(topv_ptr + offs_m * 2 + 0, mask=mask_m, other=0.0).to(tl.float32)
    val1 = tl.load(topv_ptr + offs_m * 2 + 1, mask=mask_m, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in tl.range(0, D, BLOCK_K):
        k = k0 + offs_k

        a = tl.load(
            x_ptr + offs_m[:, None] * D + k[None, :],
            mask=(offs_m[:, None] < T) & (k[None, :] < D),
            other=0.0,
        )

        for e in tl.static_range(0, 8):
            gate = tl.where(idx0 == e, val0, 0.0) + tl.where(idx1 == e, val1, 0.0)

            b = tl.load(
                w_ptr + e * D * H + k[:, None] * H + offs_n[None, :],
                mask=(k[:, None] < D) & (offs_n[None, :] < H),
                other=0.0,
            )

            acc += tl.dot(a, b) * gate[:, None]

    tl.store(
        out_ptr + offs_m[:, None] * H + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < T) & (offs_n[None, :] < H),
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w, router_logits):
        x = x.contiguous()
        w = w.contiguous()

        probs = torch.softmax(router_logits.float(), dim=-1)
        topv, topi = probs.topk(2, dim=-1)
        topv = (topv / topv.sum(dim=-1, keepdim=True)).to(x.dtype).contiguous()
        topi = topi.contiguous()

        T = x.shape[0]
        D = x.shape[1]
        H = w.shape[2]

        out = torch.empty((T, H), device=x.device, dtype=x.dtype)

        BLOCK_M = 16
        BLOCK_N = 32
        BLOCK_K = 64

        grid = (triton.cdiv(T, BLOCK_M), triton.cdiv(H, BLOCK_N))

        _moe_top2_fused_dense_kernel[grid](
            x,
            w,
            topi,
            topv,
            out,
            T,
            D,
            H,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=3,
        )

        return out