import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _lora_mid_kernel(
    x_ptr,
    A_ptr,
    lora_ids_ptr,
    mid_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    t = tl.program_id(0)
    offs_r = tl.arange(0, BLOCK_R)
    acc = tl.zeros((BLOCK_R,), tl.float32)

    lid = tl.load(lora_ids_ptr + t)

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        xv = tl.load(x_ptr + t * D + offs_d).to(tl.float32)
        av = tl.load(
            A_ptr + lid * D * R + offs_d[:, None] * R + offs_r[None, :],
            mask=offs_r[None, :] < R,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(xv[:, None] * av, axis=0)

    tl.store(mid_ptr + t * R + offs_r, acc, mask=offs_r < R)


@triton.jit
def _base_lora_kernel(
    x_ptr,
    qweight_ptr,
    scales_ptr,
    mid_ptr,
    B_ptr,
    lora_ids_ptr,
    out_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + offs_k_base

        x_tile = tl.load(
            x_ptr + offs_m[:, None] * D + offs_k[None, :],
            mask=(offs_m[:, None] < T) & (offs_k[None, :] < D),
            other=0.0,
        )

        q_tile = tl.load(
            qweight_ptr + offs_k[:, None] * N + offs_n[None, :],
            mask=(offs_k[:, None] < D) & (offs_n[None, :] < N),
            other=0,
        )

        s = tl.load(
            scales_ptr + (k0 // GROUP_SIZE) * N + offs_n,
            mask=offs_n < N,
            other=0.0,
        ).to(tl.float32)

        w_tile = (q_tile.to(tl.float32) * s[None, :]).to(tl.float16)
        acc += tl.dot(x_tile, w_tile, out_dtype=tl.float32)

    lo = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    lids = tl.load(lora_ids_ptr + offs_m, mask=offs_m < T, other=0)

    for rr in range(0, R):
        mv = tl.load(
            mid_ptr + offs_m * R + rr,
            mask=offs_m < T,
            other=0.0,
        ).to(tl.float32)

        bv = tl.load(
            B_ptr + lids[:, None] * R * N + rr * N + offs_n[None, :],
            mask=(offs_m[:, None] < T) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)

        lo += mv[:, None] * bv

    res = acc.to(tl.float16) + lo.to(tl.float16)

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        res,
        mask=(offs_m[:, None] < T) & (offs_n[None, :] < N),
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, qweight, scales, A, B, lora_ids):
        T = x.shape[0]
        D = x.shape[1]
        N = qweight.shape[1]
        R = A.shape[2]

        mid = torch.empty((T, R), device=x.device, dtype=x.dtype)
        out = torch.empty((T, N), device=x.device, dtype=x.dtype)

        block_r = triton.next_power_of_2(R)

        _lora_mid_kernel[(T,)](
            x,
            A,
            lora_ids,
            mid,
            T,
            D,
            R,
            BLOCK_D=128,
            BLOCK_R=block_r,
            num_warps=4,
            num_stages=4,
        )

        BM = 16
        BN = 64
        BK = 128
        grid = (triton.cdiv(T, BM), triton.cdiv(N, BN))

        _base_lora_kernel[grid](
            x,
            qweight,
            scales,
            mid,
            B,
            lora_ids,
            out,
            T,
            D,
            N,
            R,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_SIZE=GROUP,
            num_warps=4,
            num_stages=4,
        )

        return out