import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK = 32


@triton.jit
def _fp8_scaled_matmul_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    K,
    sxm,
    sxk,
    swk,
    swn,
    ssb,
    ssn,
    sym,
    syn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        x_vals = tl.load(
            x_ptr + offs_m[:, None] * sxm + k_idxs[None, :] * sxk,
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        ).to(tl.float16)

        w8_vals = tl.load(
            w_ptr + k_idxs[:, None] * swk + offs_n[None, :] * swn,
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        sc_vals = tl.load(
            s_ptr + (k0 // BLOCK_K) * ssb + offs_n * ssn,
            mask=offs_n < N,
            other=0.0,
        ).to(tl.float32)

        w_vals = (w8_vals.to(tl.float32) * sc_vals[None, :]).to(tl.float16)
        acc += tl.dot(x_vals, w_vals, out_dtype=tl.float32)

    tl.store(
        y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w_fp8, scales):
        M = x.shape[0]
        K = x.shape[1]
        N = w_fp8.shape[1]

        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BM = 16
        BN = 32
        BK = 32

        grid = (triton.cdiv(N, BN), triton.cdiv(M, BM))

        _fp8_scaled_matmul_kernel[grid](
            x,
            w_fp8,
            scales,
            y,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w_fp8.stride(0),
            w_fp8.stride(1),
            scales.stride(0),
            scales.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            num_warps=4,
            num_stages=3,
        )
        return y