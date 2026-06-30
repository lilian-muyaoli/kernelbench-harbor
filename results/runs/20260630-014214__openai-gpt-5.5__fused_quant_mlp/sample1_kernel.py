import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _awq_fc1_silu_kernel(
    x_ptr, qw_ptr, s_ptr, h_ptr,
    T: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in tl.range(0, D, BLOCK_K):
        k_idxs = k0 + offs_k

        a = tl.load(
            x_ptr + offs_m[:, None] * D + k_idxs[None, :],
            mask=(offs_m[:, None] < T) & (k_idxs[None, :] < D),
            other=0.0,
        )

        q = tl.load(
            qw_ptr + k_idxs[:, None] * H + offs_n[None, :],
            mask=(k_idxs[:, None] < D) & (offs_n[None, :] < H),
            other=0,
        )

        sc = tl.load(
            s_ptr + (k0 // GROUP_SIZE) * H + offs_n,
            mask=offs_n < H,
            other=0.0,
        )

        b = (q.to(tl.float32) * sc[None, :].to(tl.float32)).to(tl.float16)
        acc = tl.dot(a, b, acc)

    z = acc.to(tl.float16).to(tl.float32)
    out = z / (1.0 + tl.exp(-z))

    tl.store(
        h_ptr + offs_m[:, None] * H + offs_n[None, :],
        out.to(tl.float16),
        mask=(offs_m[:, None] < T) & (offs_n[None, :] < H),
    )


@triton.jit
def _awq_fc2_kernel(
    h_ptr, qw_ptr, s_ptr, y_ptr,
    T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in tl.range(0, H, BLOCK_K):
        k_idxs = k0 + offs_k

        a = tl.load(
            h_ptr + offs_m[:, None] * H + k_idxs[None, :],
            mask=(offs_m[:, None] < T) & (k_idxs[None, :] < H),
            other=0.0,
        )

        q = tl.load(
            qw_ptr + k_idxs[:, None] * D + offs_n[None, :],
            mask=(k_idxs[:, None] < H) & (offs_n[None, :] < D),
            other=0,
        )

        sc = tl.load(
            s_ptr + (k0 // GROUP_SIZE) * D + offs_n,
            mask=offs_n < D,
            other=0.0,
        )

        b = (q.to(tl.float32) * sc[None, :].to(tl.float32)).to(tl.float16)
        acc = tl.dot(a, b, acc)

    tl.store(
        y_ptr + offs_m[:, None] * D + offs_n[None, :],
        acc.to(tl.float16),
        mask=(offs_m[:, None] < T) & (offs_n[None, :] < D),
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, qw1, s1, qw2, s2):
        T = x.shape[0]
        D = qw1.shape[0]
        H = qw1.shape[1]

        h = torch.empty((T, H), device=x.device, dtype=x.dtype)
        y = torch.empty((T, D), device=x.device, dtype=x.dtype)

        BM = 16
        BN = 64
        BK = GROUP

        grid1 = (triton.cdiv(T, BM), triton.cdiv(H, BN))
        _awq_fc1_silu_kernel[grid1](
            x, qw1, s1, h,
            T, D, H,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_SIZE=GROUP,
            num_warps=4,
            num_stages=4,
        )

        grid2 = (triton.cdiv(T, BM), triton.cdiv(D, BN))
        _awq_fc2_kernel[grid2](
            h, qw2, s2, y,
            T, H, D,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_SIZE=GROUP,
            num_warps=4,
            num_stages=4,
        )

        return y