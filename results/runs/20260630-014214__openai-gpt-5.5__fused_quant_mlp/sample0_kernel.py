import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _awq_mlp_first_kernel(
    x_ptr,
    qw_ptr,
    s_ptr,
    h_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_T, BLOCK_N), tl.float32)

    for k0 in range(0, D, BLOCK_K):
        k = k0 + offs_k

        x = tl.load(
            x_ptr + offs_t[:, None] * D + k[None, :],
            mask=(offs_t[:, None] < T) & (k[None, :] < D),
            other=0.0,
        )

        q = tl.load(
            qw_ptr + k[:, None] * H + offs_n[None, :],
            mask=(k[:, None] < D) & (offs_n[None, :] < H),
            other=0,
        )

        scale = tl.load(
            s_ptr + (k0 // GROUP_SIZE) * H + offs_n,
            mask=offs_n < H,
            other=0.0,
        )

        w = (q.to(tl.float32) * scale[None, :].to(tl.float32)).to(tl.float16)
        acc += tl.dot(x, w)

    a = acc.to(tl.float16).to(tl.float32)
    silu = a / (1.0 + tl.exp(-a))

    tl.store(
        h_ptr + offs_t[:, None] * H + offs_n[None, :],
        silu.to(tl.float16),
        mask=(offs_t[:, None] < T) & (offs_n[None, :] < H),
    )


@triton.jit
def _awq_mlp_second_kernel(
    h_ptr,
    qw_ptr,
    s_ptr,
    y_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_T, BLOCK_N), tl.float32)

    for k0 in range(0, H, BLOCK_K):
        k = k0 + offs_k

        h = tl.load(
            h_ptr + offs_t[:, None] * H + k[None, :],
            mask=(offs_t[:, None] < T) & (k[None, :] < H),
            other=0.0,
        )

        q = tl.load(
            qw_ptr + k[:, None] * D + offs_n[None, :],
            mask=(k[:, None] < H) & (offs_n[None, :] < D),
            other=0,
        )

        scale = tl.load(
            s_ptr + (k0 // GROUP_SIZE) * D + offs_n,
            mask=offs_n < D,
            other=0.0,
        )

        w = (q.to(tl.float32) * scale[None, :].to(tl.float32)).to(tl.float16)
        acc += tl.dot(h, w)

    tl.store(
        y_ptr + offs_t[:, None] * D + offs_n[None, :],
        acc.to(tl.float16),
        mask=(offs_t[:, None] < T) & (offs_n[None, :] < D),
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, qw1, s1, qw2, s2):
        if not x.is_contiguous():
            x = x.contiguous()
        if not qw1.is_contiguous():
            qw1 = qw1.contiguous()
        if not s1.is_contiguous():
            s1 = s1.contiguous()
        if not qw2.is_contiguous():
            qw2 = qw2.contiguous()
        if not s2.is_contiguous():
            s2 = s2.contiguous()

        T = x.shape[0]
        D = x.shape[1]
        H = qw1.shape[1]

        h = torch.empty((T, H), device=x.device, dtype=x.dtype)
        y = torch.empty((T, D), device=x.device, dtype=x.dtype)

        BLOCK_T = 16
        BLOCK_N = 32
        BLOCK_K = 128

        grid1 = (triton.cdiv(H, BLOCK_N), triton.cdiv(T, BLOCK_T))
        _awq_mlp_first_kernel[grid1](
            x,
            qw1,
            s1,
            h,
            T,
            D,
            H,
            BLOCK_T=BLOCK_T,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=GROUP,
            num_warps=4,
            num_stages=3,
        )

        grid2 = (triton.cdiv(D, BLOCK_N), triton.cdiv(T, BLOCK_T))
        _awq_mlp_second_kernel[grid2](
            h,
            qw2,
            s2,
            y,
            T,
            H,
            D,
            BLOCK_T=BLOCK_T,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE=GROUP,
            num_warps=4,
            num_stages=3,
        )

        return y