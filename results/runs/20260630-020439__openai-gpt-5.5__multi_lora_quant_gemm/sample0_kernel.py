import torch
import torch.nn as nn
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def _lora_a_kernel(
    X, A, LORA_IDS, LOW,
    D: tl.constexpr, R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    rr = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    lid = tl.load(LORA_IDS + t)

    acc = tl.zeros((), tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + offs_d
        x = tl.load(X + t * D + d, mask=d < D, other=0.0).to(tl.float32)
        a = tl.load(A + lid * D * R + d * R + rr, mask=d < D, other=0.0).to(tl.float32)
        acc += tl.sum(x * a, axis=0)

    tl.store(LOW + t * R + rr, acc)


@triton.jit
def _base_lora_kernel(
    X, QW, SCALES, B, LORA_IDS, LOW, Y,
    T: tl.constexpr, D: tl.constexpr, N: tl.constexpr, R: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, D, BLOCK_K):
        k = k0 + offs_k

        x = tl.load(
            X + offs_m[:, None] * D + k[None, :],
            mask=(offs_m[:, None] < T) & (k[None, :] < D),
            other=0.0,
        )

        q = tl.load(
            QW + k[:, None] * N + offs_n[None, :],
            mask=(k[:, None] < D) & (offs_n[None, :] < N),
            other=0,
        )

        g = k0 // GROUP
        sc = tl.load(SCALES + g * N + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        w = (q.to(tl.float32) * sc[None, :]).to(tl.float16)

        acc += tl.dot(x, w, out_dtype=tl.float32)

    lids = tl.load(LORA_IDS + offs_m, mask=offs_m < T, other=0)
    lacc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for rr in range(0, R):
        low = tl.load(LOW + offs_m * R + rr, mask=offs_m < T, other=0.0).to(tl.float32)
        b = tl.load(
            B + lids[:, None] * R * N + rr * N + offs_n[None, :],
            mask=(offs_m[:, None] < T) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        lacc += low[:, None] * b

    out = acc + lacc
    tl.store(
        Y + offs_m[:, None] * N + offs_n[None, :],
        out,
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

        low = torch.empty((T, R), device=x.device, dtype=x.dtype)
        y = torch.empty((T, N), device=x.device, dtype=x.dtype)

        _lora_a_kernel[(T, R)](
            x, A, lora_ids, low,
            D, R,
            BLOCK_D=256,
            num_warps=8,
        )

        grid = (triton.cdiv(T, 16), triton.cdiv(N, 32))
        _base_lora_kernel[grid](
            x, qweight, scales, B, lora_ids, low, y,
            T, D, N, R,
            BLOCK_M=16,
            BLOCK_N=32,
            BLOCK_K=128,
            num_warps=4,
            num_stages=4,
        )

        return y