import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _zero_counts_kernel(counts, E: tl.constexpr, BLOCK_E: tl.constexpr):
    offs = tl.arange(0, BLOCK_E)
    tl.store(counts + offs, tl.zeros((BLOCK_E,), tl.int32), mask=offs < E)


@triton.jit
def _zero_output_kernel(y, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + offs, tl.zeros((BLOCK,), tl.float32), mask=offs < N)


@triton.jit
def _route_top2_kernel(
    router,
    counts,
    tokens,
    gates,
    T: tl.constexpr,
    E: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    offs_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    neg_inf = -3.4028234663852886e38
    max1 = tl.full((BLOCK_T,), neg_inf, tl.float32)
    max2 = tl.full((BLOCK_T,), neg_inf, tl.float32)
    idx1 = tl.full((BLOCK_T,), 0, tl.int32)
    idx2 = tl.full((BLOCK_T,), 0, tl.int32)

    for e in range(0, E):
        v = tl.load(router + offs_t * E + e, mask=mask_t, other=neg_inf).to(tl.float32)

        gt1 = v > max1
        gt2 = (v > max2) & (~gt1)

        old_max1 = max1
        old_idx1 = idx1

        max1 = tl.where(gt1, v, max1)
        idx1 = tl.where(gt1, e, idx1)

        max2 = tl.where(gt1, old_max1, tl.where(gt2, v, max2))
        idx2 = tl.where(gt1, old_idx1, tl.where(gt2, e, idx2))

    g1 = 1.0 / (1.0 + tl.exp(max2 - max1))
    g2 = 1.0 - g1

    pos1 = tl.atomic_add(counts + idx1, 1, mask=mask_t, sem="relaxed")
    tl.store(tokens + idx1 * CAP + pos1, offs_t, mask=mask_t)
    tl.store(gates + idx1 * CAP + pos1, g1, mask=mask_t)

    pos2 = tl.atomic_add(counts + idx2, 1, mask=mask_t, sem="relaxed")
    tl.store(tokens + idx2 * CAP + pos2, offs_t, mask=mask_t)
    tl.store(gates + idx2 * CAP + pos2, g2, mask=mask_t)


@triton.jit
def _moe_grouped_matmul_kernel(
    x,
    w,
    y,
    counts,
    tokens,
    gates,
    D: tl.constexpr,
    H: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_h = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    cnt = tl.load(counts + e)
    if m_start >= cnt:
        return

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_m = offs_m < cnt

    tok = tl.load(tokens + e * CAP + offs_m, mask=valid_m, other=0)
    gate = tl.load(gates + e * CAP + offs_m, mask=valid_m, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)

        a = tl.load(
            x + tok[:, None] * D + offs_d[None, :],
            mask=valid_m[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )

        b = tl.load(
            w + e * D * H + offs_d[:, None] * H + offs_h[None, :],
            mask=(offs_d[:, None] < D) & (offs_h[None, :] < H),
            other=0.0,
        )

        acc += tl.dot(a, b)

    acc = acc.to(tl.float16).to(tl.float32)
    contrib = acc * gate[:, None]

    tl.atomic_add(
        y + tok[:, None] * H + offs_h[None, :],
        contrib,
        mask=valid_m[:, None] & (offs_h[None, :] < H),
        sem="relaxed",
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w, router_logits):
        T = x.shape[0]
        D = x.shape[1]
        E = w.shape[0]
        H = w.shape[2]

        y = torch.empty((T, H), device=x.device, dtype=x.dtype)
        counts = torch.empty((E,), device=x.device, dtype=torch.int32)

        cap = T
        tokens = torch.empty((E, cap), device=x.device, dtype=torch.int32)
        gates = torch.empty((E, cap), device=x.device, dtype=x.dtype)

        _zero_counts_kernel[(1,)](
            counts,
            E,
            BLOCK_E=triton.next_power_of_2(E),
            num_warps=1,
        )

        _zero_output_kernel[(triton.cdiv(T * H, 1024),)](
            y,
            T * H,
            BLOCK=1024,
            num_warps=4,
        )

        _route_top2_kernel[(triton.cdiv(T, 128),)](
            router_logits,
            counts,
            tokens,
            gates,
            T,
            E,
            cap,
            BLOCK_T=128,
            num_warps=4,
        )

        _moe_grouped_matmul_kernel[
            (E, triton.cdiv(cap, 32), triton.cdiv(H, 64))
        ](
            x,
            w,
            y,
            counts,
            tokens,
            gates,
            D,
            H,
            cap,
            BLOCK_M=32,
            BLOCK_N=64,
            BLOCK_D=64,
            num_warps=4,
            num_stages=3,
        )

        return y