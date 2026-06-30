import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def moe_kernel(
    x_ptr, w_ptr, topi_ptr, topv_ptr, out_ptr,
    T, D, E, H,
    stride_xt, stride_xd,
    stride_we, stride_wd, stride_wh,
    stride_ti, stride_tk,
    stride_vi, stride_vk,
    stride_ot, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < T
    n_mask = offs_n < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over top-2 experts
    for kk in range(2):
        # gather expert index and gate per row
        e_idx = tl.load(topi_ptr + offs_m * stride_ti + kk * stride_tk, mask=m_mask, other=0)
        gate = tl.load(topv_ptr + offs_m * stride_vi + kk * stride_vk, mask=m_mask, other=0.0)
        gate = gate.to(tl.float32)

        partial = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for d0 in range(0, D, BLOCK_K):
            k = d0 + offs_k
            k_mask = k < D
            # x block: [BLOCK_M, BLOCK_K]
            x_ptrs = x_ptr + offs_m[:, None] * stride_xt + k[None, :] * stride_xd
            x_blk = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            # for each row's expert, we need w[e_idx[m], k, n]
            # build per-row weight pointers: shape [BLOCK_M, BLOCK_K, BLOCK_N] is too big.
            # Instead accumulate via matmul per distinct trick is hard; do it directly:
            # We compute partial[m,n] += sum_k x[m,k]*w[e_idx[m],k,n]
            # Use elementwise: load w with expert per-row -> [BLOCK_M, BLOCK_K, BLOCK_N] not feasible.
            # Do a k-loop matmul where weight depends on row -> approximate with broadcasting per n-block.
            w_base = w_ptr + e_idx[:, None, None] * stride_we \
                     + k[None, :, None] * stride_wd \
                     + offs_n[None, None, :] * stride_wh
            w_blk = tl.load(
                w_base,
                mask=m_mask[:, None, None] & k_mask[None, :, None] & n_mask[None, None, :],
                other=0.0,
            )
            partial += tl.sum(x_blk[:, :, None] * w_blk.to(tl.float32), axis=1)

        acc += partial * gate[:, None]

    out_ptrs = out_ptr + offs_m[:, None] * stride_ot + offs_n[None, :] * stride_oh
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(nn.Module):
    def forward(self, x, w, router_logits):
        T, D = x.shape
        E, _, H = w.shape

        probs = torch.softmax(router_logits.float(), dim=-1)
        topv, topi = probs.topk(2, dim=-1)
        topv = (topv / topv.sum(dim=-1, keepdim=True)).to(x.dtype)

        topi = topi.to(torch.int32).contiguous()
        topv = topv.contiguous()
        x = x.contiguous()
        w = w.contiguous()

        out = torch.empty((T, H), device=x.device, dtype=torch.float32)

        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(T, BLOCK_M), triton.cdiv(H, BLOCK_N))

        moe_kernel[grid](
            x, w, topi, topv, out,
            T, D, E, H,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1), w.stride(2),
            topi.stride(0), topi.stride(1),
            topv.stride(0), topv.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        return out.to(x.dtype)