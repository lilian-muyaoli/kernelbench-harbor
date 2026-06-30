import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def moe_kernel(
    x_ptr, w_ptr, out_ptr,
    topi_ptr, topv_ptr,
    T, D, H, E,
    stride_xt, stride_xd,
    stride_we, stride_wd, stride_wh,
    stride_ot, stride_oh,
    stride_tit, stride_tik,
    stride_tvt, stride_tvk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kexp in range(0, 2):
        # gate values and expert indices for this token block
        ti = tl.load(topi_ptr + offs_m * stride_tit + kexp * stride_tik,
                     mask=offs_m < T, other=0)
        tv = tl.load(topv_ptr + offs_m * stride_tvt + kexp * stride_tvk,
                     mask=offs_m < T, other=0.0).to(tl.float32)

        # Compute matmul x[m,:] @ w[expert(m), :, n]
        # Since experts differ per row, loop over distinct experts present.
        for e in range(0, E):
            mask_e = ti == e
            partial = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, D, BLOCK_K):
                k = k0 + offs_k
                x_blk = tl.load(
                    x_ptr + offs_m[:, None] * stride_xt + k[None, :] * stride_xd,
                    mask=(offs_m[:, None] < T) & (k[None, :] < D), other=0.0
                ).to(tl.float32)
                w_blk = tl.load(
                    w_ptr + e * stride_we + k[:, None] * stride_wd + offs_n[None, :] * stride_wh,
                    mask=(k[:, None] < D) & (offs_n[None, :] < H), other=0.0
                ).to(tl.float32)
                partial += tl.dot(x_blk, w_blk)
            sel = mask_e[:, None].to(tl.float32) * tv[:, None]
            acc += partial * sel

    tl.store(
        out_ptr + offs_m[:, None] * stride_ot + offs_n[None, :] * stride_oh,
        acc,
        mask=(offs_m[:, None] < T) & (offs_n[None, :] < H)
    )


class ModelNew(nn.Module):
    def forward(self, x, w, router_logits):
        T, D = x.shape
        E = w.shape[0]
        H = w.shape[2]

        probs = torch.softmax(router_logits.float(), dim=-1)
        topv, topi = probs.topk(2, dim=-1)
        topv = (topv / topv.sum(dim=-1, keepdim=True)).to(torch.float32)
        topi = topi.to(torch.int32).contiguous()
        topv = topv.contiguous()

        x = x.contiguous()
        w = w.contiguous()
        out = torch.empty((T, H), device=x.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (triton.cdiv(T, BLOCK_M), triton.cdiv(H, BLOCK_N))
        moe_kernel[grid](
            x, w, out,
            topi, topv,
            T, D, H, E,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1), w.stride(2),
            out.stride(0), out.stride(1),
            topi.stride(0), topi.stride(1),
            topv.stride(0), topv.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        return out.to(x.dtype)