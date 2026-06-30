import torch
import triton
import triton.language as tl

GROUP = 128

@triton.jit
def w4a16_kernel(
    x_ptr, qw_ptr, scales_ptr, zeros_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_sg, stride_sn,
    stride_zg, stride_zn,
    stride_om, stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # column index within packed int32: n//8 ; nibble index: n%8
    qw_col = offs_n // 8
    nib_idx = offs_n % 8
    shift = (nib_idx * 4).to(tl.int32)

    m_mask = offs_m < M
    n_mask = offs_n < N

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < K

        # x block [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk
        x_blk = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # qweight block [BLOCK_K, BLOCK_N]
        qw_ptrs = qw_ptr + k[:, None] * stride_qwk + qw_col[None, :] * stride_qwn
        qw_blk = tl.load(qw_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        w_int = (qw_blk >> shift[None, :]) & 0xF

        # group index per k
        g = k // GROUP_SIZE
        s_ptrs = scales_ptr + g[:, None] * stride_sg + offs_n[None, :] * stride_sn
        z_ptrs = zeros_ptr + g[:, None] * stride_zg + offs_n[None, :] * stride_zn
        s_blk = tl.load(s_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
        z_blk = tl.load(z_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        w_blk = (w_int.to(tl.float32) - z_blk) * s_blk

        acc += tl.dot(x_blk, w_blk)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, x, qweight, scales, zeros):
        K = qweight.shape[0]
        N = qweight.shape[1] * 8
        M = x.shape[0]

        x = x.contiguous()
        qweight = qweight.contiguous()
        scales = scales.contiguous()
        zeros = zeros.contiguous()

        out = torch.empty((M, N), device=x.device, dtype=torch.float16)

        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        w4a16_kernel[grid](
            x, qweight, scales, zeros, out,
            M, N, K,
            x.stride(0), x.stride(1),
            qweight.stride(0), qweight.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            out.stride(0), out.stride(1),
            GROUP_SIZE=GROUP,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return out