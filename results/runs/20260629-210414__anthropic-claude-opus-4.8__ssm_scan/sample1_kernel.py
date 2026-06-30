import torch
import triton
import triton.language as tl


@triton.jit
def _scan_kernel(
    u_ptr, delta_ptr, A_ptr, Bm_ptr, C_ptr, y_ptr,
    B, L, D, S,
    stride_ub, stride_ul, stride_ud,
    stride_ad, stride_as,
    stride_bb, stride_bl, stride_bs,
    BLOCK_S: tl.constexpr,
):
    # one program per (b, d)
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D

    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    # load A row [S]
    a = tl.load(A_ptr + d * stride_ad + offs_s * stride_as, mask=mask_s, other=0.0).to(tl.float32)

    h = tl.zeros([BLOCK_S], dtype=tl.float32)

    u_base = u_ptr + b * stride_ub + d * stride_ud
    delta_base = delta_ptr + b * stride_ub + d * stride_ud
    bm_base = Bm_ptr + b * stride_bb
    c_base = C_ptr + b * stride_bb
    y_base = y_ptr + b * stride_ub + d * stride_ud

    for t in range(L):
        u_t = tl.load(u_base + t * stride_ul).to(tl.float32)
        delta_t = tl.load(delta_base + t * stride_ul).to(tl.float32)
        bm_t = tl.load(bm_base + t * stride_bl + offs_s * stride_bs, mask=mask_s, other=0.0).to(tl.float32)
        c_t = tl.load(c_base + t * stride_bl + offs_s * stride_bs, mask=mask_s, other=0.0).to(tl.float32)

        dA = tl.exp(delta_t * a)
        dBu = (delta_t * bm_t) * u_t
        h = dA * h + dBu
        y = tl.sum(tl.where(mask_s, h * c_t, 0.0), axis=0)
        tl.store(y_base + t * stride_ul, y)


class ModelNew(torch.nn.Module):
    def forward(self, u, delta, A, Bm, C):
        Bb, L, D = u.shape
        S = A.shape[1]

        u_f = u.float().contiguous()
        delta_f = delta.float().contiguous()
        A_f = A.float().contiguous()
        Bm_f = Bm.float().contiguous()
        C_f = C.float().contiguous()

        y = torch.empty((Bb, L, D), device=u.device, dtype=torch.float32)

        BLOCK_S = triton.next_power_of_2(S)

        grid = (Bb * D,)
        _scan_kernel[grid](
            u_f, delta_f, A_f, Bm_f, C_f, y,
            Bb, L, D, S,
            u_f.stride(0), u_f.stride(1), u_f.stride(2),
            A_f.stride(0), A_f.stride(1),
            Bm_f.stride(0), Bm_f.stride(1), Bm_f.stride(2),
            BLOCK_S=BLOCK_S,
        )

        return y.to(u.dtype)