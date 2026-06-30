import torch
import triton
import triton.language as tl


@triton.jit
def mamba_scan_kernel(
    u_ptr, delta_ptr, A_ptr, Bm_ptr, C_ptr, y_ptr,
    B, L, D, S,
    stride_ub, stride_ul, stride_ud,
    stride_ab, stride_as,
    stride_bb, stride_bl, stride_bs,
    BLOCK_S: tl.constexpr,
):
    # one program per (batch, channel d)
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D

    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    # load A[d, :]
    A_row = tl.load(A_ptr + d * stride_as + offs_s, mask=mask_s, other=0.0).to(tl.float32)

    h = tl.zeros([BLOCK_S], dtype=tl.float32)

    for t in range(L):
        u_off = b * stride_ub + t * stride_ul + d * stride_ud
        u_val = tl.load(u_ptr + u_off).to(tl.float32)
        delta_val = tl.load(delta_ptr + u_off).to(tl.float32)

        bc_off = b * stride_bb + t * stride_bl
        Bm_val = tl.load(Bm_ptr + bc_off + offs_s, mask=mask_s, other=0.0).to(tl.float32)
        C_val = tl.load(C_ptr + bc_off + offs_s, mask=mask_s, other=0.0).to(tl.float32)

        dA = tl.exp(delta_val * A_row)
        dBu = (delta_val * Bm_val) * u_val
        h = dA * h + dBu
        y = tl.sum(tl.where(mask_s, h * C_val, 0.0), axis=0)

        tl.store(y_ptr + u_off, y)


class ModelNew(torch.nn.Module):
    def forward(self, u, delta, A, Bm, C):
        Bb, L, D = u.shape
        S = A.shape[1]

        u_f = u.contiguous()
        delta_f = delta.contiguous()
        A_f = A.contiguous()
        Bm_f = Bm.contiguous()
        C_f = C.contiguous()

        y = torch.empty((Bb, L, D), device=u.device, dtype=torch.float32)

        BLOCK_S = triton.next_power_of_2(S)

        grid = (Bb * D,)
        mamba_scan_kernel[grid](
            u_f, delta_f, A_f, Bm_f, C_f, y,
            Bb, L, D, S,
            u_f.stride(0), u_f.stride(1), u_f.stride(2),
            A_f.stride(0), A_f.stride(1),
            Bm_f.stride(0), Bm_f.stride(1), Bm_f.stride(2),
            BLOCK_S=BLOCK_S,
        )

        return y.to(u.dtype)