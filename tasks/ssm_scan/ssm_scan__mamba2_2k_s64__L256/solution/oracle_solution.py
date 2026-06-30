import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _s6_scan_fwd_kernel(
    u_ptr,
    delta_ptr,
    A_ptr,
    Bm_ptr,
    C_ptr,
    y_ptr,
    L: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_db = tl.program_id(1)

    offs_d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, S)

    mask_d = offs_d < D
    mask_ds = mask_d[:, None]

    a = tl.load(
        A_ptr + offs_d[:, None] * S + offs_s[None, :],
        mask=mask_ds,
        other=0.0,
    ).to(tl.float32)

    h = tl.zeros((BLOCK_D, S), dtype=tl.float32)

    for t in tl.range(0, L, 1):
        base_bd = (pid_b * L + t) * D
        base_bs = (pid_b * L + t) * S

        u_v = tl.load(u_ptr + base_bd + offs_d, mask=mask_d, other=0.0)
        delta_v = tl.load(delta_ptr + base_bd + offs_d, mask=mask_d, other=0.0)
        bm_v = tl.load(Bm_ptr + base_bs + offs_s)
        c_v = tl.load(C_ptr + base_bs + offs_s)

        u_f = u_v.to(tl.float32)
        delta_f = delta_v.to(tl.float32)
        bm_f = bm_v.to(tl.float32)
        c_f = c_v.to(tl.float32)

        x = (delta_f[:, None] * a).to(tl.float16).to(tl.float32)
        dA = tl.exp(x).to(tl.float16).to(tl.float32)

        tmp = (delta_f[:, None] * bm_f[None, :]).to(tl.float16).to(tl.float32)
        dBu = (tmp * u_f[:, None]).to(tl.float16).to(tl.float32)

        h = dA * h + dBu
        y = tl.sum(h * c_f[None, :], axis=1)

        tl.store(y_ptr + base_bd + offs_d, y, mask=mask_d)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u, delta, A, Bm, C):
        u = u.contiguous()
        delta = delta.contiguous()
        A = A.contiguous()
        Bm = Bm.contiguous()
        C = C.contiguous()

        B, L, D = u.shape
        S = A.shape[1]
        y = torch.empty_like(u)

        block_d = 16
        grid = (B, triton.cdiv(D, block_d))

        _s6_scan_fwd_kernel[grid](
            u,
            delta,
            A,
            Bm,
            C,
            y,
            L,
            D,
            S,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=3,
        )
        return y