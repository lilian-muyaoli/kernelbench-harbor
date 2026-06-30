"""Oracle for fused_silu_fp8 — adapted from vLLM's batched_deep_gemm_moe
`_silu_mul_fp8_quant_deep_gemm` kernel (single-expert 2D form, returns dequant q*scale).

Fuses silu(gate)*up + per-group(128) dynamic fp8(e4m3) quant + dequant in one pass.

NOTE: the output is fp8-quantized, a step function — comparing it bit-exactly to torch's
fp8 rounding is ill-posed (sub-ULP differences in silu/exp flip quant codes -> ~0.1 abs
error even with torch's own cast). This family is therefore graded with a cosine/relative
metric (how fp8 kernels are validated in practice), not allclose. This oracle is correct
under that metric and beats torch.compile.
"""
import torch
import triton
import triton.language as tl

GROUP = 128
FP8_MAX = 448.0
FP8_MIN = -448.0


@triton.jit
def _silu_mul_fp8_quant(y_ptr, out_ptr, H, GROUP_SIZE: tl.constexpr,
                        fp8_min, fp8_max, eps):
    t = tl.program_id(0)
    g = tl.program_id(1)
    cols = tl.arange(0, GROUP_SIZE)
    row = t * (2 * H)
    gate = tl.load(y_ptr + row + g * GROUP_SIZE + cols).to(tl.float32)
    up = tl.load(y_ptr + row + H + g * GROUP_SIZE + cols).to(tl.float32)
    act = (gate * (1.0 / (1.0 + tl.exp(-gate)))) * up         # silu(gate) * up
    s = tl.maximum(tl.max(tl.abs(act)), eps) / fp8_max         # per-group dynamic scale
    q = tl.clamp(act / s, fp8_min, fp8_max).to(tl.float8e4nv)  # quantize to e4m3
    deq = q.to(tl.float32) * s                                 # dequant
    tl.store(out_ptr + t * H + g * GROUP_SIZE + cols, deq)


class ModelNew(torch.nn.Module):
    def forward(self, y):
        T, H2 = y.shape
        H = H2 // 2
        G = H // GROUP
        out = torch.empty((T, H), device=y.device, dtype=torch.float32)
        _silu_mul_fp8_quant[(T, G)](y, out, H, GROUP, FP8_MIN, FP8_MAX, 1e-12)
        return out
