import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 128
FP8_MAX = 448.0  # e4m3fn max

class Model(nn.Module):
    """Fused SiLU + multiply + per-group dynamic FP8 quant (vLLM batched_deep_gemm_moe).
    y:[T,2H]=concat(gate,up). act=silu(gate)*up; per-group (GROUP along H) dynamic fp8:
    scale=amax(|act|)/FP8_MAX; q=round_fp8(clamp(act/scale)). Returns dequant q*scale.
    A fast kernel fuses silu+mul+reduction+quant in one pass."""
    def forward(self, y):
        T, H2 = y.shape
        H = H2 // 2
        gate, up = y[:, :H], y[:, H:]
        act = (F.silu(gate.float()) * up.float())
        G = H // GROUP
        a = act.reshape(T, G, GROUP)
        scale = (a.abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
        q = (a / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
        return (q * scale).reshape(T, H)

def get_inputs():
    T, H = 512, 1408
    y = torch.randn(T, 2 * H, device="cuda", dtype=torch.float16)
    return [y]
