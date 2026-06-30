# Optimize a GPU kernel

You are given a naive PyTorch reference module `Model` (in `reference.py`). Write an
optimized replacement `ModelNew` with the **same `forward` signature and outputs**.

## CRITICAL — how to submit
Your FINAL action MUST write the complete code (imports + Triton kernel + `class ModelNew`)
to **`/app/solution.py`**. Use a single shell command, e.g.:
```
cat > /app/solution.py << 'EOF'
<your full python code here>
EOF
```
The grader reads ONLY `/app/solution.py`. If you do not write this file, you score 0.

## Rules (your submission is rejected otherwise)
- `ModelNew()` must construct with **no arguments**; same `forward(...)` signature as `Model`.
- `ModelNew.forward` must compute the core work with at least one real `@triton.jit` kernel.
- Do NOT offload core compute to torch (`torch.matmul`, `torch.softmax`, `einsum`, `F.linear`, …)
  or vendor libs (`cublas`, `cudnn`, `flash_attn`, `scaled_dot_product_attention`).
- Must be numerically correct vs the reference (fp32 ground truth, dtype-aware tolerance).
- You may use torch only for tensor allocation, shape handling, and launching the kernel.

## You pass iff
correct **AND** faster than `torch.compile(Model, mode="max-autotune")` by the configured
threshold. Beating the compiler is the goal — a correct-but-slow kernel does not pass.

## The reference
```python
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
    T, H = 512, 14336
    y = torch.randn(T, 2 * H, device="cuda", dtype=torch.bfloat16)
    return [y]

```
