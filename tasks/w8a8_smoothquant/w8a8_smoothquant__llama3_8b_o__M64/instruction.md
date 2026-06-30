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

class Model(nn.Module):
    """W8A8 SmoothQuant linear. x_int8:[M,K] int8; w_int8:[K,N] int8;
    x_scale:[M,1] fp16 (per-token); w_scale:[1,N] fp16 (per-channel). out fp16."""
    def forward(self, x_int8, w_int8, x_scale, w_scale):
        acc = torch.matmul(x_int8.float(), w_int8.float())
        return (acc * x_scale.float() * w_scale.float()).to(torch.float16)

def get_inputs():
    M, K, N = 64, 4096, 4096
    x_int8 = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    w_int8 = torch.randint(-127, 128, (K, N), device="cuda", dtype=torch.int8)
    x_scale = (torch.rand(M, 1, device="cuda") * 0.01 + 0.001).to(torch.float16)
    w_scale = (torch.rand(1, N, device="cuda") * 0.01 + 0.001).to(torch.float16)
    return [x_int8, w_int8, x_scale, w_scale]

```
