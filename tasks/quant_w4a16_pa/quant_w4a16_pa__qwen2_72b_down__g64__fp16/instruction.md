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

GROUP = 64

class Model(nn.Module):
    """Packed asymmetric W4A16 (GPTQ-style) linear, decode shape.
    x:[M,K] float16; qweight:[K,N//8] int32 (8 uint4 nibbles each; nibble i -> col 8*j+i);
    scales,zeros:[K//GROUP,N] float16. w[k,n]=(unpack(qweight)[k,n]-zeros[k//GROUP,n])*scales[k//GROUP,n]."""
    def forward(self, x, qweight, scales, zeros):
        K = qweight.shape[0]; N = qweight.shape[1] * 8
        shifts = torch.arange(0, 32, 4, device=x.device, dtype=torch.int32)
        w = ((qweight.unsqueeze(-1) >> shifts) & 0xF).reshape(K, N).to(x.dtype)
        g = K // GROUP
        se = scales.reshape(g, 1, N).expand(g, GROUP, N).reshape(K, N).to(x.dtype)
        ze = zeros.reshape(g, 1, N).expand(g, GROUP, N).reshape(K, N).to(x.dtype)
        return torch.matmul(x, (w - ze) * se)

def get_inputs():
    M, K, N = 16, 29568, 8192
    nib = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int64)
    packed = torch.zeros(K, N // 8, device="cuda", dtype=torch.int64)
    for i in range(8):
        packed |= (nib[:, i::8] << (4 * i))
    qweight = packed.to(torch.int32)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    scales = (torch.randn(K // GROUP, N, device="cuda") * 0.01).to(torch.float16)
    zeros = torch.randint(0, 16, (K // GROUP, N), device="cuda").to(torch.float16)
    return [x, qweight, scales, zeros]

```
