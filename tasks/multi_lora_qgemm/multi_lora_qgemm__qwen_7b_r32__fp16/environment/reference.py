import torch
import torch.nn as nn

GROUP = 128

class Model(nn.Module):
    """Multi-LoRA quantized linear (batched adapters). x:[T,D]; qweight:[D,N] int8 (int4 base);
    scales:[D//GROUP,N] fp16; A:[L,D,r], B:[L,r,N]; lora_ids:[T].
    y[t] = x[t]@dequant(W) + (x[t]@A[lora_ids[t]])@B[lora_ids[t]] (fused, per-token adapter gather)."""
    def forward(self, x, qweight, scales, A, B, lora_ids):
        D, N = qweight.shape
        w = (qweight.reshape(D // GROUP, GROUP, N).to(x.dtype) * scales[:, None, :].to(x.dtype)).reshape(D, N)
        base = x @ w
        Ai, Bi = A[lora_ids], B[lora_ids]
        lo = torch.einsum('td,tdr->tr', x, Ai)
        lo = torch.einsum('tr,trn->tn', lo, Bi)
        return base + lo

def get_inputs():
    T, D, N, r, L = 64, 3584, 3584, 32, 8
    x = torch.randn(T, D, device="cuda", dtype=torch.float16)
    qweight = torch.randint(-8, 8, (D, N), device="cuda", dtype=torch.int8)
    scales = (torch.randn(D // GROUP, N, device="cuda") * 0.01).to(torch.float16)
    A = (torch.randn(L, D, r, device="cuda", dtype=torch.float16) * 0.02)
    B = (torch.randn(L, r, N, device="cuda", dtype=torch.float16) * 0.02)
    lora_ids = torch.randint(0, L, (T,), device="cuda")
    return [x, qweight, scales, A, B, lora_ids]
