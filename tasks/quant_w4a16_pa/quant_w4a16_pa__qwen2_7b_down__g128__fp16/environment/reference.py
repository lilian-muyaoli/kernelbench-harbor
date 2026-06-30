import torch
import torch.nn as nn

GROUP = 128

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
    M, K, N = 16, 18944, 3584
    nib = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int64)
    packed = torch.zeros(K, N // 8, device="cuda", dtype=torch.int64)
    for i in range(8):
        packed |= (nib[:, i::8] << (4 * i))
    qweight = packed.to(torch.int32)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    scales = (torch.randn(K // GROUP, N, device="cuda") * 0.01).to(torch.float16)
    zeros = torch.randint(0, 16, (K // GROUP, N), device="cuda").to(torch.float16)
    return [x, qweight, scales, zeros]
