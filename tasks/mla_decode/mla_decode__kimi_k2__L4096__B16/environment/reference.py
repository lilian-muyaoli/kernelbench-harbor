import math
import torch
import torch.nn as nn

H, DH, DC = 16, 64, 256

class Model(nn.Module):
    """MLA decode, naive decompress-then-attend. q:[B,H,DH]; c_kv:[B,L,DC] cached latent;
    W_uk:[DC,H*DH]; W_uv:[DC,H*DH]. Fast kernel must use the 'absorb' trick (attend in
    latent space; never materialize per-head K/V)."""
    def forward(self, q, c_kv, W_uk, W_uv):
        B, L, _ = c_kv.shape
        K = (c_kv @ W_uk).view(B, L, H, DH).permute(0, 2, 1, 3)
        V = (c_kv @ W_uv).view(B, L, H, DH).permute(0, 2, 1, 3)
        att = torch.einsum('bhd,bhld->bhl', q, K) / math.sqrt(DH)
        att = torch.softmax(att, dim=-1)
        return torch.einsum('bhl,bhld->bhd', att, V)

def get_inputs():
    B, L = 16, 4096
    q = torch.randn(B, H, DH, device="cuda", dtype=torch.float16)
    c_kv = torch.randn(B, L, DC, device="cuda", dtype=torch.float16) * 0.02
    W_uk = torch.randn(DC, H * DH, device="cuda", dtype=torch.float16) * 0.02
    W_uv = torch.randn(DC, H * DH, device="cuda", dtype=torch.float16) * 0.02
    return [q, c_kv, W_uk, W_uv]
