import math
import torch
import torch.nn as nn

BS = 64
TOPB = 4

class Model(nn.Module):
    """Block-sparse causal attention (NSA). q,k,v:[B,H,N,D]. Each query scores KV blocks by
    mean key, selects top-TOPB causal blocks, attends only to keys in those blocks."""
    def forward(self, q, k, v):
        B, H, N, D = q.shape
        nb = N // BS
        scale = 1.0 / math.sqrt(D)
        kb = k.view(B, H, nb, BS, D).mean(3)
        bscore = torch.einsum('bhnd,bhcd->bhnc', q, kb) * scale
        qpos = torch.arange(N, device=q.device) // BS
        cblk = qpos[:, None] >= torch.arange(nb, device=q.device)[None, :]
        bscore = bscore.masked_fill(~cblk, float('-inf'))
        topb = bscore.topk(min(TOPB, nb), dim=-1).indices
        sel = torch.zeros(B, H, N, nb, dtype=torch.bool, device=q.device)
        sel.scatter_(-1, topb, True)
        keyblk = (torch.arange(N, device=q.device) // BS)
        keep = sel.gather(-1, keyblk[None, None, None, :].expand(B, H, N, N))
        att = torch.einsum('bhnd,bhmd->bhnm', q, k) * scale
        causal = torch.arange(N, device=q.device)[:, None] >= torch.arange(N, device=q.device)[None, :]
        att = att.masked_fill(~(causal & keep), float('-inf'))
        att = torch.softmax(att, dim=-1)
        return torch.matmul(att, v)

def get_inputs():
    B, H, N, D = 2, 16, 2048, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    return [q, k, v]
