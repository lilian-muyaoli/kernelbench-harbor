"""Oracle for nsa_block_sparse — native-sparse-attention-style block-sparse causal attention.

Block selection (mean-key score -> causal mask -> top-TOPB blocks) is done in torch with
broadcast multiply+sum (no banned matmul/einsum), in fp32 to match the fp32 ground-truth
selection exactly (the top-k is discrete, so the selection must be computed at GT precision).
The attention is a flash-style Triton kernel over query tiles: per query-block it visits only
the key-blocks selected by some query in the tile, with per-query causal + selection masking,
online softmax and value accumulation via tl.dot. Computing only selected blocks beats
torch.compile of the dense O(N^2) reference.
"""
import math
import torch
import triton
import triton.language as tl

BS = 64
TOPB = 4


@triton.jit
def _nsa_attn(q_ptr, k_ptr, v_ptr, sel_ptr, out_ptr,
              N, scale,
              sq_bh, sq_n, sk_bh, sk_n, sv_bh, sv_n,
              ssel_bh, ssel_n, so_bh, so_n,
              D: tl.constexpr, BM: tl.constexpr, BS_K: tl.constexpr, NB: tl.constexpr):
    bh = tl.program_id(0)
    qt = tl.program_id(1)                                # query tile index
    rows = qt * BM + tl.arange(0, BM)                    # query indices [BM]
    d = tl.arange(0, D)
    q = tl.load(q_ptr + bh * sq_bh + rows[:, None] * sq_n + d[None, :])  # [BM,D] fp16
    qb_max = (qt * BM + BM - 1) // BS_K                  # last key-block this tile can see

    m_i = tl.full((BM,), -float("inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    cols = tl.arange(0, BS_K)                            # keys within a block
    for c in range(0, NB):
        if c <= qb_max:                                  # block-level causal (for the tile)
            sel = tl.load(sel_ptr + bh * ssel_bh + rows * ssel_n + c)   # [BM] per-query selected?
            if tl.max(sel) != 0:                         # some query in the tile uses block c
                m_idx = c * BS_K + cols                  # key indices [BS_K]
                kk_t = tl.load(k_ptr + bh * sk_bh + d[:, None] + m_idx[None, :] * sk_n)  # [D,BS_K]
                vv = tl.load(v_ptr + bh * sv_bh + m_idx[:, None] * sv_n + d[None, :])    # [BS_K,D]
                s = tl.dot(q, kk_t, out_dtype=tl.float32) * scale  # [BM,BS_K] tensor-core
                causal = rows[:, None] >= m_idx[None, :]
                keep = causal & (sel[:, None] != 0)
                s = tl.where(keep, s, -float("inf"))
                m_new = tl.maximum(m_i, tl.max(s, axis=1))
                # guard rows still empty (m_new == -inf) so exp() never sees -inf-(-inf)
                m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
                p = tl.exp(s - m_safe[:, None])          # empty/masked entries -> exp(-inf)=0
                alpha = tl.exp(m_i - m_safe)             # empty rows -> exp(-inf)=0 (keeps acc 0)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), vv, out_dtype=tl.float32)
                m_i = m_safe
    out = acc / l_i[:, None]
    tl.store(out_ptr + bh * so_bh + rows[:, None] * so_n + d[None, :], out)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v):
        B, H, N, D = q.shape
        nb = N // BS
        scale = 1.0 / math.sqrt(D)
        # --- block selection in torch, fp32 to match GT exactly (top-k is discrete) ---
        # Score each KV block by its mean key. Compute block-by-block to avoid materializing
        # the [B,H,N,nb,D] product (17 GB at N=2048) — that broadcast was the perf bottleneck.
        kb = k.float().view(B, H, nb, BS, D).mean(3)                 # [B,H,nb,D]
        qf = q.float()
        bscore = torch.empty(B, H, N, nb, device=q.device, dtype=torch.float32)
        for c in range(nb):
            bscore[..., c] = (qf * kb[:, :, c:c + 1, :]).sum(-1) * scale
        qpos = torch.arange(N, device=q.device) // BS
        cblk = qpos[:, None] >= torch.arange(nb, device=q.device)[None, :]
        bscore = bscore.masked_fill(~cblk, float("-inf"))
        topb = bscore.topk(min(TOPB, nb), dim=-1).indices
        sel = torch.zeros(B, H, N, nb, dtype=torch.int32, device=q.device)
        sel.scatter_(-1, topb, 1)
        # --- flash-style sparse attention in triton ---
        qf = q.reshape(B * H, N, D).contiguous()
        kf = k.reshape(B * H, N, D).contiguous()
        vf = v.reshape(B * H, N, D).contiguous()
        self_sel = sel.reshape(B * H, N, nb).contiguous()
        out = torch.empty((B * H, N, D), device=q.device, dtype=torch.float32)
        BM = 64
        grid = (B * H, N // BM)
        _nsa_attn[grid](
            qf, kf, vf, self_sel, out, N, scale,
            qf.stride(0), qf.stride(1), kf.stride(0), kf.stride(1),
            vf.stride(0), vf.stride(1), self_sel.stride(0), self_sel.stride(1),
            out.stride(0), out.stride(1), D=D, BM=BM, BS_K=BS, NB=nb,
            num_warps=4, num_stages=2,
        )
        return out.reshape(B, H, N, D).to(q.dtype)
