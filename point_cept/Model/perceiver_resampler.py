import torch

from torch import nn
from typing import List


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim_in: int=768,
        num_latents: int=64,
        dim_out: int=1024,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim_in))
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim_in, num_heads=8, batch_first=True)
        self.proj = nn.Linear(dim_in, dim_out)
        return

    def forward(self, pt_feats: List[torch.Tensor]) -> torch.Tensor:
        B = len(pt_feats)
        N_max = max(f.shape[0] for f in pt_feats)
        dim_in = pt_feats[0].shape[-1]
        device = pt_feats[0].device

        # pad 到相同长度并构建 key_padding_mask（True 表示被忽略的位置）
        pt_feat = torch.zeros(B, N_max, dim_in, device=device)
        pt_mask = torch.ones(B, N_max, dtype=torch.bool, device=device)
        for i, f in enumerate(pt_feats):
            n = f.shape[0]
            pt_feat[i, :n] = f
            pt_mask[i, :n] = False

        query = self.latents.repeat(B, 1, 1)

        fixed_cond, _ = self.cross_attn(
            query=query,
            key=pt_feat,
            value=pt_feat,
            key_padding_mask=pt_mask,
        )
        return self.proj(fixed_cond)
