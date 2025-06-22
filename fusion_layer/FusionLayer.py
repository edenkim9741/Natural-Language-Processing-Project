import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self, text_dim, phonetic_dim):
        super().__init__()
        self.proj = nn.Linear(text_dim + phonetic_dim, text_dim)

    def forward(self, text_feat, phonetic_feat):
        concat = torch.cat([text_feat, phonetic_feat], dim=-1)
        return self.proj(concat)  # [B, T, D]
