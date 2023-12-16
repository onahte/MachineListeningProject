import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class Embedding(nn.Module):
    def __init__(self, in_channels=3, patch_height=64, patch_width=1, emb_size=128):
        super(Embedding, self).__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.projection = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=patch_height, pw=patch_width),
            nn.Linear(patch_height * patch_width * in_channels, emb_size)
        )

    def forward(self, x):
        return self.projection(x)


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.attention = nn.TransformerEncoderLayer(d_model=dim,
                                                    nhead=n_heads,
                                                    dropout=dropout,
                                                    activation='gelu')
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attention_output, attention_output_weights = self.attention(x, x, x)
        return attention_output


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, s, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        ff = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout),
            )

    def forward(self, x):
        return self.ff(x)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x += self.fn(x, **kwargs)
        return x


class ViT(nn.Module):
    def __init__(self, 
                 channels=3, 
                 img_size=64, 
                 patch_height=64, 
                 patch_width=1, 
                 emb_dim=32,
                 n_layers=6,
                 out_dim=15,
                 dropout=0.1,
                 heads=8
                 ):
        super(ViT, self).__init__()
        self.channels = channels
        self.height = self.width = img_size
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.n_layers = n_layers

        self.patch_embedding = Embedding(in_channels=channels,
                                         patch_height=patch_height,
                                         patch_width=patch_width,
                                         emb_size=emb_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, emb_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads=heads, dropout=dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout=dropout)))
            )
            self.layers.append(transformer_block)

        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, out_dim))

    def forward(self, x):
        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        for i in range(self.n_layers):
            x = self.layers[i][x]

        return self.head(x[:, 0, :])

