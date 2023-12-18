import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

import CONFIG
from CONFIG import model_config as mofig


device = CONFIG.device


class Embedding(nn.Module):
    def __init__(self, 
                 patch_height=mofig.patch_height, 
                 patch_width=mofig.patch_width, 
                 emb_dim=mofig.emb_dim, 
                 latent_size=mofig.hidden_dim):
        super(Embedding, self).__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.emb_dim = emb_dim
        self.input_size = self.patch_height * self.patch_width * 4
        self.latent_size = latent_size
        self.batch_size = mofig.batch_size

        # Patchify
        self.patches = Rearrange('b c (ph h) (pw w) -> b (h w) (ph pw c)', 
                                 ph=self.patch_height,
                                 pw=self.patch_width)

        # Linear Projection
        self.projection = nn.Linear(self.input_size, self.latent_size)

        # Class Token
        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(device)

        # Position Embedding
        self.pos_emb = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(device)

    def forward(self, x):
        x = self.patches(x)
        x = self.projection(x)
        b, n, _ = x.shape
        x = torch.cat((self.class_token, x), dim=1)
        pos_emb = einops.repeat(self.pos_emb, 'b 1 d -> b m d', m=n+1)
        x += pos_emb
        return x

class EncoderBlock(nn.Module):
    def __init__(self, 
                 latent_size=mofig.hidden_dim,
                 num_heads=mofig.num_heads,
                 dropout=mofig.dropout):
        super(EncoderBlock, self).__init__()
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.norm = nn.LayerNorm(self.latent_size)

        self.attention = nn.MultiheadAttention(self.latent_size,
                                               self.num_heads,
                                               self.dropout)
        self.ff = nn.Sequential(
                    nn.Linear(self.latent_size, self.latent_size*3),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.latent_size*3, self.latent_size),
                    nn.Dropout(self.dropout)
                    )

    def forward(self, x):
        x_residual = x
        x = self.norm(x.clone())
        x = self.attention(x, x, x)[0]
        x_residual += x
        x = self.norm(x_residual.clone())
        x = self.ff(x)
        return x + x_residual


class ViT(nn.Module):
    def __init__(self, 
                 num_encoders=mofig.num_encoders,
                 latent_size=mofig.hidden_dim,
                 dropout=mofig.dropout,
                 num_classes=mofig.num_classes):
        super(ViT, self).__init__()
        self.num_encoders = num_encoders 
        self.latent_size = latent_size
        self.dropout = dropout
        self.num_classes = num_classes

        self.embedding = Embedding()

        self.encoder = nn.ModuleList([EncoderBlock() for i in range(self.num_encoders)])

        self.ff = nn.Sequential(
                    nn.LayerNorm(self.latent_size),
                    nn.Linear(self.latent_size, self.latent_size),
                    nn.Linear(self.latent_size, self.num_classes)
                    )

    def forward(self, x):
        x = self.embedding(x)

        for e in self.encoder:
            x = e.forward(x)

        class_token_emb = x[:, 0]

        return self.ff(class_token_emb)


