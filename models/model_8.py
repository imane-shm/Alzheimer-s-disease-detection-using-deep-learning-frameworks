import torch.nn as nn
import torch
import math

IMG_SIZE = 256
PATCH_SIZE = 16
IN_CH = 3
EMBED_DIM = 128
HEADS = 4
LAYERS = 12
MLP_RATIO = 4
DROPOUT = 0.1
ATTN_DROPOUT = 0.0

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(IN_CH, EMBED_DIM, PATCH_SIZE, PATCH_SIZE)
        n_patches = (IMG_SIZE // PATCH_SIZE) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, EMBED_DIM))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        return x

class MSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMBED_DIM, EMBED_DIM * 3)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(ATTN_DROPOUT)
        self.head_dim = EMBED_DIM // HEADS

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, HEADS, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.msa = MSA()
        self.drop1 = nn.Dropout(DROPOUT)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        hidden_dim = EMBED_DIM * MLP_RATIO
        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.drop1(self.msa(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.layers = nn.ModuleList([TransformerEncoderLayer() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x[:, 0])
