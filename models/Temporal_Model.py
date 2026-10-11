import torch
from einops import rearrange, repeat
from torch import nn, einsum
import math


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        # print(self.net(x))
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        # for name, param in self.named_parameters():
        #     if 'to_qkv' in name:
        #         print(param.data)

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)               
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                                              Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x
    
class Temporal_Transformer_AttnPool(nn.Module):
    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        dropout = 0.1
        self.num_patches = num_patches
        self.input_dim = input_dim
        
        # Position embeddings for frame tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, input_dim))
        self.temporal_transformer = Transformer(input_dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # Attention Pooling layers
        self.attention_pool = nn.Sequential(
            nn.Linear(input_dim, mlp_dim // 2),
            nn.Tanh(),
            nn.Linear(mlp_dim // 2, 1)
        )
        
        # Initialize attention pool weights
        for m in self.attention_pool.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, num_segments, input_dim) -> (B, 16, 512)
        b, n, d = x.shape
        
        # Add positional embedding to all frame tokens
        x = x + self.pos_embedding[:, :n]
        
        # Process through Transformer layers
        x = self.temporal_transformer(x) # (B, 16, 512)
        
        # Compute attention scores for each frame
        # (B, 16, 512) -> (B, 16, 1)
        attn_weights = self.attention_pool(x)
        attn_weights = torch.softmax(attn_weights, dim=1) # Normalize over 16 frames
        
        # Weighted sum: (B, 1, 16) @ (B, 16, 512) -> (B, 1, 512)
        pooled_x = torch.bmm(attn_weights.transpose(1, 2), x)
        pooled_x = pooled_x.squeeze(1) # (B, 512)
        
        return pooled_x