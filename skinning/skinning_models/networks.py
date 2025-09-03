# Modified from https://github.com/1zb/functional-diffusion

import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from timm.models.layers import DropPath

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()
        
        assert hidden_dim % 12 == 0
        
        self.embedding_dim = hidden_dim
        chunk_size = self.embedding_dim // 12
        freq = torch.pow(2, torch.arange(chunk_size)).float() * np.pi

        e = torch.zeros(6, chunk_size * 6)
        for i in range(6):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            e[i, start_idx:end_idx] = freq

        self.register_buffer('basis', e)
        self.mlp = nn.Linear(self.embedding_dim + 6, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum('bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 6
        x = self.embed(input, self.basis) # B,N,48
        embed = self.mlp(torch.cat([x, input], dim=2))  # B x N x C
        return embed
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None, modulated=False):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

        self.modulated = modulated
        if self.modulated:
            self.gamma = nn.Linear(dim, dim, bias=False)
            self.beta = nn.Linear(dim, dim, bias=False)

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if self.modulated:
            label = kwargs.pop('label')
            gamma = self.gamma(label) # b 1 c
            beta = self.beta(label) # b 1 c
            x = gamma * x + beta

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, query_mask = None, context_mask=None, rel_pos=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if exists(rel_pos):
            # rel_pos shape expected to be [b, i, j, h] or [b h, i, j]
            if rel_pos.dim() == 4:  # [b, i, j, h]
                # Reshape to match attention heads dimension
                rel_pos = rearrange(rel_pos, 'b i j h -> (b h) i j')
            
            # Add the relative positional bias to the attention scores
            sim = sim + rel_pos

        if exists(query_mask):  # shape (B, Nq)
            query_mask = query_mask.bool()
            if query_mask.dim() == 2:
                query_mask = repeat(query_mask, 'b i -> (b h) i 1', h=h)
            elif query_mask.dim() == 3:
                query_mask = repeat(query_mask, 'b n j -> (b h) n j', h=h)
            sim.masked_fill_(~query_mask, -torch.finfo(sim.dtype).max)

            
        if exists(context_mask):
            context_mask_bool = context_mask.bool()
            if context_mask_bool.dim() == 2:
                context_mask_bool = repeat(context_mask_bool, 'b j -> (b h) 1 j', h=h)
            elif context_mask_bool.dim() == 3:
                context_mask_bool = repeat(context_mask_bool, 'b n j -> (b h) n j', h=h)
            sim.masked_fill_(~context_mask_bool, -torch.finfo(sim.dtype).max)
        
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))
