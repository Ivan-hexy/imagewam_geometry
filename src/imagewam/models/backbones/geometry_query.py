"""Train-only queries over the exact post-RoPE KV consumed by Action DiT.

LiLa-WAM reference: models/vla_model_fm.py, VisualFeatureAdapter and
MultiLayerConcatFusion (339860f65d5cde9e229c037b672889bc4f70497c).
Here the memory is native edit K/V, rather than concatenated DINO features.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class EditingQueryFusion(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=512, num_queries=64,
                 num_heads=8, num_layers=25, refine_layers=2):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError('Query width must be divisible by attention heads')
        self.num_queries, self.hidden_dim, self.num_heads = num_queries, hidden_dim, num_heads
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * .02)
        self.key_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
        self.value_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_embeddings = nn.Parameter(torch.randn(num_layers, hidden_dim) * .02)
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.refine = nn.TransformerEncoder(layer, refine_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _heads(self, tensor):
        b,n,_ = tensor.shape
        return tensor.reshape(b,n,self.num_heads,-1).transpose(1,2)

    def forward(self, cache: dict, prefix_valid: torch.Tensor):
        layers = list(cache['double']) + list(cache['single'])
        if len(layers) != len(self.key_norms):
            raise ValueError(f'Expected {len(self.key_norms)} KV layers, got {len(layers)}')
        b,n,_ = layers[0]['k'].shape
        if prefix_valid.shape != (b,n) or not prefix_valid.any(dim=-1).all():
            raise ValueError('KV mask must match the inference-visible prefix')
        query = self.query_tokens.expand(b,-1,-1)
        q = self._heads(self.q_proj(self.q_norm(query)))
        fused = []
        for i,layer in enumerate(layers):
            k = self.k_proj(self.key_norms[i](layer['k'])) + self.layer_embeddings[i]
            v = self.v_proj(self.value_norms[i](layer['v'])) + self.layer_embeddings[i]
            out = F.scaled_dot_product_attention(q,self._heads(k),self._heads(v),
                attn_mask=prefix_valid[:,None,None,:],dropout_p=0.)
            fused.append(out.transpose(1,2).reshape(b,self.num_queries,self.hidden_dim))
        weights = self.layer_logits.softmax(0)
        memory = sum(w*x for w,x in zip(weights,fused))
        return self.final_norm(self.refine(query + self.out_proj(memory)))


class VirtualFutureConditioner(nn.Module):
    """Small nonzero residual: future patches read Z_edit; current patches stay native."""
    def __init__(self, patch_dim=1024, edit_dim=512, num_heads=16, initial_gate=.01):
        super().__init__()
        self.heads = num_heads
        self.norm = nn.LayerNorm(patch_dim)
        self.edit_norm = nn.LayerNorm(edit_dim)
        self.q = nn.Linear(patch_dim,patch_dim)
        self.k = nn.Linear(edit_dim,patch_dim)
        self.v = nn.Linear(edit_dim,patch_dim)
        self.proj = nn.Linear(patch_dim,patch_dim)
        self.ffn = nn.Sequential(nn.LayerNorm(patch_dim),nn.Linear(patch_dim,2*patch_dim),
            nn.GELU(),nn.Linear(2*patch_dim,patch_dim))
        self.future_type = nn.Parameter(torch.zeros(1,1,patch_dim))
        # Nonzero gates preserve small initial perturbations AND gradients to edit KV.
        self.cross_gate = nn.Parameter(torch.tensor(float(initial_gate)))
        self.ffn_gate = nn.Parameter(torch.tensor(float(initial_gate)))

    def forward(self, current_patches, edit):
        b,n,d = current_patches.shape
        def heads(x):return x.reshape(b,x.shape[1],self.heads,d//self.heads).transpose(1,2)
        query = self.norm(current_patches + self.future_type)
        memory = self.edit_norm(edit)
        update = F.scaled_dot_product_attention(heads(self.q(query)),heads(self.k(memory)),heads(self.v(memory)),dropout_p=0.)
        update = update.transpose(1,2).reshape(b,n,d)
        future = current_patches + self.future_type + self.cross_gate*self.proj(update)
        return future + self.ffn_gate*self.ffn(future)
