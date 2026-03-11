import torch
import torch.nn as nn
import numpy as np
import copy
from .patcher import fMRIPatcher

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, x):
        res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + res
        
        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + res
        return x

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=256, depth=6, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """x: [B, seq_len, embed_dim]"""
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

class JEPAModel(nn.Module):
    """
    Core Joint Embedding Predictive Architecture for fMRI eye blocks.
    Composed of:
     - 4D Patcher
     - Context Encoder (Learns on masked inputs)
     - Target Encoder (EMA weights, evaluates on masked targets)
     - Predictor (Attempts to decode target representation from context)
    """
    def __init__(self, embed_dim=256, encoder_depth=6, predictor_depth=3, num_heads=8, max_n_s=500, max_n_t=500):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.patcher = fMRIPatcher(embed_dim=embed_dim)
        
        # Dual Positional Embeddings (Additive Grid)
        self.pos_s = nn.Embedding(max_n_s, embed_dim)
        self.pos_t = nn.Embedding(max_n_t, embed_dim)
        
        # Predictor specialized Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Core ViTs
        self.context_encoder = ViTEncoder(embed_dim, encoder_depth, num_heads)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        
        self.predictor = ViTEncoder(embed_dim, predictor_depth, num_heads)
        
        # Stop gradient on target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
    def _add_positional_embedding(self, tokens, spatial_ids, temporal_ids):
        """
        Adds independent spatial and temporal positional embeddings to tokens. 
        spatial_ids/temporal_ids: 1D tensors describing the location of each token.
        """
        ps = self.pos_s(spatial_ids)
        pt = self.pos_t(temporal_ids)
        return tokens + ps + pt
        
    def update_target_encoder(self, momentum=0.996):
        """Exponential Moving Average (EMA) update for strict JEPA architectures."""
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

    def forward_target(self, target_tokens, c_idx_tensor, t_idx_tensor, N_S, N_T):
        """Passes target tokens strictly through the EMA Target Encoder"""
        with torch.no_grad():
            # Recover indices locally for embeddings. 
            # This logic mimics the N_S x N_T grid flattening arithmetic: seq_idx = (s_id * N_T) + t_id
            target_s_ids = t_idx_tensor // N_T
            target_t_ids = t_idx_tensor % N_T
            
            x = self._add_positional_embedding(target_tokens, target_s_ids, target_t_ids)
            target_reps = self.target_encoder(x)
        return target_reps.contiguous()
        
    def forward_context(self, context_tokens, c_idx_tensor, N_S, N_T):
        """Passes valid unmasked tokens into Context Encoder"""
        context_s_ids = c_idx_tensor // N_T
        context_t_ids = c_idx_tensor % N_T
        
        x = self._add_positional_embedding(context_tokens, context_s_ids, context_t_ids)
        return self.context_encoder(x)

    def forward_predict(self, context_reps, t_idx_tensor, N_S, N_T):
        """
        Takes learned Context representations, and attempts to predict
        Target representations purely given the *Positional Embeddings* of the missing targets.
        """
        B, N_target = t_idx_tensor.shape
        
        # 1. Expand Mask Tokens for all targets
        mask_tokens = self.mask_token.expand(B, N_target, -1)
        
        # 2. Add specific Target Positional Embeddings
        target_s_ids = t_idx_tensor // N_T
        target_t_ids = t_idx_tensor % N_T
        mask_tokens = self._add_positional_embedding(mask_tokens, target_s_ids, target_t_ids)
        
        # 3. Concatenate (Context + Mask Tokens)
        # The exact implementation of classical JEPA passes only the masks into the Predictor, 
        # cross-attending to the Context. Alternatively, we can self-attend a concatenated sequence.
        # ViT typically self-attends context concatenated with MASK.
        concat_sequence = torch.cat([context_reps, mask_tokens], dim=1)
        
        pred_full = self.predictor(concat_sequence)
        
        # 4. Extract *only* the predictions corresponding to the Mask Tokens
        # They were concatenated at the end, so we slice them off
        pred_targets = pred_full[:, -N_target:, :]
        return pred_targets.contiguous()
