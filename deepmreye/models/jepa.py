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
    def __init__(self, embed_dim=256, encoder_depth=6, predictor_depth=3, num_heads=8, max_n_s=500, max_n_t=500, use_tr=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_tr = use_tr
        
        self.patcher = fMRIPatcher(embed_dim=embed_dim)
        
        # Dual Positional Embeddings (Additive Grid)
        self.pos_s = nn.Embedding(max_n_s, embed_dim)
        self.pos_t = nn.Embedding(max_n_t, embed_dim)

        # TR Conditioning & Continuous Temporal Positional Encoding
        self.num_time_freqs = 16
        self.temp_pos_mlp = nn.Sequential(
            nn.Linear(self.num_time_freqs * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.tr_mlp = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )

        # Predictor specialized Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings are added to the patch tokens, so their scale
        # decides how much of a token is "where it is" versus "what is in it".
        # std 0.02 is the ViT / MAE / I-JEPA convention.
        nn.init.trunc_normal_(self.pos_s.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_t.weight, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.temp_pos_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.tr_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Core ViTs
        self.context_encoder = ViTEncoder(embed_dim, encoder_depth, num_heads)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        
        self.predictor = ViTEncoder(embed_dim, predictor_depth, num_heads)
        
        # Stop gradient on target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def _sinusoidal_time_embedding(self, t_sec, max_period=100.0):
        device = t_sec.device
        bands = torch.exp(torch.linspace(0, np.log(max_period), self.num_time_freqs, device=device))
        angles = t_sec.unsqueeze(-1) * bands.view(1, 1, -1)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return torch.cat([sin, cos], dim=-1)

    def _add_positional_embedding(self, tokens, spatial_ids, temporal_ids, tr=None):
        """
        Adds independent spatial and temporal positional embeddings to tokens. 
        spatial_ids/temporal_ids: tensors describing the location of each token.
        tr: repetition time tensor [B] in seconds (or None).
        """
        ps = self.pos_s(spatial_ids)
        
        if self.use_tr and tr is not None:
            # Reshape tr to [B, 1]
            tr_b = tr.view(-1, 1).to(tokens.device)
            # Physical time in seconds for each token center
            t_sec = (temporal_ids.float() + 0.5) * float(self.patcher.t_patch) * tr_b
            sin_emb = self._sinusoidal_time_embedding(t_sec)
            pt = self.temp_pos_mlp(sin_emb)
            
            log_tr = torch.log(torch.clamp(tr_b, min=1e-3))
            ptr = self.tr_mlp(log_tr).unsqueeze(1)
            
            return tokens + ps + pt + ptr
        else:
            pt = self.pos_t(temporal_ids)
            return tokens + ps + pt
        
    def update_target_encoder(self, momentum=0.996):
        """Exponential Moving Average (EMA) update for strict JEPA architectures."""
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

    def forward_target(self, target_tokens, c_idx_tensor, t_idx_tensor, N_S, N_T, tr=None):
        """Passes target tokens strictly through the EMA Target Encoder"""
        with torch.no_grad():
            target_s_ids = t_idx_tensor // N_T
            target_t_ids = t_idx_tensor % N_T
            
            x = self._add_positional_embedding(target_tokens, target_s_ids, target_t_ids, tr=tr)
            target_reps = self.target_encoder(x)
        return target_reps.contiguous()
        
    def forward_context(self, context_tokens, c_idx_tensor, N_S, N_T, tr=None):
        """Passes valid unmasked tokens into Context Encoder"""
        context_s_ids = c_idx_tensor // N_T
        context_t_ids = c_idx_tensor % N_T
        
        x = self._add_positional_embedding(context_tokens, context_s_ids, context_t_ids, tr=tr)
        return self.context_encoder(x)

    def forward_predict(self, context_reps, t_idx_tensor, N_S, N_T, tr=None):
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
        mask_tokens = self._add_positional_embedding(mask_tokens, target_s_ids, target_t_ids, tr=tr)
        
        # 3. Concatenate (Context + Mask Tokens)
        concat_sequence = torch.cat([context_reps, mask_tokens], dim=1)
        
        pred_full = self.predictor(concat_sequence)
        
        # 4. Extract *only* the predictions corresponding to the Mask Tokens
        pred_targets = pred_full[:, -N_target:, :]
        return pred_targets.contiguous()

