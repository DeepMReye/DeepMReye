import torch
import torch.nn as nn
import numpy as np

class fMRIPatcher(nn.Module):
    """
    Takes 4D fMRI blocks [B, X, Y, Z, T] and patchifies them 
    both spatially (grouping valid eye voxels) and temporally (grouping TRs).
    """
    def __init__(self, embed_dim=256, temp_patch_size=5, spat_patch_size=8, valid_mask=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.t_patch = temp_patch_size
        self.s_patch = spat_patch_size
        
        # valid_mask: [X,Y,Z] binary tensor of the eye-balls.
        # If none, we will dynamically calculate it based on non-zero variance.
        self.register_buffer('valid_mask', valid_mask if valid_mask is not None else None)
        
        # Calculate expected number of raw values per token
        # Spatially we take s_patch^3 voxels, temporally t_patch TRs.
        self.voxels_per_spat_patch = self.s_patch ** 3
        self.raw_dim = self.voxels_per_spat_patch * self.t_patch
        
        self.proj = nn.Linear(self.raw_dim, self.embed_dim)
        
    def _create_spatial_patches(self, x_4d):
        """
        x_4d: [B, X, Y, Z, T]
        Groups X,Y,Z into overlapping/non-overlapping cubes of size `s_patch`.
        Returns: [B, num_spat_patches, voxels_per_patch, T]
        """
        B, X, Y, Z, T = x_4d.shape
        
        # Unfold spatial dimensions (stride = spat_patch_size for non-overlapping)
        # Using pytorch unfolding. For 3D, we need to unfold sequentially or reshape.
        # Much simpler to just pad then reshape if we assume dense grids, 
        # but since we have a tight bounding box, we can just grid loop or unfold.
        
        # Let's pad to multiples of s_patch
        pad_x = (self.s_patch - X % self.s_patch) % self.s_patch
        pad_y = (self.s_patch - Y % self.s_patch) % self.s_patch
        pad_z = (self.s_patch - Z % self.s_patch) % self.s_patch
        
        import torch.nn.functional as F
        # padding format (backwards): Z, Y, X
        x_pad = F.pad(x_4d, (0, 0, 0, pad_z, 0, pad_y, 0, pad_x)) # T dimension is first in pad tuple natively, wait, no.
        # The pad tuple needs to go backwards starting from the *last* dimension.
        # last dim is T (no pad), then Z, Y, X.
        x_pad = F.pad(x_4d, (0, 0, 0, pad_z, 0, pad_y, 0, pad_x))
        
        B, nX, nY, nZ, T = x_pad.shape
        
        # Reshape into patches
        x_patch = x_pad.view(B, 
                             nX // self.s_patch, self.s_patch,
                             nY // self.s_patch, self.s_patch,
                             nZ // self.s_patch, self.s_patch,
                             T)
                             
        # Rearrange to grouped patches
        # [B, nX_p, nY_p, nZ_p, s_patch, s_patch, s_patch, T]
        x_patch = x_patch.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        
        num_spat_patches = (nX // self.s_patch) * (nY // self.s_patch) * (nZ // self.s_patch)
        x_patch = x_patch.view(B, num_spat_patches, self.voxels_per_spat_patch, T)
        
        # If we have a mask, we should only keep the spatial patches that actually contain valid eye data.
        # For simplicity in this general module, we'll keep all within the tight bounding box,
        # or filter those with strictly 0 variance (zero-padded background).
        
        return x_patch, num_spat_patches
        
    def _create_temporal_patches(self, x_patch, num_spat_patches):
        """
        x_patch: [B, num_spat_patches, voxels_per_patch, T]
        Groups T into non-overlapping temporal windows of size `t_patch`.
        Returns: [B, num_spat_patches, num_temp_patches, expected_raw_dim]
        """
        B, N_S, V, T = x_patch.shape
        # Pad temporal if necessary
        pad_t = (self.t_patch - T % self.t_patch) % self.t_patch
        
        import torch.nn.functional as F
        # pad for last dimension (T)
        if pad_t > 0:
            x_patch = F.pad(x_patch, (0, pad_t))
            
        T_new = x_patch.shape[-1]
        num_temp_patches = T_new // self.t_patch
        
        # [B, N_S, V, N_T, t_patch]
        x_patch = x_patch.view(B, N_S, V, num_temp_patches, self.t_patch)
        
        # Rearrange to put N_S and N_T together explicitly as a grid
        # [B, N_S, N_T, V, t_patch]
        x_grid = x_patch.permute(0, 1, 3, 2, 4).contiguous()
        
        # Flatten the raw values within the patch
        # [B, N_S, N_T, V * t_patch]
        x_grid = x_grid.view(B, N_S, num_temp_patches, self.raw_dim)
        
        return x_grid, num_temp_patches

    def forward(self, x):
        """
        x: [B, X, Y, Z, T] 
        Returns: tokens [B, N_S * N_T, D], spatial_idx [N_S], temporal_idx [N_T]
        """
        x_spat, N_S = self._create_spatial_patches(x)
        x_grid, N_T = self._create_temporal_patches(x_spat, N_S)
        
        # Project raw patches into embedding dimension
        B = x.shape[0]
        # [B, N_S, N_T, D]
        embeddings = self.proj(x_grid)
        
        # Flatten the grid into a 1D sequence for ViT ingestion
        # [B, N_S * N_T, D]
        seq = embeddings.view(B, N_S * N_T, self.embed_dim)
        
        return seq, N_S, N_T

def apply_double_cross_mask(seq, N_S, N_T, spatial_ratio=0.3, temporal_ratio=0.3, device='cuda'):
    """
    Applies the Continuous 2D Double-Cross Masking Logic on the 2D Grid.
    seq: [B, N_S * N_T, D]
    Returns:
       context_seq: [B, N_unmasked, D]  (Only valid unmasked tokens)
       target_seq: [B, N_masked, D]     (Only masked tokens to be predicted)
       context_idx: [B, N_unmasked]     (Indices to recover position)
       target_idx: [B, N_masked]        (Indices to recover position)
    """
    B, total_tokens, D = seq.shape
    
    num_drop_s = int(N_S * spatial_ratio)
    num_drop_t = int(N_T * temporal_ratio)
    
    # We sample independently for each batch item for robustness
    context_tokens = []
    target_tokens = []
    context_indices = []
    target_indices = []
    
    for b in range(B):
        # 1. Sample dropped spatial indices
        dropped_s = torch.randperm(N_S, device=device)[:num_drop_s]
        # 2. Sample dropped temporal indices
        dropped_t = torch.randperm(N_T, device=device)[:num_drop_t]
        
        is_masked = torch.zeros((N_S, N_T), dtype=torch.bool, device=device)
        is_masked[dropped_s, :] = True
        is_masked[:, dropped_t] = True
        
        # Flatten back to the sequence length representation
        is_masked_flat = is_masked.view(-1)
        
        # Extract sequences
        c_idx = torch.where(~is_masked_flat)[0]
        t_idx = torch.where(is_masked_flat)[0]
        
        context_tokens.append(seq[b, c_idx])
        target_tokens.append(seq[b, t_idx])
        context_indices.append(c_idx)
        target_indices.append(t_idx)
        
    # In a dynamic masking scenario, the number of masked vs unmasked tokens 
    # strictly varies per batch if overlapping logic isn't perfectly identical!
    # However, since `dropped_s` and `dropped_t` are exact counts, the union of the mask 
    # will exactly match in size for every item in the batch.
    
    # Size of mask = (drop_s * N_T) + (drop_t * N_S) - (drop_s * drop_t)
    expected_masked = (num_drop_s * N_T) + (num_drop_t * N_S) - (num_drop_s * num_drop_t)
    
    try:
        context_seq = torch.stack(context_tokens)
        target_seq = torch.stack(target_tokens)
        c_idx_tensor = torch.stack(context_indices)
        t_idx_tensor = torch.stack(target_indices)
        return context_seq, target_seq, c_idx_tensor, t_idx_tensor
    except RuntimeError:
        # If sizes mismatch for extremely bizarre rounding edge cases, pad.
        raise ValueError("Dynamic masking created heterogeneous sequence lengths. Implementation edge case hit.")
