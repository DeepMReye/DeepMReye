import pytest
import torch
import numpy as np
from deepmreye.models.patcher import fMRIPatcher, apply_double_cross_mask
from deepmreye.models.jepa import JEPAModel

def test_fmri_patcher_shapes():
    """
    Validates that a 4D fMRI block tensor correctly unfolds into 
    the spatial vs temporal 2D Token Grid.
    """
    # Create fake batch [B, X, Y, Z, T]
    B, X, Y, Z, T = 2, 8, 8, 8, 20
    x_input = torch.randn(B, X, Y, Z, T)
    
    # Init patcher: spatial patch size = 2 (so 2x2x2 = 8 voxels)
    # temporal patch size = 5 TRs
    # embed size = 128
    patcher = fMRIPatcher(embed_dim=128, temp_patch_size=5, spat_patch_size=2)
    
    seq, N_S, N_T = patcher(x_input)
    
    # Expected Grid Dimensions:
    # N_S = (8/2) * (8/2) * (8/2) = 4 * 4 * 4 = 64 spatial cubes
    # N_T = (20/5) = 4 temporal windows
    assert N_S == 64, f"Expected 64 spatial patches, got {N_S}"
    assert N_T == 4, f"Expected 4 temporal patches, got {N_T}"
    
    # Seq total length = 64 * 4 = 256 tokens per batch item
    assert seq.shape == (B, 256, 128), f"Expected seq shape {(B, 256, 128)}, got {seq.shape}"
    
def test_fmri_patcher_padding():
    """
    Validates the padding logic inside the patcher if X, Y, Z, or T 
    are not perfectly divisible by the patching sizes.
    """
    # Create non-divisible block: [B, X, Y, Z, T]
    B, X, Y, Z, T = 1, 9, 9, 9, 23
    x_input = torch.randn(B, X, Y, Z, T)
    
    patcher = fMRIPatcher(embed_dim=64, temp_patch_size=5, spat_patch_size=2)
    seq, N_S, N_T = patcher(x_input)
    
    # 9 should pad to 10. (10/2) = 5. So N_S = 5 * 5 * 5 = 125
    assert N_S == 125, f"Expected padded 125 spatial patches, got {N_S}"
    
    # 23 should pad to 25. (25/5) = 5. So N_T = 5
    assert N_T == 5, f"Expected padded 5 temporal patches, got {N_T}"
    
    assert seq.shape == (B, 125 * 5, 64)
    
def test_double_cross_masking():
    """
    Validates that the Double-Cross algorithm drops exactly the requested
    amount of spatial AND temporal patches across the 2D grid.
    """
    B, N_S, N_T, D = 4, 30, 20, 256
    seq = torch.randn(B, N_S * N_T, D)
    
    # 1. Edge Case: No Masking (0, 0)
    ctx, tgt, c_idx, t_idx = apply_double_cross_mask(seq, N_S, N_T, spatial_ratio=0.0, temporal_ratio=0.0, device='cpu')
    assert ctx.shape[1] == N_S * N_T, "Should keep all tokens in context"
    assert tgt.shape[1] == 0, "Target should be empty"
    
    # 2. Strict Spatial Masking (0.5, 0.0) -> Cross-Patch
    ctx, tgt, c_idx, t_idx = apply_double_cross_mask(seq, N_S, N_T, spatial_ratio=0.5, temporal_ratio=0.0, device='cpu')
    expected_tgt_len = int(N_S * 0.5) * N_T # Half the spatial blocks dropped across ALL time
    assert tgt.shape[1] == expected_tgt_len
    assert ctx.shape[1] == (N_S * N_T) - expected_tgt_len
    
    # 3. Strict Temporal Masking (0.0, 0.5) -> Cross-Time
    ctx, tgt, c_idx, t_idx = apply_double_cross_mask(seq, N_S, N_T, spatial_ratio=0.0, temporal_ratio=0.5, device='cpu')
    expected_tgt_len_t = int(N_T * 0.5) * N_S
    assert tgt.shape[1] == expected_tgt_len_t
    
    # 4. Double-Cross (0.5, 0.5)
    ctx, tgt, c_idx, t_idx = apply_double_cross_mask(seq, N_S, N_T, spatial_ratio=0.5, temporal_ratio=0.5, device='cpu')
    num_drop_s = int(N_S * 0.5)
    num_drop_t = int(N_T * 0.5)
    expected_double_tgt = (num_drop_s * N_T) + (num_drop_t * N_S) - (num_drop_s * num_drop_t)
    assert tgt.shape[1] == expected_double_tgt
    assert ctx.shape[1] + tgt.shape[1] == N_S * N_T

def test_jepa_forward_engine():
    """
    Tests an end-to-end forward pass through the un-initialized PyTorch architecture
    to verify dimensional gradients align.
    """
    B, X, Y, Z, T = 2, 10, 10, 10, 20
    x_input = torch.randn(B, X, Y, Z, T)
    
    # Initialize tiny model
    model = JEPAModel(embed_dim=64, encoder_depth=2, predictor_depth=1, num_heads=4)
    model.eval() # Disable dropout for deterministic check
    
    seq, N_S, N_T = model.patcher(x_input)
    
    # Drop half spatial, half temporal
    ctx, tgt, c_idx, t_idx = apply_double_cross_mask(seq, N_S, N_T, 0.5, 0.5, device='cpu')
    
    # Forward context
    context_reps = model.forward_context(ctx, c_idx, N_S, N_T)
    assert context_reps.shape == ctx.shape, "Context Encoder should preserve token sequence dimensions."
    
    # Target EMA 
    target_reps = model.forward_target(tgt, c_idx, t_idx, N_S, N_T)
    assert target_reps.shape == tgt.shape, "Target Encoder should preserve missing sequence dimensions."
    
    # Predictoer
    pred_reps = model.forward_predict(context_reps, t_idx, N_S, N_T)
    assert pred_reps.shape == target_reps.shape, f"Predictor must output shape exactly matching target reps. Expected {target_reps.shape}, got {pred_reps.shape}"
