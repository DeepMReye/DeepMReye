import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import pytest
from deepmreye.models.jepa import JEPAModel
from deepmreye.models.patcher import apply_double_cross_mask

def test_jepa_tr_conditioning_forward():
    # The block has to be large enough that the mask ratios actually drop
    # something: apply_double_cross_mask takes int(N * ratio), so at one spatial
    # patch and two temporal patches nothing is masked, the target sequence is
    # empty, and the shape assertions below compare against zero tokens.
    B, X, Y, Z, T = 1, 16, 16, 16, 20
    x = torch.randn(B, X, Y, Z, T)
    tr = torch.tensor([0.8], dtype=torch.float32)

    # TR-conditioned model
    model_tr = JEPAModel(embed_dim=16, encoder_depth=1, predictor_depth=1, num_heads=2, use_tr=True)
    seq, n_s, n_t = model_tr.patcher(x)
    
    ctx, tgt, c_idx, t_idx = apply_double_cross_mask(seq, n_s, n_t, spatial_ratio=0.2, temporal_ratio=0.4, device='cpu')

    target_reps = model_tr.forward_target(tgt, c_idx, t_idx, n_s, n_t, tr=tr)
    context_reps = model_tr.forward_context(ctx, c_idx, n_s, n_t, tr=tr)
    pred_reps = model_tr.forward_predict(context_reps, t_idx, n_s, n_t, tr=tr)

    assert target_reps.shape[0] == B
    assert target_reps.shape[-1] == 16
    assert pred_reps.shape == target_reps.shape

def test_jepa_no_tr_forward():
    # The block has to be large enough that the mask ratios actually drop
    # something: apply_double_cross_mask takes int(N * ratio), so at one spatial
    # patch and two temporal patches nothing is masked, the target sequence is
    # empty, and the shape assertions below compare against zero tokens.
    B, X, Y, Z, T = 1, 16, 16, 16, 20
    x = torch.randn(B, X, Y, Z, T)

    # Legacy / No-TR model
    model_notr = JEPAModel(embed_dim=16, encoder_depth=1, predictor_depth=1, num_heads=2, use_tr=False)
    seq, n_s, n_t = model_notr.patcher(x)
    
    ctx, tgt, c_idx, t_idx = apply_double_cross_mask(seq, n_s, n_t, spatial_ratio=0.3, temporal_ratio=0.0, device='cpu')

    target_reps = model_notr.forward_target(tgt, c_idx, t_idx, n_s, n_t, tr=None)
    context_reps = model_notr.forward_context(ctx, c_idx, n_s, n_t, tr=None)
    pred_reps = model_notr.forward_predict(context_reps, t_idx, n_s, n_t, tr=None)

    assert target_reps.shape[0] == B
    assert target_reps.shape[-1] == 16
    assert pred_reps.shape == target_reps.shape

