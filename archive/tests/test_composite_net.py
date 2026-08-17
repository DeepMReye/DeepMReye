"""Unit tests for deepmreye.models.composite_net."""
import tempfile
import torch
import numpy as np

from deepmreye.models.composite_net import CompositeNet, save_composite_net, load_composite_net


def test_composite_net_dimensions():
    B, V, K, H = 16, 500, 32, 64
    model = CompositeNet(n_voxels=V, bottleneck_dim=K, hidden_dim=H, alpha=0.1)

    x = torch.randn(B, V)
    y = torch.randn(B, 2)

    z, x_hat, y_hat = model(x)
    assert z.shape == (B, K), f"Expected bottleneck shape {(B, K)}, got {z.shape}"
    assert x_hat.shape == (B, V), f"Expected x_hat shape {(B, V)}, got {x_hat.shape}"
    assert y_hat.shape == (B, 2), f"Expected y_hat shape {(B, 2)}, got {y_hat.shape}"

    loss, gaze_loss, recon_loss = model.compute_loss(x, y)
    assert torch.isfinite(loss), "Loss should be finite"
    assert torch.isfinite(gaze_loss), "Gaze loss should be finite"
    assert torch.isfinite(recon_loss), "Reconstruction loss should be finite"


def test_composite_net_masked_loss():
    B, V, K = 8, 200, 16
    model = CompositeNet(n_voxels=V, bottleneck_dim=K)

    x = torch.randn(B, V)
    y = torch.randn(B, 2)

    # Partial valid mask (some rows have NaN targets)
    mask = torch.tensor([True, True, False, True, False, False, True, True])

    loss, gaze_loss, recon_loss = model.compute_loss(x, y, valid_mask=mask)
    assert torch.isfinite(loss), "Masked loss should be finite"


def test_composite_net_checkpoint_save_load():
    V, K = 300, 48
    model = CompositeNet(n_voxels=V, bottleneck_dim=K, alpha=0.2)

    x = torch.randn(4, V)
    z_orig, _, y_orig = model(x)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        save_composite_net(model, tmp.name, metadata={"test_key": "test_val"})
        loaded_model, meta = load_composite_net(tmp.name)

        z_loaded, _, y_loaded = loaded_model(x)

        assert meta.get("test_key") == "test_val"
        assert torch.allclose(z_orig.cpu(), z_loaded.cpu(), atol=1e-4)
        assert torch.allclose(y_orig.cpu(), y_loaded.cpu(), atol=1e-4)


def test_composite_net_optimization_step():
    V, K = 100, 16
    model = CompositeNet(n_voxels=V, bottleneck_dim=K, alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x = torch.randn(32, V)
    y = torch.randn(32, 2)

    initial_loss, _, _ = model.compute_loss(x, y)

    for _ in range(20):
        optimizer.zero_grad()
        loss, _, _ = model.compute_loss(x, y)
        loss.backward()
        optimizer.step()

    final_loss, _, _ = model.compute_loss(x, y)
    assert final_loss.item() < initial_loss.item(), "Multi-task loss should decrease after optimization steps"
