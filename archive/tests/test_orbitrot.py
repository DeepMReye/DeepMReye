"""The rotation bottleneck.

Two things here are worth guarding rather than merely exercising.

The hand-written ``_sample_trilinear`` / ``_affine_grid`` exist only because MPS
has no ``grid_sampler_3d_backward``; they are therefore a *reimplementation of a
reference*, and the way they fail is silently. The first version paired
``align_corners=False``'s pixel mapping with ``True``'s base grid, which shifts
every sample by half a cell -- the model still trains, just on a subtly wrong
render. So they are tested against ``torch.nn.functional`` directly.

And the claim the architecture rests on is that two numbers *cannot* express
anything but an orientation. That is only true if the decoder's sole access to
them is through a rotation, so the tests check the bottleneck's width and that
the render actually moves with the angle.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from deepmreye.crossorbit import _batch_loss  # noqa: E402
from deepmreye.orbitrot import (  # noqa: E402
    MAX_ANGLE,
    ORBIT_SHAPE,
    RotationOrbitModel,
    _affine_grid,
    _rotation,
    _sample_trilinear,
)

SMALL = (8, 10, 6)


def _model(**kw):
    kw.setdefault("n_angles", 2)
    kw.setdefault("n_nuisance", 8)
    kw.setdefault("width", 4)
    kw.setdefault("shape", SMALL)
    kw.setdefault("template_channels", 3)
    return RotationOrbitModel(device="cpu", seed=0, **kw)


# ------------------------------------------------------------------ rotation

@pytest.mark.parametrize("n", [2, 3])
def test_rotations_are_orthogonal_with_unit_determinant(n):
    a = torch.randn(6, n) * 0.5
    r = _rotation(a)
    assert r.shape == (6, 3, 3)
    assert torch.allclose(r @ r.transpose(1, 2), torch.eye(3).expand(6, 3, 3),
                          atol=1e-5)
    assert torch.allclose(torch.linalg.det(r), torch.ones(6), atol=1e-5)


def test_zero_angle_is_the_identity():
    assert torch.allclose(_rotation(torch.zeros(3, 2)), torch.eye(3).expand(3, 3, 3),
                          atol=1e-6)


def test_the_two_angles_rotate_about_different_axes():
    """If they shared an axis the bottleneck would be one-dimensional and could
    not span gaze."""
    a = _rotation(torch.tensor([[0.4, 0.0]]))
    b = _rotation(torch.tensor([[0.0, 0.4]]))
    assert not torch.allclose(a, b, atol=1e-3)


# --------------------------------------------------- the hand-written sampler

@pytest.mark.parametrize("seed", range(6))
def test_affine_grid_matches_torch(seed):
    torch.manual_seed(seed)
    d, h, w = (int(x) for x in torch.randint(4, 11, (3,)))
    theta = torch.randn(3, 3, 4) * 0.4
    theta[:, :, :3] += torch.eye(3)
    ref = F.affine_grid(theta, [3, 2, d, h, w], align_corners=False)
    assert torch.allclose(ref, _affine_grid(theta, (d, h, w), ref.device, ref.dtype),
                          atol=1e-5)


@pytest.mark.parametrize("seed", range(6))
def test_trilinear_sampling_matches_torch(seed):
    """The half-cell bug lived exactly here and changed nothing observable
    except the numbers."""
    torch.manual_seed(seed)
    d, h, w = (int(x) for x in torch.randint(4, 11, (3,)))
    vol = torch.randn(3, 2, d, h, w)
    theta = torch.randn(3, 3, 4) * 0.4
    theta[:, :, :3] += torch.eye(3)
    grid = F.affine_grid(theta, list(vol.shape), align_corners=False)
    ref = F.grid_sample(vol, grid, align_corners=False, padding_mode="border")
    assert torch.allclose(ref, _sample_trilinear(vol, grid), atol=1e-4)


def test_the_identity_transform_returns_the_volume_unchanged():
    vol = torch.randn(2, 3, *SMALL)
    eye = torch.eye(3, 4).expand(2, 3, 4).contiguous()
    grid = _affine_grid(eye, SMALL, vol.device, vol.dtype)
    assert torch.allclose(vol, _sample_trilinear(vol, grid), atol=1e-5)


def test_sampling_outside_the_volume_clamps_rather_than_zeroing():
    """``padding_mode='border'``. Zero padding would make a large angle look
    like a loss of signal and bias the angle toward zero."""
    vol = torch.ones(1, 1, *SMALL)
    grid = torch.full((1, *SMALL, 3), 5.0)
    assert torch.allclose(_sample_trilinear(vol, grid), torch.ones(1, 1, *SMALL))


def test_sampling_is_differentiable_in_both_the_volume_and_the_grid():
    vol = torch.randn(1, 2, *SMALL, requires_grad=True)
    grid = (torch.rand(1, *SMALL, 3) * 1.6 - 0.8).requires_grad_(True)
    _sample_trilinear(vol, grid).sum().backward()
    assert vol.grad is not None and torch.isfinite(vol.grad).all()
    assert grid.grad is not None and torch.isfinite(grid.grad).all()
    assert float(grid.grad.abs().sum()) > 0


# -------------------------------------------------------------- the model

def test_the_bottleneck_is_exactly_two_numbers_per_orbit():
    """The whole claim: gaze has two degrees of freedom and so does the latent."""
    angles, _ = _model(n_angles=2).encode(torch.randn(4, 1, *SMALL))
    assert angles.shape == (4, 2)


def test_angles_are_bounded_so_the_render_stays_in_frame():
    m = _model()
    big = torch.randn(8, 1, *SMALL) * 50
    angles, _ = m.encode(big)
    assert angles.abs().max() <= MAX_ANGLE + 1e-6


def test_the_render_moves_with_the_angle():
    """If it did not, the bottleneck would be dead by construction and no amount
    of training could revive it."""
    m = _model()
    with torch.no_grad():
        a = m.render(torch.zeros(1, 2))
        b = m.render(torch.tensor([[0.5, -0.3]]))
    assert float((a - b).abs().mean()) > 1e-3


def test_decode_reconstructs_the_orbit_shape():
    m = _model()
    x = torch.randn(3, 1, *SMALL)
    angles, nuis = m.encode(x)
    assert m.decode(angles, nuis).shape == (3, 1, *SMALL)


def test_the_rotation_centre_changes_the_render():
    """A rotation about the wrong point is a rotation plus a translation, which
    is why the centre is learned rather than assumed to be the crop's middle."""
    m = _model()
    angles = torch.tensor([[0.4, 0.2]])
    with torch.no_grad():
        before = m.render(angles).clone()
        m.centre += 0.5
        after = m.render(angles)
    assert float((before - after).abs().mean()) > 1e-4


def test_gradients_reach_every_parameter_including_template_and_centre():
    m = _model()
    x, x2 = torch.randn(2, 2, *SMALL), torch.randn(2, 2, *SMALL)
    loss, _ = _batch_loss(m, x, x2)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    assert float(m.template.grad.abs().sum()) > 0
    assert torch.isfinite(m.centre.grad).all()


def test_it_trains_through_the_shared_crossorbit_loss():
    """The comparison against ``xorb`` is only controlled if both run through
    the identical objective, selection rule and ablation."""
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x, x2 = torch.randn(4, 2, *SMALL), torch.randn(4, 2, *SMALL)
    first = None
    for _ in range(25):
        loss, _ = _batch_loss(m, x, x2)
        first = float(loss) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first


def test_the_shuffle_ablation_is_applicable_to_this_bottleneck():
    """``_batch_loss(shuffle_coords=True)`` is what decides whether the decoder
    uses the angle at all; it must accept a 2-wide latent, not just Kx3."""
    m = _model()
    rng = np.random.default_rng(0)
    x, x2 = torch.randn(4, 2, *SMALL), torch.randn(4, 2, *SMALL)
    _, r2 = _batch_loss(m, x, x2, shuffle_coords=True, rng=rng)
    assert np.isfinite(r2)


def test_the_untrained_control_is_reproducible_from_the_seed():
    a, b = _model(), _model()
    x = torch.randn(2, 1, *SMALL)
    assert torch.allclose(a.encode(x)[0], b.encode(x)[0])


def test_a_different_seed_gives_a_different_model():
    x = torch.randn(2, 1, *SMALL)
    other = RotationOrbitModel(2, 8, 4, seed=7, device="cpu", shape=SMALL,
                               template_channels=3)
    assert not torch.allclose(_model().encode(x)[0], other.encode(x)[0])


def test_save_and_load_round_trip_preserves_the_encoding(tmp_path):
    from deepmreye.crossorbit import save
    from deepmreye.orbitrot import load

    m = _model()
    path = save(tmp_path / "m.pt", m, {"width": 4})
    back, meta = load(path, device="cpu")
    x = torch.randn(3, 1, *SMALL)
    with torch.no_grad():
        assert torch.allclose(m.encode(x)[0], back.encode(x)[0], atol=1e-6)
        assert torch.allclose(m.template, back.template, atol=1e-6)
        assert torch.allclose(m.centre, back.centre, atol=1e-6)
    assert meta["width"] == 4


def test_the_default_orbit_shape_is_the_corpus_one():
    m = RotationOrbitModel(2, 8, 4, seed=0, device="cpu", template_channels=2)
    assert m.shape == tuple(ORBIT_SHAPE)
    assert m.template.shape[2:] == tuple(ORBIT_SHAPE)


def test_the_nuisance_path_is_wider_than_the_angle_path():
    """The asymmetry is the design: appearance has somewhere to go that is not
    the bottleneck, which is what stops it colonising the angle."""
    m = _model(n_angles=2, n_nuisance=32)
    angles, nuis = m.encode(torch.randn(2, 1, *SMALL))
    assert nuis.shape[1] > angles.shape[1] * 4


# ------------------------------------------------------------ rotating parts

def test_parts_widen_the_bottleneck_without_changing_its_kind():
    """The capacity control. The latent is still nothing but rotations -- there
    are simply more of them, one per rotating part."""
    m = _model(n_angles=2, n_parts=4)
    angles, _ = m.encode(torch.randn(3, 1, *SMALL))
    assert m.k == 8
    assert angles.shape == (3, 8)


def test_one_part_is_exactly_the_original_model():
    a = _model(n_parts=1)
    x = torch.randn(2, 1, *SMALL)
    with torch.no_grad():
        assert a.encode(x)[0].shape == (2, 2)
        assert a.render(torch.zeros(2, 2)).shape == (2, 3, *SMALL)


def test_render_emits_one_channel_block_per_part():
    m = _model(n_parts=3, template_channels=2)
    with torch.no_grad():
        out = m.render(torch.zeros(2, 6))
    assert out.shape == (2, 6, *SMALL)          # 3 parts x 2 channels


def test_each_part_rotates_independently():
    """If the parts shared a rotation the extra dimensions would be redundant
    and the capacity control would measure nothing."""
    m = _model(n_parts=2, template_channels=2)
    base = torch.zeros(1, 4)
    moved = base.clone()
    moved[0, 0] = 0.5                            # only part 0's first angle
    with torch.no_grad():
        a, b = m.render(base), m.render(moved)
    part0 = (a[:, :2] - b[:, :2]).abs().mean()
    part1 = (a[:, 2:] - b[:, 2:]).abs().mean()
    assert float(part0) > 1e-3
    assert float(part1) == pytest.approx(0.0, abs=1e-7)


def test_each_part_has_its_own_learned_centre():
    m = _model(n_parts=3)
    assert m.centre.shape == (3, 3)


def test_gradients_reach_every_part(reps=2):
    m = _model(n_parts=3, template_channels=2)
    x, x2 = torch.randn(reps, 2, *SMALL), torch.randn(reps, 2, *SMALL)
    loss, _ = _batch_loss(m, x, x2)
    loss.backward()
    grad = m.template.grad.view(3, 2, -1).abs().sum(dim=(1, 2))
    assert (grad > 0).all()
    assert torch.isfinite(m.centre.grad).all()


def test_a_multi_part_model_round_trips(tmp_path):
    from deepmreye.crossorbit import save
    from deepmreye.orbitrot import load

    m = _model(n_parts=3, template_channels=2)
    back, _ = load(save(tmp_path / "p.pt", m, {"width": 4}), device="cpu")
    x = torch.randn(2, 1, *SMALL)
    assert back.n_parts == 3 and back.k == 6
    with torch.no_grad():
        assert torch.allclose(m.encode(x)[0], back.encode(x)[0], atol=1e-6)


def test_an_untrained_control_must_match_the_trained_latent_width():
    """The invariant the probe's control path broke: a control built from stale
    config had 4 latent dims against a trained model's 24, which inflates every
    trained-minus-untrained margin computed from it. Rebuilding from the model's
    own attributes is what makes the two shapes agree by construction."""
    trained = _model(n_angles=2, n_parts=6, template_channels=2)
    control = RotationOrbitModel(
        trained.n_angles, trained.n_nuisance, 4, seed=1000, device="cpu",
        shape=trained.shape, template_channels=trained.template_channels,
        n_parts=trained.n_parts).eval()

    assert control.k == trained.k == 12
    x = torch.randn(2, 1, *SMALL)
    with torch.no_grad():
        assert control.encode(x)[0].shape == trained.encode(x)[0].shape
    # ...and it really is untrained: different weights, same architecture.
    assert not torch.allclose(control.template, trained.template)
