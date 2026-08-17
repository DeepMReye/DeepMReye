"""Tests for `deepmreye.gauge`: zero-label gaze decoding & gauge matching.

Verifies:
- `gauge_by_teacher` recovers component index, sign, and quality.
- `gauge_by_teacher` is invariant to negating the teacher/variate relationship (recovers sign flip).
- `regress_out` projects out confound subspaces.
- `corr` handles NaN values and zero-variance inputs gracefully.
- `as_rows` flattens blocks into `[T, n_masked]` float64 correctly.
- `select_gauge` identifies the best matching signed components across runs.
- `decode` operates in both 'fixed' and 'adapted' modes on synthetic data.
"""
import numpy as np
import pytest

from deepmreye.gauge import (
    DEFAULT_GAUGE,
    as_rows,
    corr,
    decode,
    gauge_by_teacher,
    oracle_gauge,
    orbit_views,
    regress_out,
    run_cca,
    select_gauge,
)


def test_gauge_by_teacher_recovers_index_and_sign():
    rng = np.random.default_rng(42)
    t = 200
    teacher = rng.normal(size=t)

    # 5 variates; component 3 is -1.0 * teacher + slight noise
    variates = rng.normal(size=(t, 5))
    variates[:, 3] = -1.0 * teacher + 0.05 * rng.normal(size=t)

    idx, sign, quality = gauge_by_teacher(variates, teacher)
    assert idx == 3
    assert sign == -1.0
    assert quality > 0.95


def test_gauge_by_teacher_handles_nans_or_constant():
    t = 100
    teacher = np.ones(t)  # constant -> zero variance
    variates = np.random.randn(t, 4)

    idx, sign, quality = gauge_by_teacher(variates, teacher)
    assert idx == 0
    assert sign == 1.0
    assert np.isnan(quality)


def test_regress_out_orthogonality():
    rng = np.random.default_rng(123)
    t = 150
    confounds = rng.normal(size=(t, 2))
    x = rng.normal(size=(t, 5))

    clean = regress_out(x, confounds)
    c = np.column_stack([np.ones(t), confounds])

    # Dot product of clean columns with confound columns should be ~0
    residuals_cov = c.T @ clean
    assert np.allclose(residuals_cov, 0, atol=1e-10)


def test_corr_cases():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    b = -2.0 * a + 1.0

    assert np.isclose(corr(a, b), -1.0)

    # Short valid length
    a_nan = a.copy()
    a_nan[:5] = np.nan
    assert np.isnan(corr(a_nan, b))  # ok.sum() = 5 < 10

    # Constant array
    const = np.ones(10)
    assert np.isnan(corr(a, const))


def test_as_rows_and_orbit_views():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, :2] = True  # 2 left voxels
    mask[3, 3, :2] = True  # 2 right voxels
    # total 4 masked voxels

    t = 50
    block = np.arange(4 * 4 * 4 * t).reshape(4, 4, 4, t).astype(np.float32)
    rows = as_rows(block, mask)

    assert rows.shape == (t, 4)
    assert rows.dtype == np.float64

    fake_basis = {
        "left_index": np.array([0, 1]),
        "right_index": np.array([2, 3]),
    }
    left, right = orbit_views(rows, fake_basis)
    assert left.shape == (t, 2)
    assert right.shape == (t, 2)


def test_select_gauge_synthetic():
    rng = np.random.default_rng(999)
    t, k = 150, 10
    runs = []

    for _ in range(5):
        gaze = rng.normal(size=(t, 2))
        variates = rng.normal(size=(t, k))
        # axis 0 (x) mapped to comp 4 with +1 sign
        variates[:, 4] = gaze[:, 0] + 0.1 * rng.normal(size=t)
        # axis 1 (y) mapped to comp 2 with -1 sign
        variates[:, 2] = -gaze[:, 1] + 0.1 * rng.normal(size=t)
        runs.append((variates, gaze))

    gauge = select_gauge(runs, k=k)
    assert gauge["x"] == (4, 1.0)
    assert gauge["y"] == (2, -1.0)


def test_decode_modes_and_oracle():
    rng = np.random.default_rng(777)
    x_dim, y_dim, z_dim, t = 4, 4, 4, 120
    mask = np.zeros((x_dim, y_dim, z_dim), dtype=bool)
    mask[:2, :, :] = True
    mask[2:, :, :] = True  # 64 masked voxels
    n_masked = int(mask.sum())

    block = rng.normal(size=(x_dim, y_dim, z_dim, t)).astype(np.float32)

    # Fake basis dictionary
    li = np.arange(n_masked // 2)
    ri = np.arange(n_masked // 2, n_masked)
    fake_basis = {
        "mean": np.zeros(n_masked, dtype=np.float32),
        "left_index": li,
        "right_index": ri,
        "left_weights": rng.normal(size=(len(li), 32)).astype(np.float32),
        "right_weights": rng.normal(size=(len(ri), 32)).astype(np.float32),
        "canonical_correlations": np.ones(32, dtype=np.float32),
    }

    # Test fixed decode
    pred_fixed, info_fixed = decode(block, mask, fake_basis, mode="fixed")
    assert pred_fixed.shape == (t, 2)
    assert info_fixed["mode"] == "fixed"

    # Test adapted decode
    pred_adapted, info_adapted = decode(block, mask, fake_basis, mode="adapted", n_cc=4, n_pca=6)
    assert pred_adapted.shape == (t, 2)
    assert info_adapted["mode"] == "adapted"

    # Test oracle gauge
    variates = rng.normal(size=(t, 8))
    gaze = rng.normal(size=(t, 2))
    fit = slice(0, t // 2)
    orc = oracle_gauge(variates, gaze, fit)
    assert "x" in orc and "y" in orc
    assert 0 <= orc["x"][0] < 8
    assert orc["x"][1] in (-1.0, 1.0)
