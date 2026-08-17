"""Shared Response Modeling (SRM) / Generalized Procrustes Hyperalignment.

Aligns individual participant voxel spaces into a common k-dimensional latent
functional space prior to probe training.

For training participants, SRM learns a global template basis V_global and
subject-specific orthogonal rotation matrices W_p = V_p R_p such that individual
functional representations align with the consensus space.

For unseen test participants, their alignment matrix W_test is derived strictly
unsupervised from their own fMRI voxel time series aligned to V_global, with
zero access to gaze targets.
"""
import numpy as np
from sklearn.utils.extmath import randomized_svd


class SharedResponseModel:
    """Computes subject-specific SRM projections into a global consensus space."""

    def __init__(self, n_components=64):
        self.n_components = n_components
        self.v_global = None
        self.mu_global = None
        self.subject_bases = {}

    def fit(self, subject_data, seed=0):
        """Fit global template and subject alignment matrices.

        Parameters
        ----------
        subject_data : dict
            Mapping from subject_id (str) to voxel matrix [N_samples, D].
        seed : int
            Random seed for SVD.
        """
        all_centered = []
        subject_means = {}

        # 1. Compute per-subject means and center time-series
        for sub_id, rows in subject_data.items():
            rows = np.asarray(rows, dtype=np.float64)
            mu = rows.mean(axis=0)
            subject_means[sub_id] = mu
            all_centered.append(rows - mu)

        # 2. Compute global consensus basis V_global
        concat_rows = np.concatenate(all_centered, axis=0)
        self.mu_global = concat_rows.mean(axis=0)
        k = int(min(self.n_components, min(concat_rows.shape) - 1))

        _, _, vt_global = randomized_svd(
            concat_rows - self.mu_global, n_components=k, n_iter=4, random_state=seed
        )
        self.v_global = vt_global.T  # [D, k]

        # 3. Fit subject-specific orthogonal rotations W_p
        for sub_id, rows in subject_data.items():
            rows = np.asarray(rows, dtype=np.float64)
            mu = subject_means[sub_id]
            centered = rows - mu
            n_samples, d_voxels = centered.shape

            k_sub = min(k, min(n_samples, d_voxels) - 1)
            if k_sub < 1:
                self.subject_bases[sub_id] = (mu, self.v_global)
                continue

            _, _, vt_sub = randomized_svd(
                centered, n_components=k_sub, n_iter=4, random_state=seed
            )
            v_sub = vt_sub.T  # [D, k_sub]

            # Orthogonal Procrustes alignment of V_sub to V_global
            # Minimize || V_sub @ R - V_global ||_F^2 => SVD(V_sub.T @ V_global)
            M = v_sub.T @ self.v_global[:, :k_sub]
            u_p, _, vt_p = np.linalg.svd(M)
            r_p = u_p @ vt_p  # [k_sub, k_sub]
            w_p = v_sub @ r_p  # [D, k_sub]

            # Pad to [D, k] if k_sub < k
            if k_sub < k:
                w_full = np.zeros((d_voxels, k), dtype=np.float64)
                w_full[:, :k_sub] = w_p
                w_full[:, k_sub:] = self.v_global[:, k_sub:]
                w_p = w_full

            self.subject_bases[sub_id] = (mu, w_p)

    def _fit_unseen_subject(self, sub_id, rows, seed=0):
        """Derive W_test for a held-out test participant strictly unsupervised."""
        rows = np.asarray(rows, dtype=np.float64)
        mu = rows.mean(axis=0)
        centered = rows - mu
        n_samples, d_voxels = centered.shape
        k = self.v_global.shape[1]

        k_sub = min(k, min(n_samples, d_voxels) - 1)
        if k_sub < 1:
            self.subject_bases[sub_id] = (mu, self.v_global)
            return

        _, _, vt_sub = randomized_svd(
            centered, n_components=k_sub, n_iter=4, random_state=seed
        )
        v_sub = vt_sub.T

        M = v_sub.T @ self.v_global[:, :k_sub]
        u_p, _, vt_p = np.linalg.svd(M)
        r_p = u_p @ vt_p
        w_p = v_sub @ r_p

        if k_sub < k:
            w_full = np.zeros((d_voxels, k), dtype=np.float64)
            w_full[:, :k_sub] = w_p
            w_full[:, k_sub:] = self.v_global[:, k_sub:]
            w_p = w_full

        self.subject_bases[sub_id] = (mu, w_p)

    def transform(self, rows, sub_ids, seed=0):
        """Transform voxel rows [B, D] into SRM features [B, k] per subject."""
        rows = np.asarray(rows, dtype=np.float64)
        sub_ids = np.asarray(sub_ids)
        out = np.zeros((rows.shape[0], self.v_global.shape[1]), dtype=np.float64)

        unique_subs = np.unique(sub_ids)
        for sub_id in unique_subs:
            idx = np.where(sub_ids == sub_id)[0]
            sub_rows = rows[idx]

            if sub_id not in self.subject_bases:
                self._fit_unseen_subject(sub_id, sub_rows, seed=seed)

            mu, w_p = self.subject_bases[sub_id]
            out[idx] = (sub_rows - mu) @ w_p

        return out
