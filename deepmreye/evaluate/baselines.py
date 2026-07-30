"""The readout zoo: linear and simple non-linear maps from features to gaze.

Every arm of the evaluation -- raw voxels, a random encoder, a trained
checkpoint -- produces a feature matrix, and the *same* set of readouts is
fitted on each. Keeping the readouts here, separate from feature extraction,
is what makes the comparison honest: if the trained encoder wins, it wins with
the readout that the raw-voxel baseline also got.

Why these and not others:

- ``mean``      the zero line. Predicts the training-set mean gaze for
                everything. A model that does not beat this has learned nothing,
                and it is the reference R^2 is measured against.
- ``linear``    ordinary least squares. Included precisely because it should do
                badly: with ~500 features and correlated voxels it overfits, and
                the gap to ``ridge-cv`` is how much of the baseline's
                performance is regularisation rather than signal.
- ``ridge-cv``  ridge with alpha chosen by leave-one-out generalised CV over a
                wide log grid. **This is the baseline that matters.** A ridge at
                a hardcoded alpha=1.0 is the first thing a reviewer attacks,
                because it is trivially possible to under-tune the baseline you
                are trying to beat.
- ``pca-ridge`` unsupervised dimensionality reduction, then ridge. This is the
                closest non-learned analogue of "frozen encoder + linear probe":
                it also compresses to a low-dimensional space without seeing
                gaze. If the JEPA representation does not beat *this*, it has
                not shown that learning the compression helps.
- ``pls``       partial least squares -- supervised dimensionality reduction,
                the standard readout in the neuroimaging literature, so its
                absence would be conspicuous.
- ``rf``/``gbt`` a non-linear readout on the same features. Cheap insurance
                against the reading that a learned representation only helps
                because it is non-linear; if trees on raw voxels close the gap,
                that is the explanation.
- ``svr``/``lgbm``/``mlp`` the three non-DeepMReye regressors compared against
                the original CNN in ``media/deepmreye_benchmarks.ipynb``
                (``sklearn.svm.SVR``, ``lightgbm.LGBMRegressor``,
                ``sklearn.neural_network.MLPRegressor``). Included to reproduce
                that comparison on the current corpus. SVR is O(n^2)-O(n^3) in
                the number of training rows -- fine on a single dataset's worth
                of windows, potentially slow on a pooled leave-one-dataset-out
                fold; subsample with ``--max-windows`` if it does not finish.

Every readout is wrapped in a ``StandardScaler``. Feature scales differ by
orders of magnitude between arms (voxel means against transformer
activations), and an unscaled ridge penalty means something different in each,
which would make the arms incomparable rather than merely different.
"""
import lightgbm as lgb
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Wide enough that the chosen alpha is interior for both arms: voxel features
# want heavy regularisation, encoder embeddings much less.
ALPHA_GRID = np.logspace(-2, 6, 17)

# Readouts run by default. `linear`, `rf` and `gbt` are available but off --
# OLS is informative once, not every run, and the tree models cost minutes.
DEFAULT_READOUTS = ("mean", "ridge-cv", "pca-ridge", "pls")

ALL_READOUTS = ("mean", "linear", "ridge", "ridge-cv", "pca-ridge", "pls", "rf", "gbt",
                "svr", "lgbm", "mlp")


def _n_components(requested, n_samples, n_features):
    """Largest usable component count, never zero."""
    return max(1, min(requested, n_samples - 1, n_features))


def build_readout(name, n_samples, n_features, n_components=32, seed=0):
    """Construct one readout, sized for the data it will see.

    ``n_samples``/``n_features`` are needed because PCA and PLS cannot ask for
    more components than the data has -- on a small held-out fold that is a hard
    error rather than a degradation, so it is clamped here once.
    """
    scale = StandardScaler()

    if name == "mean":
        # No scaler: a constant predictor does not look at the features at all.
        return DummyRegressor(strategy="mean")
    if name == "linear":
        return make_pipeline(scale, LinearRegression())
    if name == "ridge":
        return make_pipeline(scale, Ridge(alpha=1.0))
    if name == "ridge-cv":
        # RidgeCV's default is efficient leave-one-out GCV, so the whole grid
        # costs about one fit -- there is no excuse for not tuning it.
        return make_pipeline(scale, RidgeCV(alphas=ALPHA_GRID))
    if name == "pca-ridge":
        k = _n_components(n_components, n_samples, n_features)
        return make_pipeline(scale, PCA(n_components=k, random_state=seed),
                             RidgeCV(alphas=ALPHA_GRID))
    if name == "pls":
        k = _n_components(n_components, n_samples, n_features)
        return make_pipeline(scale, PLSRegression(n_components=k))
    if name == "rf":
        return make_pipeline(
            scale, RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                         n_jobs=-1, random_state=seed))
    if name == "gbt":
        # HistGradientBoosting is single-output; gaze is two.
        return make_pipeline(
            scale, MultiOutputRegressor(
                HistGradientBoostingRegressor(max_iter=200, random_state=seed)))
    if name == "svr":
        # SVR is single-output too.
        return make_pipeline(scale, MultiOutputRegressor(SVR()))
    if name == "lgbm":
        return make_pipeline(
            scale, MultiOutputRegressor(
                lgb.LGBMRegressor(random_state=seed, verbosity=-1)))
    if name == "mlp":
        # MLPRegressor is natively multi-output, unlike the three above it.
        return make_pipeline(scale, MLPRegressor(random_state=seed, max_iter=500))
    raise ValueError(f"unknown readout {name!r}; known: {', '.join(ALL_READOUTS)}")


def fit_readout(name, x, y, n_components=32, seed=0):
    """Fit one readout on ``x [N, D]`` -> ``y [N, 2]``. None if unfittable."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or len(x) < 3 or len(x) != len(y):
        return None

    model = build_readout(name, len(x), x.shape[1], n_components, seed)
    model.fit(x, y)
    return model


def predict(model, x):
    """Predict, normalising shape.

    PLSRegression returns ``[N, 2]`` like everything else, but a DummyRegressor
    fitted on 2-column y also does -- except older sklearn returns ``[N]`` for
    single-output. Reshape defensively so downstream metric code never has to.
    """
    pred = np.asarray(model.predict(np.asarray(x, dtype=np.float64)))
    if pred.ndim == 1:
        pred = pred[:, None]
    return pred
