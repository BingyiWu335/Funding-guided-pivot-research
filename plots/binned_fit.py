import numpy as np
from scipy.stats import t


def fit_binned(x, y):
    x, y = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    assert len(x) == len(y) > 2 and np.isfinite(x).all() and np.isfinite(y).all()
    assert np.all(np.diff(x) > 0)
    coef, cov = np.polyfit(x, y, 1, cov=True)
    df = len(x) - 2
    return dict(
        coef=coef, cov=cov, df=df, critical=float(t.ppf(0.975, df)), n_bins=len(x)
    )


def predict(fit, x):
    x = np.asarray(x, dtype=float)
    X = np.column_stack([x, np.ones(len(x))])
    y = X @ fit["coef"]
    margin = fit["critical"] * np.sqrt(
        np.maximum(np.einsum("ij,jk,ik->i", X, fit["cov"], X), 0)
    )
    return (y, y - margin, y + margin)
