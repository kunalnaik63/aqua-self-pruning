import numpy as np


def make_spirals(n_per_class=180, classes=3, noise=0.18, seed=0):
    """Small deterministic-from-seed 2-D spiral classification dataset."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * classes, 2), dtype=float)
    y = np.zeros(n_per_class * classes, dtype=int)
    for cls in range(classes):
        ix = slice(cls * n_per_class, (cls + 1) * n_per_class)
        r = np.linspace(0.03, 1.0, n_per_class)
        theta = np.linspace(cls * 4.0, (cls + 1) * 4.0, n_per_class) + rng.normal(0, noise, n_per_class)
        X[ix] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        y[ix] = cls
    order = rng.permutation(len(y))
    return X[order], y[order]


def train_test_split(X, y, test_fraction=0.25, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_test = int(round(len(y) * test_fraction))
    te, tr = order[:n_test], order[n_test:]
    return X[tr], y[tr], X[te], y[te]
