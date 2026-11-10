"""Uncertainty: Monte Carlo success probability and split-conformal cost intervals."""
import numpy as np


def monte_carlo_success(evaluate, priors, n=500, seed=0):
    """evaluate(sample: dict) -> bool. priors: {name: (mean, std)} sampled as normal."""
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n):
        s = {k: rng.normal(m, sd) for k, (m, sd) in priors.items()}
        ok += bool(evaluate(s))
    return ok / n


def conformal_halfwidth(y_cal, yhat_cal, alpha=0.1):
    """Split conformal: q = the ceil((n+1)(1-alpha))/n quantile of |y - yhat|."""
    r = np.sort(np.abs(np.asarray(y_cal) - np.asarray(yhat_cal)))
    n = len(r)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(r[min(k, n - 1)])


def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))
