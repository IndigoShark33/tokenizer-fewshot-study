"""The statistics of analysis.py as importable functions (identical code), for analysis_belebele.py.
analysis.py keeps its own copies so the v2 results reproduce from it unchanged."""
import numpy as np
from scipy.stats import norm, rankdata, spearmanr


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy, rxz, ryz = spearmanr(rx, ry)[0], spearmanr(rx, rz)[0], spearmanr(ry, rz)[0]
    with np.errstate(invalid="ignore", divide="ignore"):   # undefined (nan) if a variable is perfectly rank-correlated with z
        return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def boot_ci(x, y, n=10000):
    rng = np.random.default_rng(0); idx = np.arange(len(x)); cs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        cs.append(spearmanr(np.array(x)[s], np.array(y)[s])[0])
    return np.nanpercentile(cs, [2.5, 97.5]).round(3)


def perm_p(x, y, z, n=5000):
    obs = partial_spearman(x, y, z)
    rng = np.random.default_rng(0); yv = np.array(y); cnt = 0; valid = 0
    for _ in range(n):
        r = partial_spearman(x, rng.permutation(yv), z)
        if np.isnan(r): continue
        valid += 1; cnt += abs(r) >= abs(obs)
    return obs, (cnt + 1) / (valid + 1)


def min_detectable(n, k=0, power=0.80, alpha=0.05):
    zc = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(np.tanh(zc * np.sqrt(1.06 / (n - 3 - k))))
