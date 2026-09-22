"""Gradient-boosting twin state estimator (tabular baseline).

Kept free of torch: LightGBM and torch ship separate OpenMP runtimes that crash intermittently
when loaded into the same process on macOS.
"""
import lightgbm as lgb
import numpy as np


def lag_features(obs):
    """Per-minute features plus causal rolling means (5, 15, 60 min) for the tabular baseline."""
    cs = np.cumsum(np.vstack([np.zeros((1, obs.shape[1])), obs]), 0)
    out = [obs]
    n = len(obs)
    for w in (5, 15, 60):
        lo = np.maximum(np.arange(1, n + 1) - w, 0)
        out.append((cs[1:] - cs[lo]) / (np.arange(1, n + 1) - lo)[:, None])
    return np.hstack(out).astype(np.float32)


class LGBTwin:
    def __init__(self, seed):
        common = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=8)
        self.beh = lgb.LGBMClassifier(class_weight="balanced", **common)
        self.ly = lgb.LGBMClassifier(**common)
        self.cb = lgb.LGBMRegressor(**common)

    def fit(self, obs, tgt, seed, max_rows=400_000):
        rng = np.random.default_rng(seed)
        X = [lag_features(o) for v in obs for o in v]
        T = [tgt[c] for _ in obs for c in range(len(obs[0]))]
        Xa = np.concatenate(X)
        y = np.concatenate([t[0] for t in T])
        ly = np.concatenate([t[1] for t in T])
        cb = np.concatenate([t[2] for t in T])
        lab = y >= 0
        self.beh.fit(Xa[lab], y[lab])
        ok = np.where(~np.isnan(ly) & ~np.isnan(cb))[0]
        sub = rng.choice(ok, min(max_rows, len(ok)), replace=False)
        self.ly.fit(Xa[sub], ly[sub])
        self.cb.fit(Xa[sub], cb[sub])
        return self

    def predict(self, obs, X=None):
        X = lag_features(obs) if X is None else X
        return self.beh.predict_proba(X), self.ly.predict_proba(X)[:, 1], self.cb.predict(X)
