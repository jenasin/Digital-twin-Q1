"""Run the degradation grid on the held-out cows of one fold and store twin states.

    python -m dtq1.evaluate --fold 0

For every (condition, held-out cow) the degraded streams and observations are computed once and
fed to every trained estimator (arch x regime x seed). Stored per estimator:
  hourly  (n_cond, M+1, H, 3) : lying minutes, feeding minutes, CBT (degC); index M = ensemble mean
  minute metrics on the ensemble mean (behaviour on the annotated day, lying, CBT)
  labeled-day probabilities (ensemble mean) for pooled calibration analysis
plus the condition-dependent observed THI (hourly) shared by all estimators.
"""
import argparse
import pickle
import time
import warnings
import zlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from . import config as C
from .data import _ffill, load_streams, observation, targets
from .degrade import conditions, degrade

ARCHS = ["lgbm", "gru", "transformer", "ssm"]
REGIMES = ["clean", "aug"]
H = C.N_MIN // 60
OUT = C.RESULTS / "raw"


def hourly(x):
    return x[: H * 60].reshape(H, 60)


def load_members(k, arch, regime, d_in):
    """LightGBM and torch must not share a process (see gbm.py): callers pass one family only."""
    d = C.ROOT / "models" / f"fold{k}"
    ms = []
    for s in C.SEEDS:
        f = d / f"{arch}_{regime}_s{s}.{'pkl' if arch == 'lgbm' else 'pt'}"
        while not f.exists():                  # a parallel worker pool may still be training it
            time.sleep(30)
        if arch == "lgbm":
            with open(d / f"lgbm_{regime}_s{s}.pkl", "rb") as f:
                m = pickle.load(f)
            for est in (m.beh, m.ly, m.cb):
                est.set_params(n_jobs=2)           # several evaluations share the CPU
            ms.append(m)
        else:
            import torch
            from .models import TwinNet
            net = TwinNet(arch, d_in, L=30)
            net.load_state_dict(torch.load(d / f"{arch}_{regime}_s{s}.pt"))
            ms.append(net.eval())
    return ms


class DirectSensor:
    """No state estimator: decisions read sensors directly (last valid value carried forward).

    lying = per-cow threshold on the ankle accelerometer, calibrated on the first 24 h after
    installation (clean data), as done for commercial leg sensors; CBT = bolus reading; no feeding.
    """

    def __init__(self, cow):
        a = load_streams(cow)["ankle"][:1440]
        y = targets(cow)[1][:1440]
        ok = ~np.isnan(a).any(1) & ~np.isnan(y)
        self.lr = LogisticRegression(C=100).fit(a[ok], y[ok] > 0.5)

    def predict(self, s):
        ank, _ = _ffill(s["ankle"])
        cbt, _ = _ffill(s["cbt"])
        ply = np.full(len(ank), np.nan)
        ok = ~np.isnan(ank).any(1)
        if ok.any():
            ply[ok] = self.lr.predict(ank[ok])          # hard threshold, as in commercial lying sensors
        pb = np.full((len(ank), len(C.BEHAVIOURS)), np.nan)
        return pb, ply, cbt[:, 0]


def health(s):
    """Hourly sensor-health indicators an on-farm monitor can compute without ground truth.

    Returns (n_modalities, H, 2): fraction of expected samples received, flat-line flag
    (zero variance of the first channel over the hour while samples are present).
    """
    out = np.zeros((len(C.MODALITIES), H, 2), np.float32)
    for i, m in enumerate(C.MODALITIES):
        x = s[m][:, 0].astype(np.float64)
        per = len(x) // C.N_MIN
        xh = x[: H * 60 * per].reshape(H, 60 * per)
        avail = (~np.isnan(xh)).mean(1)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            flat = (np.nanstd(xh, 1) < 1e-9) & (avail >= 0.5)
        out[i, :, 0], out[i, :, 1] = avail, flat
    return out


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def state_metrics(pb, ply, cbt, lab, ly_true, cbt_true):
    m = lab >= 0
    y = lab[m]
    p = pb[m]
    onehot = np.eye(p.shape[1])[y]
    return {
        "beh_f1": f1_score(y, p.argmax(1), average="macro", labels=range(p.shape[1]), zero_division=0),
        "beh_acc": (p.argmax(1) == y).mean(),
        "beh_ece": ece(p, y),
        "beh_brier": ((p - onehot) ** 2).sum(1).mean(),
        "beh_nll": -np.log(np.clip(p[np.arange(len(y)), y], 1e-6, 1)).mean(),
        "ly_acc": ((ply >= 0.5) == (ly_true > 0.5)).mean(),
        "ly_brier": ((ply - ly_true) ** 2).mean(),
        "cbt_mae": np.nanmean(np.abs(cbt - cbt_true)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--archs", nargs="*", default=ARCHS)
    ap.add_argument("--regimes", nargs="*", default=REGIMES)
    ap.add_argument("--seeds", type=int, nargs="*", default=C.SEEDS)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first n conditions")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    C.SEEDS[:] = a.seeds
    if "lgbm" in a.archs and len(a.archs) > 1:
        ap.error("evaluate lgbm and neural archs in separate processes (OpenMP clash)")
    if "lgbm" in a.archs:
        from .gbm import lag_features
    else:
        from .models import predict_nn
    k = a.fold
    done_file, marker = OUT / f"fold{k}{a.tag}.pkl", OUT / f"fold{k}{a.tag}.running"
    if not a.limit and done_file.exists():
        print(f"fold{k}{a.tag} already evaluated", flush=True)
        return
    if not a.limit and marker.exists():            # another process is evaluating this fold: wait for it
        while not done_file.exists():
            time.sleep(60)
        print(f"fold{k}{a.tag} evaluated by another process", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.limit:
        marker.touch()
    with open(C.ROOT / "models" / f"fold{k}" / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    norm, (cmu, csd) = meta["norm"], meta["cbt_norm"]
    grid = conditions()
    if a.limit:
        grid = grid[: a.limit]
    test = C.FOLDS[k]
    d_in = observation(load_streams(test[0]), norm).shape[1]
    members = {(ar, rg): load_members(k, ar, rg, d_in) for ar in a.archs for rg in a.regimes}
    M = len(C.SEEDS)
    res = {"grid": grid, "cows": test, "thi": {}, "health": {}, "truth": {}, "hourly": {}, "metrics": {}, "labeled": {}}
    t0 = time.time()
    for cow in test:
        lab, ly_true, cbt_true = targets(cow)
        res["truth"][cow] = {"lying_min": hourly(ly_true).sum(1), "cbt": hourly(cbt_true).mean(1),
                             "label": lab, "lying": ly_true}
        res["thi"][cow] = np.zeros((len(grid), H), np.float32)
        direct = DirectSensor(cow)
        res["health"][cow] = np.zeros((len(grid), len(C.MODALITIES), H, 2), np.float32)
        res["hourly"][(("direct", "-"), cow)] = np.zeros((len(grid), 1, H, 3), np.float32)
        for key in members:
            res["hourly"][(key, cow)] = np.zeros((len(grid), M + 1, H, 3), np.float32)
            res["labeled"][(key, cow)] = np.zeros((len(grid), int((lab >= 0).sum()), len(C.BEHAVIOURS)), np.float16)
        for gi, (target, kind, level) in enumerate(grid):
            s = degrade(load_streams(cow), target, kind, level, seed=zlib.crc32(f"{cow}|{target}|{kind}|{level}".encode()))
            thi_ff, _ = _ffill(s["thi"][:, 2:3])
            res["thi"][cow][gi] = np.nanmean(hourly(thi_ff[:, 0]), 1)
            res["health"][cow][gi] = health(s)
            obs = observation(s, norm)
            X = lag_features(obs) if "lgbm" in a.archs else None
            _, ply, cb = direct.predict(s)
            hd = res["hourly"][(("direct", "-"), cow)]
            hd[gi, 0, :, 0] = hourly(ply).sum(1)
            hd[gi, 0, :, 1] = np.nan
            hd[gi, 0, :, 2] = hourly(cb).mean(1)
            for key, ms in members.items():
                outs = []
                for m in ms:
                    if key[0] == "lgbm":
                        pb, ply, cb = m.predict(obs, X)
                    else:
                        pb, ply, cb = predict_nn(m, obs)
                    outs.append((pb, ply, cb * csd + cmu))
                pb = np.mean([o[0] for o in outs], 0)
                ply = np.mean([o[1] for o in outs], 0)
                cb = np.mean([o[2] for o in outs], 0)
                hr = res["hourly"][(key, cow)]
                for mi, o in enumerate(outs + [(pb, ply, cb)]):
                    hr[gi, mi, :, 0] = hourly(o[1]).sum(1)
                    hr[gi, mi, :, 1] = hourly(o[0][:, 2]).sum(1)
                    hr[gi, mi, :, 2] = hourly(o[2]).mean(1)
                res["metrics"][(key, cow, gi)] = state_metrics(pb, ply, cb, lab, ly_true, cbt_true)
                res["labeled"][(key, cow)][gi] = pb[lab >= 0]
            if gi % 10 == 0:
                print(f"fold{k} {cow} cond {gi}/{len(grid)} {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"fold{k}{a.tag}.pkl", "wb") as f:
        pickle.dump(res, f)
    marker.unlink(missing_ok=True)
    print(f"fold{k} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
