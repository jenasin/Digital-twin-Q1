"""Stream loading and per-minute feature extraction (the twin's observation model)."""
import warnings
from functools import lru_cache

import numpy as np

from . import config as C

STREAMS = C.PROC / "streams"


@lru_cache(maxsize=None)
def _load(cow):
    z = np.load(STREAMS / f"{cow}.npz")
    t = np.load(STREAMS / "thi.npz")
    return {"imu": z["imu"], "uwb": z["uwb"], "cbt": z["cbt"], "ankle": z["ankle"], "thi": t["thi"],
            "lying": z["lying"], "label": z["label"]}


def load_streams(cow):
    """Clean streams for a cow (arrays are shared; degrade() copies before modifying)."""
    return dict(_load(cow))


def _block(x, k):
    """(n*k, ch) -> (n, k, ch)."""
    n = len(x) // k
    return x[: n * k].reshape(n, k, x.shape[1])


def _nanmean(b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, axis=1)


def _ffill(x):
    """Forward fill NaNs along axis 0; also returns minutes since last valid observation."""
    n = len(x)
    valid = ~np.isnan(x).any(1)
    idx = np.where(valid, np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    out = np.where(idx[:, None] >= 0, x[np.maximum(idx, 0)], np.nan)
    age = np.where(idx >= 0, np.arange(n) - idx, n).astype(np.float32)
    return out, age


# Feature groups: name -> modality (used for masking, availability and ablation)
FEATURES = []


def minute_features(s):
    """Per-minute observation vector from (possibly degraded) streams.

    Returns X (N_MIN, F) float32 with NaN for unavailable values, before carry-forward.
    """
    f = {}
    # neck IMU, 1 Hz -> minute
    imu = _block(s["imu"][:, :6], 60)
    m = _nanmean(imu)
    f["imu_ax"], f["imu_ay"], f["imu_az"] = m[:, 0], m[:, 1], m[:, 2]
    f["imu_sx"], f["imu_sy"], f["imu_sz"] = m[:, 3], m[:, 4], m[:, 5]
    f["imu_pitch"] = np.degrees(np.arctan2(m[:, 0], np.sqrt(m[:, 1] ** 2 + m[:, 2] ** 2)))
    dyn = np.sqrt((imu[:, :, 3:6] ** 2).sum(-1))
    f["imu_act"] = _nanmean(dyn[:, :, None])[:, 0]
    with np.errstate(invalid="ignore"):  # 0 * NaN keeps missing seconds missing
        f["imu_act_hi"] = _nanmean((dyn > 1.0).astype(float)[:, :, None] + 0 * dyn[:, :, None])[:, 0]
    # neck UWB, 15 s -> minute
    u = _block(s["uwb"], 4)
    um = _nanmean(u)
    f["uwb_x"], f["uwb_y"], f["uwb_z"] = um[:, 0], um[:, 1], um[:, 2]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f["uwb_zsd"] = np.nanstd(u[:, :, 2], axis=1) + 0 * um[:, 2]
        step = np.sqrt((np.diff(u[:, :, :2], axis=1) ** 2).sum(-1))
        f["uwb_step"] = _nanmean(step[:, :, None])[:, 0]
    # CBT, ankle, THI at 60 s
    f["cbt"] = s["cbt"][:, 0]
    f["ank_x"], f["ank_y"], f["ank_z"] = s["ankle"][:, 0], s["ankle"][:, 1], s["ankle"][:, 2]
    f["thi_t"], f["thi_rh"], f["thi"] = s["thi"][:, 0], s["thi"][:, 1], s["thi"][:, 2]
    names = list(f)
    X = np.column_stack([np.asarray(f[k], np.float32)[: C.N_MIN] for k in names])
    if not FEATURES:
        FEATURES.extend(names)
    return X


def modality_of(name):
    return {"imu": "imu", "uwb": "uwb", "cbt": "cbt", "ank": "ankle", "thi": "thi"}[name.split("_")[0]]


def observation(s, norm):
    """Model input: normalised carried-forward features + per-modality availability and staleness.

    norm = (mu, sd) computed on clean training data.
    """
    X = minute_features(s)
    mu, sd = norm
    cols, extra = [], []
    for m in C.MODALITIES:
        idx = [i for i, n in enumerate(FEATURES) if modality_of(n) == m]
        xm = X[:, idx]
        ff, age = _ffill(xm)
        ff = (ff - mu[idx]) / sd[idx]
        ff = np.nan_to_num(ff, nan=0.0)
        avail = (~np.isnan(xm).any(1)).astype(np.float32)
        cols.append(ff)
        extra += [avail, np.log1p(age) / 5.0]
    tod = (np.arange(C.N_MIN) * 60 + C.T0 - 5 * 3600) % 86400 / 86400.0   # local time (CDT)
    extra += [np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)]
    return np.column_stack(cols + [np.column_stack(extra)]).astype(np.float32)


def feature_names():
    names = []
    for m in C.MODALITIES:
        names += [n for n in FEATURES if modality_of(n) == m]
    for m in C.MODALITIES:
        names += [f"{m}_avail", f"{m}_age"]
    return names + ["tod_sin", "tod_cos"]


def fit_norm(cows):
    Xs = np.concatenate([minute_features(load_streams(c)) for c in cows])
    mu = np.nanmean(Xs, 0)
    sd = np.nanstd(Xs, 0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def targets(cow):
    """Twin state targets from clean data: behaviour label (-1 if none), ankle lying, CBT (degC)."""
    s = _load(cow)
    return s["label"].astype(np.int64), s["lying"].astype(np.float32), s["cbt"][:, 0].astype(np.float32)
