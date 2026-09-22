"""Sensor degradation operators applied to native-rate streams.

Every operator works on the stream at its native sampling period so that the effect on the
per-minute features is the same as a physically degraded sensor would produce.

kinds   : noise, dropout, drift, delay, failure, stuck
levels  : 0 (clean) .. 4 (most severe)
targets : one modality from config.MODALITIES, or "all"
"""
import numpy as np

from . import config as C

# native period (s) of each modality stream
PERIOD = {"imu": 1, "uwb": 15, "cbt": 60, "ankle": 60, "thi": 60}

KINDS = ["noise", "dropout", "drift", "delay", "failure", "stuck"]
LEVELS = {
    "noise": [0, 0.25, 0.5, 1.0, 2.0],      # sigma, multiples of channel SD
    "dropout": [0, 0.10, 0.25, 0.50, 0.75],  # missing fraction (bursty, mean burst 30 min)
    "drift": [0, 0.25, 0.5, 1.0, 2.0],       # bias reached at end of deployment, multiples of channel SD
    "delay": [0, 5, 15, 60, 180],            # minutes of transmission / processing lag
    "failure": [0, 0.10, 0.25, 0.50, 1.0],   # fraction of deployment after sensor dies (NaN)
    "stuck": [0, 0.10, 0.25, 0.50, 1.0],     # fraction of deployment with frozen last value
}
BURST_MIN = 30

# Channel SDs used to scale noise/drift (filled by channel_scales()).
_SCALES = {}


def channel_scales():
    """Per-channel SD of each modality over all cows (clean data)."""
    if _SCALES:
        return _SCALES
    from .data import load_streams
    acc = {m: [] for m in C.MODALITIES}
    for cow in C.COWS:
        s = load_streams(cow)
        for m in C.MODALITIES:
            acc[m].append(s[m])
    for m in C.MODALITIES:
        _SCALES[m] = np.nanstd(np.concatenate(acc[m]), axis=0).astype(np.float32)
    return _SCALES


def thi_formula(t_c, rh):
    """NRC (1971) THI, as used for dairy heat stress."""
    return (1.8 * t_c + 32) - (0.55 - 0.0055 * rh) * (1.8 * t_c - 26)


def _burst_mask(n_min, frac, rng, burst=BURST_MIN):
    """Two-state Markov (Gilbert) missingness at minute resolution with given stationary loss fraction."""
    if frac <= 0:
        return np.zeros(n_min, bool)
    p_bg = 1.0 / burst                       # bad -> good
    p_gb = frac * p_bg / (1 - frac)          # good -> bad
    mask = np.zeros(n_min, bool)
    t, bad = 0, rng.random() < frac
    while t < n_min:
        run = rng.geometric(p_bg if bad else p_gb)
        if bad:
            mask[t:t + run] = True
        t += run
        bad = not bad
    return mask


def _expand(mask_min, period):
    """Minute-level mask -> native-rate mask."""
    return np.repeat(mask_min, 60 // period)


def _apply(x, m, kind, level, rng, scale):
    """Degrade a single stream x (n, ch) of modality m. Returns a new array."""
    x = x.copy()
    n = len(x)
    per = PERIOD[m]
    v = LEVELS[kind][level]
    if v == 0:
        return x
    if m == "imu":
        cols = slice(0, 3)                 # means carry posture; std columns handled for noise
    else:
        cols = slice(0, x.shape[1])

    if kind == "noise":
        sig = v * scale
        if m == "imu":
            # white noise on 10 Hz samples: per-second mean gets sigma/sqrt(n), per-second SD inflates
            nn = np.clip(np.nan_to_num(x[:, 6], nan=10), 1, None)[:, None]
            x[:, 0:3] += rng.normal(0, 1, (n, 3)) * sig[0:3] / np.sqrt(nn)
            x[:, 3:6] = np.sqrt(x[:, 3:6] ** 2 + sig[0:3] ** 2)
        elif m == "thi":
            x[:, 0] += rng.normal(0, sig[0], n)
            x[:, 1] = np.clip(x[:, 1] + rng.normal(0, sig[1], n), 0, 100)
            x[:, 2] = thi_formula(x[:, 0], x[:, 1])
        else:
            x[:, cols] += rng.normal(0, 1, x[:, cols].shape) * sig[cols]
    elif kind == "drift":
        ramp = np.linspace(0, 1, n)[:, None]
        sign = rng.choice([-1, 1], size=x.shape[1])
        if m == "cbt":
            sign[:] = 1                    # conservative: a warming bolus/logger creates false alerts
        if m == "thi":
            sign[:] = -1                   # a reading-low THI sensor suppresses heat-abatement decisions
            x[:, 0] += sign[0] * v * scale[0] * ramp[:, 0]
            x[:, 1] = np.clip(x[:, 1] + sign[1] * v * scale[1] * ramp[:, 0], 0, 100)
            x[:, 2] = thi_formula(x[:, 0], x[:, 1])
        else:
            x[:, cols] += (sign * v * scale)[cols] * ramp
    elif kind == "dropout":
        mask = _expand(_burst_mask(n * per // 60, v, rng), per)[:n]
        x[mask] = np.nan
    elif kind == "delay":
        k = int(v * 60 // per)
        x[k:] = x[:n - k].copy()
        x[:k] = np.nan
    elif kind == "failure":
        x[int(n * (1 - v)):] = np.nan
    elif kind == "stuck":
        t0 = int(n * (1 - v))
        valid = np.where(~np.isnan(x[:max(t0, 1), 0]))[0]
        last = x[valid[-1]] if len(valid) else np.nanmean(x, 0)
        x[t0:] = last
    return x


def degrade(streams, target, kind, level, seed):
    """Return a degraded copy of the stream dict. `target` = modality name or 'all'."""
    if level == 0:
        return streams
    scales = channel_scales()
    out = dict(streams)
    mods = C.MODALITIES if target == "all" else [target]
    for i, m in enumerate(mods):
        rng = np.random.default_rng([seed, i, KINDS.index(kind), level])
        out[m] = _apply(streams[m], m, kind, level, rng, scales[m])
    return out


def conditions(include_clean=True):
    """Full experimental grid of (target, kind, level)."""
    grid = [("none", "none", 0)] if include_clean else []
    for target in C.MODALITIES + ["all"]:
        for kind in KINDS:
            for level in range(1, 5):
                grid.append((target, kind, level))
    return grid
