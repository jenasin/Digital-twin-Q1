"""Zero-shot time-series foundation model (Chronos-Bolt) as a virtual CBT sensor.

Valid CBT readings are passed through; every run of missing minutes is filled with a zero-shot
forecast from the cow's preceding CBT history (5-min resolution, up to 512 steps = 42.7 h of
context). Gaps longer than 24 h repeat the last forecast day (diurnal profile).

    python -m dtq1.tsfm        -> results/raw/tsfm.pkl  (hourly CBT per cow and CBT condition)
"""
import pickle
import warnings
import zlib

import numpy as np
import torch

from . import config as C
from .data import _ffill, load_streams
from .degrade import KINDS, degrade

H = C.N_MIN // 60
AGG = 5                     # minutes per forecasting step
MAX_STEPS = 288             # 24 h


def _pipeline():
    from chronos import BaseChronosPipeline
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=dev, torch_dtype=torch.float32)


def fill_cbt(pipe, cbt):
    """cbt: (N_MIN,) with NaN gaps -> gap-filled series."""
    out = cbt.copy()
    miss = np.isnan(cbt)
    if not miss.any():
        return out
    starts = np.where(miss & ~np.r_[False, miss[:-1]])[0]
    ends = np.where(miss & ~np.r_[miss[1:], False])[0] + 1
    ctxs, specs = [], []
    for s, e in zip(starts, ends):
        hist = cbt[:s]
        n5 = len(hist) // AGG
        if n5 < 12:                                     # < 1 h of history: nothing to condition on
            out[s:e] = np.nanmean(cbt) if (~miss).any() else 38.5
            continue
        h5 = hist[len(hist) - n5 * AGG:].reshape(n5, AGG)[-512:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ctx = np.nanmean(h5, 1)
        ctxs.append(torch.tensor(ctx, dtype=torch.float32))
        specs.append((s, e))
    if not ctxs:
        return out
    steps = min(MAX_STEPS, max(int(np.ceil((e - s) / AGG)) for s, e in specs))
    L = max(len(c) for c in ctxs)
    batch = torch.full((len(ctxs), L), float("nan"))
    for i, c in enumerate(ctxs):
        batch[i, L - len(c):] = c
    med = []
    for i in range(0, len(batch), 32):
        q, _ = pipe.predict_quantiles(batch[i:i + 32], prediction_length=steps, quantile_levels=[0.5])
        med.append(q[..., 0].cpu().numpy())
    med = np.concatenate(med)
    for (s, e), f in zip(specs, med):
        fm = np.repeat(f, AGG)                          # back to 1-min resolution
        g = e - s
        if g > len(fm):
            day = fm[-1440:] if len(fm) >= 1440 else fm
            fm = np.concatenate([fm, np.resize(day, g - len(fm))])
        out[s:e] = fm[:g]
    return out


def main():
    pipe = _pipeline()
    grid = [("none", "none", 0)] + [("cbt", k, lv) for k in KINDS for lv in range(1, 5)]
    res = {"grid": grid, "hourly": {}}
    for cow in C.COWS:
        arr = np.zeros((len(grid), 2, H), np.float32)   # [chronos, carry-forward]
        for gi, (target, kind, level) in enumerate(grid):
            s = degrade(load_streams(cow), target, kind, level, seed=zlib.crc32(f"{cow}|{target}|{kind}|{level}".encode()))
            cbt = s["cbt"][:, 0].astype(np.float64)
            filled = fill_cbt(pipe, cbt)
            ff, _ = _ffill(cbt[:, None])
            ff = np.where(np.isnan(ff[:, 0]), np.nanmean(cbt) if (~np.isnan(cbt)).any() else 38.5, ff[:, 0])
            arr[gi, 0] = filled[: H * 60].reshape(H, 60).mean(1)
            arr[gi, 1] = ff[: H * 60].reshape(H, 60).mean(1)
        res["hourly"][cow] = arr
        print(cow, "done", flush=True)
    with open(C.RESULTS / "raw" / "tsfm.pkl", "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()
