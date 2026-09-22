"""Convert MmCows raw files into clean native-rate streams on regular grids.

Output per cow  data/processed/streams/{cow}.npz :
    imu   (N_MIN*60, 7)  1 Hz: acc mean xyz, acc std xyz, n_samples      (from 10 Hz neck IMU)
    uwb   (N_MIN*4, 3)   15 s: neck position x, y, z (m)
    cbt   (N_MIN, 1)     60 s: core body temperature (degC)
    ankle (N_MIN, 3)     60 s: ankle accelerometer xyz (g)
    lying (N_MIN,)       60 s: ankle-derived lying flag (reference only, never an input)
    label (N_MIN,)       60 s: behaviour class (config.BEHAVIOURS) on the annotated day, -1 elsewhere
Shared  data/processed/streams/thi.npz :
    thi   (N_MIN, 3)     60 s: indoor temperature, relative humidity, THI
Missing samples are NaN.
"""
import glob

import numpy as np
import pandas as pd

from . import config as C

OUT = C.PROC / "streams"


def _grid(ts, values, period, n):
    """Place samples on a regular grid starting at T0 (nearest slot, last sample wins)."""
    idx = np.rint((np.asarray(ts, float) - C.T0) / period).astype(int)
    ok = (idx >= 0) & (idx < n)
    out = np.full((n, values.shape[1]), np.nan, np.float32)
    out[idx[ok]] = values[ok]
    return out


def load_imu(cow):
    """10 Hz neck accelerometer -> 1 Hz summary (mean and std per axis)."""
    n = C.N_MIN * 60
    acc = np.zeros((n, 3)); acc2 = np.zeros((n, 3)); cnt = np.zeros(n)
    for f in sorted(glob.glob(str(C.RAW / "main_data" / "immu" / C.TAG[cow] / "*.csv"))):
        d = pd.read_csv(f, usecols=["timestamp", "accel_x_mps2", "accel_y_mps2", "accel_z_mps2"])
        d = d.dropna()
        sec = np.floor(d["timestamp"].to_numpy() - C.T0).astype(int)
        ok = (sec >= 0) & (sec < n)
        sec = sec[ok]
        a = d[["accel_x_mps2", "accel_y_mps2", "accel_z_mps2"]].to_numpy()[ok]
        np.add.at(cnt, sec, 1)
        for k in range(3):
            np.add.at(acc[:, k], sec, a[:, k])
            np.add.at(acc2[:, k], sec, a[:, k] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = acc / cnt[:, None]
        std = np.sqrt(np.maximum(acc2 / cnt[:, None] - mean ** 2, 0))
    bad = cnt < 3
    mean[bad] = np.nan; std[bad] = np.nan
    return np.column_stack([mean, std, cnt]).astype(np.float32)


def load_uwb(cow):
    fs = sorted(glob.glob(str(C.RAW / "main_data" / "uwb" / C.TAG[cow] / "*.csv")))
    d = pd.concat([pd.read_csv(f) for f in fs])
    xyz = d[["coord_x_cm", "coord_y_cm", "coord_z_cm"]].to_numpy() / 100.0
    return _grid(d["timestamp"], xyz, 15, C.N_MIN * 4)


def load_cbt(cow):
    d = pd.read_csv(C.RAW / "main_data" / "cbt" / f"{cow}.csv")
    return _grid(d["timestamp"], d[["temperature_C"]].to_numpy(), 60, C.N_MIN)


def load_ankle(cow):
    d = pd.read_csv(C.RAW / "sub_data" / "ankle_accel" / f"{cow}.csv")
    acc = _grid(d["timestamp"], d[["accel_x", "accel_y", "accel_z"]].to_numpy(), 60, C.N_MIN)
    fs = sorted(glob.glob(str(C.RAW / "main_data" / "ankle" / cow / "*.csv")))
    ly = pd.concat([pd.read_csv(f) for f in fs])
    lying = _grid(ly["timestamp"], ly[["lying"]].to_numpy(), 60, C.N_MIN)[:, 0]
    return acc, lying


def load_labels(cow):
    """1 s manual behaviour labels -> per-minute majority class (>=50 % of seconds known)."""
    d = pd.read_csv(C.RAW / "behavior_labels" / "individual" / f"{cow}_{C.LABEL_DAY}.csv")
    cls = d["behavior"].map(C.RAW2CLS).fillna(-1).astype(int).to_numpy()
    minute = np.floor((d["timestamp"].to_numpy() - C.T0) / 60).astype(int)
    out = np.full(C.N_MIN, -1, np.int8)
    k = len(C.BEHAVIOURS)
    counts = np.zeros((C.N_MIN, k + 1))
    ok = (minute >= 0) & (minute < C.N_MIN)
    np.add.at(counts, (minute[ok], np.where(cls[ok] < 0, k, cls[ok])), 1)
    known = counts[:, :k].sum(1)
    sel = known >= 30
    out[sel] = counts[sel, :k].argmax(1)
    return out


def load_thi():
    d = pd.read_csv(C.RAW / "main_data" / "thi" / "average.csv")
    # column is labelled temperature_F in the release but values are degC
    return _grid(d["timestamp"], d[["temperature_F", "humidity_per", "THI"]].to_numpy(), 60, C.N_MIN)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "thi.npz", thi=load_thi())
    for cow in C.COWS:
        ankle, lying = load_ankle(cow)
        np.savez_compressed(OUT / f"{cow}.npz", imu=load_imu(cow), uwb=load_uwb(cow), cbt=load_cbt(cow),
                            ankle=ankle, lying=lying, label=load_labels(cow))
        print(cow, "done", flush=True)


if __name__ == "__main__":
    main()
