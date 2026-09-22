"""Turn raw evaluation outputs into tidy tables of state error, decision error and abstention.

    python -m dtq1.analysis            -> results/state.csv, results/decisions.csv, results/abstention.csv
"""
import glob
import pickle

import numpy as np
import pandas as pd

from . import config as C
from .decisions import RULES, compare, decide, quantities

# which modalities a rule's inputs depend on (used by the sensor-health gate)
RULE_MODS = {"heat": ["cbt"], "lying": ["ankle", "imu", "uwb"], "feed": ["imu", "uwb"], "cool": ["thi"]}
MOD_IDX = {m: i for i, m in enumerate(C.MODALITIES)}
FIRST_HOUR = 96          # every rule has its full look-back window from here on


def load_raw():
    """One merged result dict per fold (LightGBM and neural estimators are evaluated separately)."""
    out = []
    for k in range(len(C.FOLDS)):
        parts = sorted(glob.glob(str(C.RESULTS / "raw" / f"fold{k}_*.pkl")))
        if not parts:
            continue
        res = None
        for f in parts:
            with open(f, "rb") as fh:
                r = pickle.load(fh)
            if res is None:
                res = r
            else:
                for key in ("hourly", "metrics", "labeled"):
                    res[key].update(r[key])
        out.append(res)
    return out


def hourly_state(hr, thi):
    """hr: (H, 3) -> dict of hourly quantities used by the decision layer."""
    return {"lying_min": hr[:, 0], "feed_min": hr[:, 1], "cbt": hr[:, 2], "thi": thi}


def health_gate(health, rule, window=24, min_avail=0.25, max_bad=3):
    """Abstain when an input modality of the rule is unhealthy.

    health: (n_mod, H, 2) availability fraction and flat-line flag. An hour is unhealthy when fewer
    than 25 % of the expected samples arrived (0.2 % of clean cow-hours; UWB and IMU lose packets
    naturally) or the signal is flat. Hourly rules abstain on an unhealthy hour; 24-h
    rules abstain when >= 3 of the last 24 hours were unhealthy.
    """
    bad = np.zeros(health.shape[1], bool)
    for m in RULE_MODS[rule]:
        h = health[MOD_IDX[m]]
        if m == "imu" and np.all(h[:, 0] == 0):
            continue                                    # modality absent from this build
        bad |= (h[:, 0] < min_avail) | (h[:, 1] > 0)
    if rule in ("heat", "cool"):
        return bad
    c = np.concatenate([[0], np.cumsum(bad)])
    idx = np.arange(len(bad))
    return (c[idx + 1] - c[np.maximum(idx + 1 - window, 0)]) >= max_bad


def tables(raws):
    st_rows, dec_rows, ab_rows = [], [], []
    for res in raws:
        grid = res["grid"]
        keys = sorted({k for (k, cow) in res["hourly"]})
        for cow in res["cows"]:
            tr = res["truth"][cow]
            oracle = decide({"lying_min": tr["lying_min"], "feed_min": np.full_like(tr["cbt"], np.nan),
                             "cbt": tr["cbt"], "thi": res["thi"][cow][0]})
            for key in keys:
                hr = res["hourly"][(key, cow)]
                M = hr.shape[1] - 1 if hr.shape[1] > 1 else 1
                ref_state = hourly_state(hr[0, -1], res["thi"][cow][0])
                ref_q = quantities(ref_state)
                ref = decide(ref_state)
                for gi, (target, kind, level) in enumerate(grid):
                    thi = res["thi"][cow][gi]
                    st = hourly_state(hr[gi, -1], thi)
                    q = quantities(st)
                    dec = decide(st)
                    base = dict(arch=key[0], regime=key[1], cow=cow, target=target, kind=kind, level=level)
                    if key[0] != "direct":
                        met = res["metrics"][(key, cow, gi)]
                        st_rows.append({**base, **met})
                    hsel = slice(FIRST_HOUR, None)
                    # member-level decisions for ensemble disagreement
                    if hr.shape[1] > 1:
                        mem = [decide(hourly_state(hr[gi, m], thi)) for m in range(M)]
                    else:
                        mem = [dec]
                    for r in RULES:
                        if r == "feed" and key[0] == "direct":
                            continue
                        dq = np.abs(q[r][0] - ref_q[r][0])[hsel]
                        row = {**base, "rule": r, "state_dev": np.nanmean(dq)}
                        cmp = compare(ref[r][hsel], dec[r][hsel])
                        row.update(cmp)
                        if r != "feed":
                            oc = compare(oracle[r][hsel], dec[r][hsel])
                            row.update({f"oracle_{k}": v for k, v in oc.items() if k in ("disagree", "unsafe", "miss_rate", "regret")})
                        dec_rows.append(row)
                        # abstention policies
                        votes = np.nanmean([m_[r] for m_ in mem], 0)
                        ens = (votes > 0) & (votes < 1)
                        gate = health_gate(res["health"][cow][gi], r)
                        for pol, ab in [("none", None), ("ensemble", ens), ("health", gate), ("ensemble+health", ens | gate)]:
                            c = compare(ref[r][hsel], dec[r][hsel], None if ab is None else ab[hsel])
                            ab_rows.append({**base, "rule": r, "policy": pol, **c})
    return pd.DataFrame(st_rows), pd.DataFrame(dec_rows), pd.DataFrame(ab_rows)


def tsfm_table(raws, arch=None):
    """Hourly CBT error and heat-decision error vs ground truth for CBT degradations:
    Chronos-Bolt gap filling, carry-forward, and the learned twin estimators."""
    f = C.RESULTS / "raw" / "tsfm.pkl"
    if not f.exists():
        return pd.DataFrame()
    with open(f, "rb") as fh:
        ts = pickle.load(fh)
    rows = []
    thr = 39.0
    hsel = slice(FIRST_HOUR, None)
    for res in raws:
        gidx = {g: i for i, g in enumerate(res["grid"])}
        for cow in res["cows"]:
            truth = res["truth"][cow]["cbt"][hsel]
            for ti, (target, kind, level) in enumerate(ts["grid"]):
                gi = gidx[(target, kind, level)]
                cands = {"chronos": ts["hourly"][cow][ti, 0], "carry": ts["hourly"][cow][ti, 1]}
                for (a, reg), c in [(k, c) for (k, c) in res["hourly"] if c == cow and k[0] not in ("direct",)]:
                    cands[f"{a}_{reg}"] = res["hourly"][((a, reg), cow)][gi, -1, :, 2]
                for meth, v in cands.items():
                    v = v[hsel]
                    ok = ~np.isnan(truth) & ~np.isnan(v)
                    ref, pred = (truth >= thr), (v >= thr)
                    rows.append(dict(method=meth, cow=cow, kind=kind, level=level,
                                     cbt_mae=np.abs(v - truth)[ok].mean(),
                                     heat_disagree=(ref != pred)[ok].mean(),
                                     heat_missed=(ref & ~pred)[ok].sum() / max(ref[ok].sum(), 1)))
    return pd.DataFrame(rows)


def main():
    raws = load_raw()
    st, dec, ab = tables(raws)
    tsfm_table(raws).to_csv(C.RESULTS / "tsfm.csv", index=False)
    C.RESULTS.mkdir(exist_ok=True, parents=True)
    st.to_csv(C.RESULTS / "state.csv", index=False)
    dec.to_csv(C.RESULTS / "decisions.csv", index=False)
    ab.to_csv(C.RESULTS / "abstention.csv", index=False)
    print(len(st), len(dec), len(ab))


if __name__ == "__main__":
    main()
