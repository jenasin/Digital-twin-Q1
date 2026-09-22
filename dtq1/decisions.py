"""Decision layer of the digital twin: hourly, rule-based management actions per cow.

Rules (thresholds from the literature, base rates checked on clean MmCows data):
  heat   individual heat-stress intervention : hourly mean CBT >= 39.0 degC
         (hyperthermia onset in lactating cows, e.g. Polsky & von Keyserlingk 2017, J Dairy Sci)
  lying  welfare / lameness check           : lying time in the trailing 24 h < 10 h
         (e.g. Ito et al. 2009, J Dairy Sci; Tucker et al. 2021, J Dairy Sci)
  feed   health check                       : feeding time in trailing 24 h < 80 % of the cow's
         own mean over the preceding 72 h (drop in feeding time precedes clinical disease;
         e.g. Weary et al. 2009, J Anim Sci)
  cool   barn heat-abatement (herd level)    : hourly mean THI >= 72 (Armstrong 1994, J Dairy Sci)

Every rule returns a boolean per hour; hours without a full look-back window are NaN.
"""
import numpy as np

RULES = ["heat", "lying", "feed", "cool"]
THRESH = {"heat": 39.0, "lying": 10.0, "feed": 0.80, "cool": 72.0}
# cost of a missed action relative to an unnecessary one (sensitivity analysis varies it)
COST_FN, COST_FP = 5.0, 1.0


def _roll_sum(x, w):
    """Trailing w-hour sum; NaN only for windows that contain a missing hour."""
    nan = np.isnan(x)
    c = np.concatenate([[0.0], np.cumsum(np.where(nan, 0.0, x))])
    n = np.concatenate([[0], np.cumsum(nan)])
    out = np.full(len(x), np.nan)
    out[w - 1:] = c[w:] - c[:-w]
    out[w - 1:][(n[w:] - n[:-w]) > 0] = np.nan
    return out


def quantities(hourly):
    """Decision-relevant quantities from hourly twin state.

    hourly: dict with arrays (H,) : lying_min, feed_min, cbt, thi
    Returns dict rule -> (quantity, threshold, direction) where direction=+1 means alert if q >= thr.
    """
    ly24 = _roll_sum(hourly["lying_min"], 24) / 60.0
    fd24 = _roll_sum(hourly["feed_min"], 24)
    base = np.full_like(fd24, np.nan)
    for h in range(96, len(fd24)):
        prev = fd24[h - 96:h - 24]
        base[h] = np.nanmean(prev)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = fd24 / base
    return {
        "heat": (hourly["cbt"], THRESH["heat"], +1),
        "lying": (ly24, THRESH["lying"], -1),
        "feed": (ratio, THRESH["feed"], -1),
        "cool": (hourly["thi"], THRESH["cool"], +1),
    }


def decide(hourly):
    """rule -> float array (H,) with 1 = act, 0 = no action, NaN = undefined."""
    out = {}
    for r, (q, thr, d) in quantities(hourly).items():
        a = (q >= thr) if d > 0 else (q < thr)
        out[r] = np.where(np.isnan(q), np.nan, a.astype(float))
    return out


def margin(hourly):
    """Signed distance of each quantity to its threshold, in rule units (positive = act)."""
    out = {}
    for r, (q, thr, d) in quantities(hourly).items():
        out[r] = d * (q - thr)
    return out


def compare(ref, pred, abstain=None):
    """Decision agreement metrics for one rule. ref/pred: arrays of {0,1,NaN}; abstain: bool mask.

    A prediction that cannot be computed (NaN, e.g. a dead sensor with nothing to carry forward)
    means the system stays silent: it counts as 'no action', not as a skipped hour.
    """
    pred = np.nan_to_num(pred, nan=0.0)
    ok = ~np.isnan(ref)
    if abstain is None:
        abstain = np.zeros_like(ref, bool)
    auto = ok & ~abstain
    n_ok, n_auto = ok.sum(), auto.sum()
    r, p = ref[auto], pred[auto]
    fn = ((r == 1) & (p == 0)).sum()
    fp = ((r == 0) & (p == 1)).sum()
    pos = (ref[ok] == 1).sum()
    return {
        "n": int(n_ok),
        "coverage": n_auto / max(n_ok, 1),
        "disagree": (fn + fp) / max(n_ok, 1),         # among all hours (abstained hours count as agreement-free)
        "unsafe": fn / max(n_ok, 1),                  # missed actions per decision hour
        "false_alarm": fp / max(n_ok, 1),
        "miss_rate": fn / max(pos, 1),                # missed actions / reference actions
        "regret": (COST_FN * fn + COST_FP * fp) / max(n_ok, 1),
        "ref_rate": pos / max(n_ok, 1),
    }
