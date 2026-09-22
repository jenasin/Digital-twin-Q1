"""Train twin state estimators for one cross-validation fold.

    python -m dtq1.train --fold 0 --arch gru --regime clean
regime 'clean' : trained on clean sensor data only (conventional practice)
regime 'aug'   : degradation-aware training (clean + randomly degraded copies of every training cow)
"""
import argparse
import os
import pickle
import time

import numpy as np

from . import config as C
from .data import fit_norm, load_streams, observation, targets
from .degrade import KINDS, degrade

N_AUG = 12
MODELS = C.ROOT / "models"


def fold_split(k):
    test = C.FOLDS[k]
    return [c for c in C.COWS if c not in test], test


def cbt_norm(cows):
    v = np.concatenate([targets(c)[2] for c in cows])
    return float(np.nanmean(v)), float(np.nanstd(v))


def training_set(k, regime):
    """Observations obs[version][cow] and targets for the training cows of fold k."""
    train, _ = fold_split(k)
    norm = fit_norm(train)
    cmu, csd = cbt_norm(train)
    versions = [[observation(load_streams(c), norm) for c in train]]
    if regime == "aug":
        # stratified coverage: every (sensor, fault type) combination appears ~2.7 times across the
        # 12 x 8 degraded cow-copies, each with a random severity
        rng = np.random.default_rng(1000 + k)
        combos = [(t, kd) for t in C.MODALITIES + ["all"] for kd in KINDS]
        order = rng.permutation(len(combos))
        for v in range(N_AUG):
            obs = []
            for ci, c in enumerate(train):
                target, kind = combos[order[(v * len(train) + ci) % len(combos)]]
                level = int(rng.integers(1, 5))
                obs.append(observation(degrade(load_streams(c), target, kind, level, seed=10_000 + 97 * v + ci), norm))
            versions.append(obs)
    tgt = []
    for c in train:
        lab, ly, cb = targets(c)
        tgt.append((lab, ly, ((cb - cmu) / csd).astype(np.float32)))
    return versions, tgt, norm, (cmu, csd)


def class_weights(tgt):
    lab = np.concatenate([t[0] for t in tgt])
    cnt = np.bincount(lab[lab >= 0], minlength=len(C.BEHAVIOURS)).astype(float)
    w = 1.0 / np.sqrt(np.maximum(cnt, 1))
    return w / w.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--arch", required=True, choices=["lgbm", "gru", "transformer", "ssm"])
    ap.add_argument("--regime", required=True, choices=["clean", "aug"])
    ap.add_argument("--seeds", type=int, nargs="*", default=C.SEEDS)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()

    out = MODELS / f"fold{a.fold}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    obs, tgt, norm, cnorm = training_set(a.fold, a.regime)
    meta = {"norm": norm, "cbt_norm": cnorm}
    with open(out / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"[fold{a.fold} {a.arch} {a.regime}] data ready {time.time() - t0:.0f}s", flush=True)
    cw = class_weights(tgt)
    if os.environ.get("DTQ1_DEVICE") == "cpu":
        import torch
        torch.set_num_threads(int(os.environ.get("DTQ1_THREADS", "4")))
    for seed in a.seeds:
        path = out / f"{a.arch}_{a.regime}_s{seed}"
        # skip finished seeds; an exclusive lock file lets several worker pools share the job list
        if os.path.exists(f"{path}.pt") or os.path.exists(f"{path}.pkl"):
            continue
        try:
            os.close(os.open(f"{path}.lock", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            print(f"[fold{a.fold} {a.arch} {a.regime}] seed {seed} taken by another worker", flush=True)
            continue
        if a.arch == "lgbm":
            from .gbm import LGBTwin
            m = LGBTwin(seed).fit(obs, tgt, seed)
            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(m, f)
        else:
            import torch
            from .models import train_nn
            net = train_nn(a.arch, obs, tgt, seed, class_w=cw, epochs=a.epochs, log=lambda s: print(s, flush=True))
            torch.save(net.state_dict(), f"{path}.pt")
        os.remove(f"{path}.lock")
        print(f"[fold{a.fold} {a.arch} {a.regime}] seed {seed} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
