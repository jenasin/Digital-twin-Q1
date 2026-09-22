#!/usr/bin/env bash
# Inventory of the pipeline: what is running, which artifacts exist, which are stale,
# and what each log last said. Writes logs/status.md and prints the same to stdout.
cd "$(dirname "$0")/.."
python3 - "$@" <<'PY'
import os, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

R = Path(".").resolve()
now = time.time()
out = []
def w(s=""): out.append(s)
def age(p):
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")
def size(p):
    n = p.stat().st_size
    for u in ("B", "K", "M", "G"):
        if n < 1024: return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.0f}T"

w(f"# Pipeline status — {datetime.now():%Y-%m-%d %H:%M}")

# ---------------------------------------------------------------- what is running
w("\n## Běží\n")
ps = subprocess.run(["ps", "-axo", "pid,etime,command"], capture_output=True, text=True).stdout
live = [l for l in ps.splitlines() if "dtq1." in l or "tectonic" in l or "resume.sh" in l]
live = [l for l in live if "grep" not in l and "ps -axo" not in l]
if live:
    for l in live:
        pid, et, cmd = l.split(None, 2)
        cmd = re.sub(r"^\S*python\S*\s+", "", cmd)[:70]
        w(f"- `{pid}` {et:>12}  {cmd}")
else:
    w("- nic neběží")

# ---------------------------------------------------------------- evaluation grid progress
w("\n## Evaluační mřížka (145 podmínek × 2 held-out krávy × fold)\n")
w("| fold | lgbm | nn | poslední záznam v nn logu |")
w("|---|---|---|---|")
for k in range(5):
    row = []
    for tag in ("_lgbm", "_nn"):
        pkl = R / f"results/raw/fold{k}{tag}.pkl"
        run = R / f"results/raw/fold{k}{tag}.running"
        row.append("hotovo" if pkl.exists() else ("běží" if run.exists() else "chybí"))
    log = R / f"logs/eval_f{k}_nn.log"
    last, note = "—", ""
    if log.exists():
        conds = [l for l in log.read_text(errors="ignore").splitlines() if " cond " in l]
        if conds: last = conds[-1].strip()
        if not (R / f"results/raw/fold{k}_nn.pkl").exists() and not (R / f"results/raw/fold{k}_nn.running").exists():
            note = "  ⚠ log je z minulého běhu"
        if "failed assertion" in log.read_text(errors="ignore"):
            note += "  ⚠ Metal assertion"
    w(f"| {k} | {row[0]} | {row[1]} | `{last}`{note} |")

# ---------------------------------------------------------------- artifacts and staleness
w("\n## Artefakty\n")
CHAIN = [
    ("raw výsledky",   ["results/raw/fold*_nn.pkl", "results/raw/fold*_lgbm.pkl", "results/raw/tsfm.pkl"], []),
    ("tidy tabulky",   ["results/state.csv", "results/decisions.csv", "results/abstention.csv", "results/tsfm.csv"], ["results/raw/fold*.pkl"]),
    # figures.py writes the tables and the figures in one pass, so they are one layer:
    # comparing them against each other only measures the order of writes inside that process
    ("tabulky + obrázky", ["results/table_*.csv", "figures/fig[3-9]*.pdf"],
                          ["results/state.csv", "results/decisions.csv", "results/abstention.csv"]),
    ("LaTeX vstupy",   ["paper/jds/numbers.tex", "paper/jds/tables.tex", "paper/jds/figures.tex"], ["results/table_*.csv", "results/state.csv", "results/decisions.csv"]),
    ("manuskript",     ["paper/jds/manuscript.pdf", "paper/jds/manuscript.docx"], ["paper/jds/manuscript.tex", "paper/jds/numbers.tex", "paper/jds/results.tex", "paper/jds/discussion.tex", "paper/jds/conclusions.tex"]),
    ("submission",     ["paper/jds/submission_figures/Figure*.tiff"], ["figures/fig[3-9]*.tiff"]),
]
def newest(pats):
    ts = [p.stat().st_mtime for pat in pats for p in R.glob(pat) if p.exists()]
    return max(ts) if ts else 0

w("| vrstva | souborů | nejnovější | stav |")
w("|---|---|---|---|")
upstream = 0.0                      # newest mtime anywhere above the current layer
for name, pats, deps in CHAIN:
    files = sorted({p for pat in pats for p in R.glob(pat)})
    if not files:
        w(f"| {name} | 0 | — | **chybí** |"); continue
    oldest_t = min(p.stat().st_mtime for p in files)
    dep_t = max(newest(deps) if deps else 0, upstream)
    upstream = max(upstream, max(p.stat().st_mtime for p in files))
    state = "zastaralé" if dep_t and oldest_t < dep_t else "aktuální"
    mark = "⚠ **" + state + "**" if state == "zastaralé" else state
    w(f"| {name} | {len(files)} | {datetime.fromtimestamp(max(p.stat().st_mtime for p in files)):%m-%d %H:%M} | {mark} |")

# ---------------------------------------------------------------- figures with their manuscript numbers
w("\n## Obrázky a jejich čísla v článku\n")
ftex = R / "paper/jds/figures.tex"
order = re.findall(r"figures/([a-z0-9_]+)\.pdf", ftex.read_text()) if ftex.exists() else []
labels = re.findall(r"\\label\{fig:([a-z]+)\}", ftex.read_text()) if ftex.exists() else []
w("| v článku | soubor | label | pdf / png / tiff | přegenerováno |")
w("|---|---|---|---|---|")
for i, name in enumerate(order, 1):
    have = "".join("✓" if (R / f"figures/{name}.{e}").exists() else "✗" for e in ("pdf", "png", "tiff"))
    p = R / f"figures/{name}.pdf"
    lab = labels[i-1] if i <= len(labels) else "—"
    w(f"| Figure{i} | `{name}` | fig:{lab} | {have[0]} / {have[1]} / {have[2]} | {age(p) if p.exists() else '—'} |")
orphan = sorted(p.stem for p in R.glob("figures/*.pdf") if p.stem not in order)
if orphan:
    w(f"\nMimo článek: {', '.join('`'+o+'`' for o in orphan)}")

# ---------------------------------------------------------------- logs
w("\n## Logy\n")
def verdict(p):
    t = p.read_text(errors="ignore")
    if "failed assertion" in t or "Traceback" in t: return "CHYBA"
    if not t.strip(): return "prázdný"
    return "ok"
groups = {}
for p in sorted(R.glob("logs/*.log")):
    g = re.sub(r"_f\d+", "_fN", p.stem)
    groups.setdefault(g, []).append(p)
w("| skupina | ks | stav | nejnovější |")
w("|---|---|---|---|")
for g, ps in sorted(groups.items()):
    vs = [verdict(p) for p in ps]
    bad = sum(v == "CHYBA" for v in vs)
    empty = sum(v == "prázdný" for v in vs)
    parts = ([f"{bad}× chyba"] if bad else []) + ([f"{empty}× prázdný"] if empty else [])
    s = "⚠ " + ", ".join(parts) if parts else "ok"
    w(f"| `{g}` | {len(ps)} | {s} | {max(age(p) for p in ps)} |")

txt = "\n".join(out) + "\n"
(R / "logs/status.md").write_text(txt)
print(txt)
PY
