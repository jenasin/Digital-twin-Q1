"""Publication figures and tables from results/*.csv.

    python -m dtq1.figures
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config as C  # noqa: E402
from .degrade import KINDS, LEVELS  # noqa: E402

# validated reference palette (categorical slots in fixed order; sequential blue ramp)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "v", "x", "+"]     # JDS-permitted symbols only
MUTED = "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ARCH_LABEL = {"lgbm": "LightGBM", "gru": "GRU", "transformer": "Transformer", "ssm": "Selective SSM", "direct": "Direct sensor"}
ARCHS = ["lgbm", "gru", "transformer", "ssm"]
RULE_LABEL = {"heat": "Heat-stress (CBT)", "lying": "Lying deficit", "feed": "Feeding drop", "cool": "Barn cooling (THI)"}
KIND_LABEL = {"noise": "Noise", "dropout": "Dropout", "drift": "Drift", "delay": "Delay", "failure": "Failure", "stuck": "Stuck-at"}
MOD_LABEL = {"imu": "Neck IMU", "uwb": "UWB position", "cbt": "Body temp.", "ankle": "Ankle accel.", "thi": "THI sensor", "all": "All sensors"}

# Journal of Dairy Science figure rules: final width 8.9 / 14 / 19 cm, type >= 8 pt (Arial/Helvetica),
# black axes and ticks >= 1 pt, lines >= 1 pt, no unnecessary grid lines, RGB, >= 300 dpi.
CM = 1 / 2.54
W1, W2, W3 = 8.9 * CM, 14 * CM, 19 * CM
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": "black", "axes.labelcolor": "black", "xtick.color": "black",
    "ytick.color": "black", "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "lines.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def boot_ci(x, n=2000, seed=0):
    """Mean and 95 % bootstrap CI over cows."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def save(fig, name):
    C.FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGS / f"{name}.png")
    fig.savefig(C.FIGS / f"{name}.pdf")
    fig.savefig(C.FIGS / f"{name}.tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def best_arch(dec):
    """Architecture with the lowest mean regret over the degradation grid (either training regime)."""
    d = dec[(dec.kind != "none") & (dec.arch.isin(ARCHS))]
    return d.groupby(["arch", "regime"]).regret.mean().idxmin()[0]


# ----------------------------------------------------------------------------- tables
def table_clean(st, dec):
    s = st[st.kind == "none"].groupby(["arch", "regime"])[["beh_f1", "beh_acc", "beh_ece", "ly_acc", "cbt_mae"]]
    t = s.mean().round(3)
    o = dec[(dec.kind == "none")].groupby(["arch", "regime", "rule"]).oracle_disagree.mean().unstack().round(3)
    t = t.join(o.add_prefix("oracle_disagree_"), how="outer")
    t.to_csv(C.RESULTS / "table_clean_performance.csv")
    return t


def table_levels():
    rows = []
    for k in KINDS:
        rows.append({"kind": k, **{f"L{i}": LEVELS[k][i] for i in range(1, 5)}})
    pd.DataFrame(rows).to_csv(C.RESULTS / "table_degradation_levels.csv", index=False)


def table_summary(dec):
    d = dec[(dec.kind != "none") & (dec.arch != "direct")]
    t = d.groupby(["regime", "kind"])[["state_dev", "disagree", "unsafe", "false_alarm", "regret"]].mean()
    t.round(4).to_csv(C.RESULTS / "table_summary_by_kind.csv")
    return t


def table_stats(dec):
    """Paired tests over cows (Wilcoxon signed-rank) on per-cow mean regret across degraded conditions."""
    from scipy.stats import wilcoxon
    d = dec[(dec.kind != "none")]
    pc = d.groupby(["arch", "regime", "cow"]).regret.mean()
    rows = []
    present = set(pc.index.get_level_values(0))
    for a in [a for a in ARCHS if a in present]:
        x, y = pc.loc[(a, "clean")], pc.loc[(a, "aug")]
        cows = x.index.intersection(y.index)
        p = wilcoxon(x[cows], y[cows]).pvalue
        rows.append(dict(comparison=f"{a}: clean vs degradation-aware", mean_a=x.mean(), mean_b=y.mean(),
                         rel_change=(y.mean() - x.mean()) / x.mean(), p=p, n=len(cows)))
    # twin vs direct sensor on the rules both can take (heat, lying, cool)
    dd = d[d.rule.isin(["heat", "lying", "cool"])]
    pcd = dd.groupby(["arch", "regime", "cow"]).regret.mean()
    xd = pcd.loc[("direct", "-")]
    for a in [a for a in ARCHS if a in present]:
        y = pcd.loc[(a, "aug")]
        cows = xd.index.intersection(y.index)
        p = wilcoxon(xd[cows], y[cows]).pvalue
        rows.append(dict(comparison=f"direct sensor vs {a} (aug), shared rules", mean_a=xd.mean(), mean_b=y.mean(),
                         rel_change=(y.mean() - xd.mean()) / xd.mean(), p=p, n=len(cows)))
    t = pd.DataFrame(rows)
    t.to_csv(C.RESULTS / "table_stats.csv", index=False)
    return t


# ----------------------------------------------------------------------------- figures
def fig_heatmap(dec, arch):
    """Decision disagreement with the clean twin: sensor x degradation kind (level 3), per regime."""
    d = dec[(dec.arch == arch) & (dec.level == 3)]
    fig, axes = plt.subplots(1, 2, figsize=(W3, 7.2 * CM), sharey=True)
    mods = C.MODALITIES + ["all"]
    vmax = d.groupby(["regime", "target", "kind"]).disagree.mean().max()
    for ax, reg in zip(axes, ["clean", "aug"]):
        m = d[d.regime == reg].groupby(["target", "kind"]).disagree.mean().unstack().reindex(index=mods, columns=KINDS)
        im = ax.imshow(m.values * 100, cmap=BLUES, vmin=0, vmax=vmax * 100, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j] * 100
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 * vmax * 100 else INK)
        ax.set_xticks(range(len(KINDS)), [KIND_LABEL[k] for k in KINDS], rotation=30, ha="right")
        ax.set_yticks(range(len(mods)), [MOD_LABEL[m] for m in mods])
        ax.set_title("Clean training" if reg == "clean" else "Degradation-aware training", color=INK)
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Decisions changed vs. clean twin (%)")
    save(fig, "fig3_heatmap_disagreement")


def fig_dose_response(dec, arch):
    """State deviation and missed actions vs severity, per degradation kind (all-sensor target)."""
    rules = ["heat", "lying", "feed", "cool"]
    fig, axes = plt.subplots(2, 4, figsize=(W3, 9.5 * CM), sharex=True)
    for j, r in enumerate(rules):
        for row, metric, lab in [(0, "state_dev", "State deviation"), (1, "unsafe", "Missed actions (%)")]:
            ax = axes[row, j]
            for ci, k in enumerate(KINDS):
                sel = dec[(dec.arch == arch) & (dec.regime == "clean") & (dec.rule == r) &
                          ((dec.kind == k) | (dec.kind == "none")) & (dec.target.isin(["all", "none"]))]
                g = sel.groupby("level")[metric].mean() * (100 if metric == "unsafe" else 1)
                ax.plot(g.index, g.values, color=(SERIES + ["#4a3aa7"])[ci], marker=MARKERS[ci],
                        ms=3.5, lw=1.2, label=KIND_LABEL[k])
            if row == 0:
                ax.set_title(RULE_LABEL[r], color=INK)
            if j == 0:
                ax.set_ylabel(lab)
            if row == 1:
                ax.set_xlabel("Severity level")
            ax.set_xticks(range(5))
    axes[0, 0].legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.18), fontsize=8)
    save(fig, "fig4_dose_response")


AFFECTED = {"heat": ["cbt", "all"], "lying": ["ankle", "imu", "uwb", "all"], "feed": ["imu", "uwb", "ankle", "all"],
            "cool": ["thi", "all"]}
FAMILY = {"noise": "Random (noise, dropout)", "dropout": "Random (noise, dropout)",
          "drift": "Systematic (drift, delay)", "delay": "Systematic (drift, delay)",
          "failure": "Dead (failure, stuck-at)", "stuck": "Dead (failure, stuck-at)"}


def fig_propagation(dec, arch):
    """State deviation vs decision disagreement, by fault family: how state error propagates to actions."""
    fig, axes = plt.subplots(1, 4, figsize=(W3, 6.2 * CM))
    units = {"heat": "°C", "lying": "h/24 h", "feed": "ratio", "cool": "THI units"}
    fams = list(dict.fromkeys(FAMILY.values()))
    for ax, r in zip(axes, ["heat", "lying", "feed", "cool"]):
        d = dec[(dec.arch == arch) & (dec.regime == "clean") & (dec.rule == r) & (dec.kind != "none") &
                dec.target.isin(AFFECTED[r])]
        for ci, f in enumerate(fams):
            s = d[d.kind.map(FAMILY) == f]
            ax.scatter(s.state_dev, s.disagree * 100, s=9, alpha=0.6, color=SERIES[ci], marker=MARKERS[ci],
                       lw=0.8 if MARKERS[ci] in "x+" else 0, label=f)
        ax.set_title(RULE_LABEL[r], color=INK)
        ax.set_xlabel(f"State deviation, {units[r]}")
    axes[0].set_ylabel("Decisions changed, %")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, "fig5_propagation")


def table_propagation(dec, arch):
    """Decisions changed per unit of state deviation, by rule and fault type (clean-trained estimator)."""
    d = dec[(dec.arch == arch) & (dec.regime == "clean") & (dec.kind != "none")]
    d = d[d.apply(lambda r: r.target in AFFECTED[r.rule], axis=1)]
    g = d.groupby(["rule", "kind"])[["state_dev", "disagree", "unsafe", "false_alarm"]].mean()
    g["gain"] = g.disagree / g.state_dev
    g.to_csv(C.RESULTS / "table_propagation.csv")
    return g


def fig_regimes(dec):
    """Mean regret per architecture, clean vs degradation-aware training (95 % CI over cows)."""
    d = dec[dec.kind != "none"]
    per_cow = d.groupby(["arch", "regime", "cow"]).regret.mean().reset_index()
    fig, ax = plt.subplots(figsize=(W1, 6.5 * CM))
    xs = ARCHS + ["direct"]
    for ci, reg in enumerate(["clean", "aug"]):
        for i, a in enumerate(xs):
            reg_ = "-" if a == "direct" else reg
            if a == "direct" and reg == "aug":
                continue
            m, lo, hi = boot_ci(per_cow[(per_cow.arch == a) & (per_cow.regime == reg_)].regret)
            x = i + (-0.17 if reg == "clean" else 0.17) * (a != "direct")
            ax.errorbar(x, m, [[m - lo], [hi - m]], fmt=MARKERS[ci], color=MUTED if a == "direct" else SERIES[ci],
                        ms=5, capsize=2, lw=1, label=(None if i else ("Clean training" if reg == "clean" else "Degradation-aware")))
    ax.set_xticks(range(len(xs)), [ARCH_LABEL[a] for a in xs], rotation=20, ha="right")
    ax.set_ylabel("Decision regret per cow-hour")
    ax.legend(fontsize=8)
    save(fig, "fig6_architecture_regimes")


def fig_abstention(ab, arch):
    """Coverage and missed-action rate under abstention policies, by degradation family."""
    fam = {"noise": "Noise", "dropout": "Missing data", "failure": "Missing data", "stuck": "Flat-line",
           "drift": "Drift, delay", "delay": "Drift, delay"}
    d = ab[(ab.arch == arch) & (ab.regime == "aug") & (ab.kind != "none")].copy()
    d["family"] = d.kind.map(fam)
    pols = ["none", "ensemble", "health", "ensemble+health"]
    pl = {"none": "No abstention", "ensemble": "Ensemble disagreement", "health": "Sensor-health gate",
          "ensemble+health": "Ensemble + health"}
    fams = ["Missing data", "Flat-line", "Noise", "Drift, delay"]
    fig, axes = plt.subplots(1, 2, figsize=(W3, 6.5 * CM))
    w = 0.2
    for pi, p in enumerate(pols):
        g = d[d.policy == p].groupby("family")
        cov = g.coverage.mean().reindex(fams) * 100
        uns = g.unsafe.mean().reindex(fams) * 100
        x = np.arange(len(fams)) + (pi - 1.5) * w
        axes[0].bar(x, uns, w * 0.9, color=SERIES[pi], label=pl[p])
        axes[1].bar(x, cov, w * 0.9, color=SERIES[pi])
    for ax, lab in zip(axes, ["Missed actions among automated decisions (%)", "Decisions automated (%)"]):
        ax.set_xticks(range(len(fams)), fams)
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=8)
    save(fig, "fig7_abstention")


def fig_tsfm(tsfm):
    if tsfm is None or tsfm.empty:
        return
    fig, ax = plt.subplots(figsize=(W1, 6.5 * CM))
    best = tsfm[~tsfm.method.isin(["chronos", "carry"])].groupby("method").cbt_mae.mean().idxmin()
    for ci, (meth, lab) in enumerate([("chronos", "Chronos-Bolt (zero-shot)"), ("carry", "Carry-forward"),
                                      (best, f"Twin estimator ({best})")]):
        g = tsfm[tsfm.method == meth].groupby(["kind", "level"]).cbt_mae.mean()
        for k, ls in [("dropout", "-"), ("failure", "--")]:
            if k in g.index.get_level_values(0):
                s = g.loc[k]
                ax.plot(s.index, s.values, ls, color=SERIES[ci], marker=MARKERS[ci], ms=3.5, lw=1.2,
                        label=f"{lab}, {k}")
    ax.set_xlabel("Severity level")
    ax.set_ylabel("Hourly CBT error (degC)")
    ax.legend(fontsize=8)
    save(fig, "fig8_tsfm_cbt")


def main():
    st = pd.read_csv(C.RESULTS / "state.csv")
    dec = pd.read_csv(C.RESULTS / "decisions.csv")
    ab = pd.read_csv(C.RESULTS / "abstention.csv")
    tsfm = pd.read_csv(C.RESULTS / "tsfm.csv") if (C.RESULTS / "tsfm.csv").exists() else None
    arch = best_arch(dec)
    print("best architecture:", arch)
    table_levels()
    print(table_clean(st, dec).to_string())
    print(table_summary(dec).round(4).to_string())
    print(table_stats(dec).round(4).to_string())
    fig_heatmap(dec, arch)
    fig_dose_response(dec, arch)
    fig_propagation(dec, arch)
    print(table_propagation(dec, arch).round(4).to_string())
    fig_regimes(dec)
    fig_abstention(ab, arch)
    fig_tsfm(tsfm)


if __name__ == "__main__":
    main()
