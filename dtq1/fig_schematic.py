"""Figure 1: schematic of the digital twin and of where sensor faults and safeguards act.

    python -m dtq1.fig_schematic
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from .figures import CM, W3, save  # noqa: E402  (also applies the JDS rcParams)

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#f0efec"


def box(ax, x, y, w, h, title, lines, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=edge, lw=1.2))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=8, weight="bold")
    ax.text(x + w / 2, y + h - 0.42, "\n".join(lines), ha="center", va="top", fontsize=8, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color="black", style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, lw=1.1, color=color))


def main():
    fig, ax = plt.subplots(figsize=(W3, 8.2 * CM))
    ax.set_xlim(-0.1, 10.6)
    ax.set_ylim(-0.2, 4.1)
    ax.axis("off")
    box(ax, 0.0, 0.9, 1.75, 2.4, "Sensors", ["Neck IMU 10 Hz", "UWB 15 s", "Body temp. 60 s",
                                            "Leg accel. 60 s", "Pen THI 60 s"], "black")
    ax.add_patch(FancyBboxPatch((2.0, 1.2), 1.3, 1.8, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=GRAY, ec=ORANGE, lw=1.2, ls="--"))
    ax.text(2.65, 2.88, "Faults", ha="center", va="top", fontsize=8, weight="bold")
    ax.text(2.65, 2.58, "noise\ndropout\ndrift, delay\nfailure\nstuck-at", ha="center", va="top",
            fontsize=8, linespacing=1.25)
    ax.add_patch(FancyBboxPatch((3.55, 0.55), 4.3, 3.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="none", ec=BLUE, lw=1.2))
    ax.text(5.7, 3.77, "Digital twin (per cow, every minute)", ha="center", va="top", fontsize=8,
            weight="bold", color=BLUE)
    box(ax, 3.7, 0.9, 1.95, 2.4, "Observation", ["per-minute", "features,", "carry-forward,", "availability,",
                                                  "staleness"], "black")
    box(ax, 5.8, 0.9, 1.95, 2.4, "State estimator", ["LightGBM, GRU,", "Transformer,", "selective SSM",
                                                     "(5-seed ensembles)", "behavior, lying,", "body temp."], "black")
    box(ax, 8.15, 1.75, 2.35, 2.1, "Decisions (hourly)", ["heat-stress check", "lying < 10 h/24 h",
                                                          "feeding drop", "barn cooling"], "black")
    ax.add_patch(FancyBboxPatch((8.15, 0.0), 2.35, 1.4, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=AQUA, lw=1.2))
    ax.text(9.325, 1.3, "Safeguards", ha="center", va="top", fontsize=8, weight="bold")
    ax.text(9.325, 1.0, "ensemble disagreement\nsensor-health gate\n→ defer to farmer", ha="center",
            va="top", fontsize=8, linespacing=1.2)
    ax.text(5.7, 0.25, "Baselines without a twin: direct sensor reading;\nzero-shot Chronos virtual body-temperature sensor",
            ha="center", va="center", fontsize=8)
    arrow(ax, 1.75, 2.1, 2.0, 2.1)
    arrow(ax, 3.3, 2.1, 3.7, 2.1)
    arrow(ax, 5.65, 2.1, 5.8, 2.1)
    arrow(ax, 7.75, 2.6, 8.15, 2.75)
    arrow(ax, 9.325, 1.75, 9.325, 1.4, color=AQUA)
    save(fig, "fig1_twin_schematic")


if __name__ == "__main__":
    main()
