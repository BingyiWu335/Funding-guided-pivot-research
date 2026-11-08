import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import DATA
from plot_inputs import one_row, scan_csv

BASE = DATA / "figure4"
BLUE = "#4888C8"
TEAL = "#0C8B8E"
CORAL = "#EB7774"
GRID = "#E7E7E7"
TEXT = "#151515"
PIVOT_SHADE = "#EAF4F4"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.linewidth": 0.72,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
    }
)


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def tidy_main_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.45, linestyle=(0, (3, 3)))
    ax.tick_params(labelsize=6.9, pad=2.2)
    ax.set_facecolor("white")


def panel_letter(fig, x, y, value):
    fig.text(x, y, value, fontsize=9.2, fontweight="bold", ha="left", va="top")


def draw_means(ax):
    data = scan_csv(BASE / "rfa_means.csv")

    def row(group, cell):
        return one_row(
            data.filter((pl.col("group") == group) & (pl.col("cell") == cell))
        )

    control_a, control_p = (row("Non-RFA", "SA"), row("Non-RFA", "ID"))
    rfa_a, rfa_p = (row("RFA", "SA"), row("RFA", "ID"))
    x = np.array([0.0, 1.0])
    for first, second, colour in [(control_a, control_p, BLUE), (rfa_a, rfa_p, TEAL)]:
        means = [first["mean"], second["mean"]]
        cis = [first.ci95, second.ci95]
        ax.plot(x, means, color=colour, linewidth=1.9, zorder=2)
        ax.errorbar(
            x,
            means,
            yerr=cis,
            fmt="s",
            markersize=6.0,
            color=colour,
            ecolor=colour,
            elinewidth=0.9,
            capsize=2.3,
            markeredgecolor="white",
            markeredgewidth=0.65,
            zorder=4,
        )
    ax.text(
        0,
        rfa_a["mean"] + 0.035,
        "RFA",
        color=TEAL,
        fontsize=7.3,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax.text(
        0,
        control_a["mean"] - 0.018,
        "Non-RFA",
        color=BLUE,
        fontsize=7.3,
        fontweight="bold",
        ha="center",
        va="top",
    )
    mean_tests = scan_csv(BASE / "rfa_mean_tests.csv")
    for group, colour, first, second, offset in [
        ("RFA", TEAL, rfa_a, rfa_p, 0.047),
        ("Non-RFA", BLUE, control_a, control_p, 0.03),
    ]:
        test = one_row(mean_tests.filter(pl.col("group") == group))
        stars = "" if test.star is None else str(test.star)
        ax.text(
            0.5,
            (first["mean"] + second["mean"]) / 2 + offset,
            f"Δ = {test.difference:+.3f}{stars}",
            color=colour,
            fontsize=6.2,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.63, 0.89)
    ax.set_xticks([0, 1], ["Grant\nAligning", "Grant\nPivoting"])
    ax.set_yticks([0.65, 0.7, 0.75, 0.8, 0.85])
    ax.set_ylabel("Citation impact", fontsize=7.5, labelpad=6)
    tidy_main_axis(ax)


def draw_coefficients(ax):
    frame = scan_csv(BASE / "rfa_coefficients.csv")
    items = [
        ("RFA", "continuous_RFA", 2),
        ("1 − Grant pivot", "continuous_GrantCos", 1),
        ("RFA × (1 − Grant pivot)", "GrantCos_RFA", 0),
    ]
    ax.set_facecolor("#EFF7F7")
    ax.axvline(0, color="#808080", linewidth=0.8, zorder=1)
    for label, term, y in items:
        row = one_row(frame.filter(pl.col("term") == term))
        value_stars = "" if row.star is None else str(row.star)
        ax.errorbar(
            row.est,
            y,
            xerr=np.array([[row.est - row.ci_lo], [row.ci_hi - row.est]]),
            fmt="o",
            markersize=4.8,
            color=TEAL,
            ecolor=TEAL,
            elinewidth=0.9,
            capsize=2.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=4,
        )
        ax.text(
            0.004,
            y + 0.32,
            f"{row.est:+.3f}{value_stars}\n{label}",
            color="#006B70",
            fontsize=5.2,
            ha="left",
            va="top",
            linespacing=1.12,
        )
    ax.set_xlim(-0.02, 0.25)
    ax.set_ylim(-0.6, 2.75)
    ax.set_xticks([0, 0.1, 0.2])
    ax.set_yticks([])
    ax.set_xlabel("Citation impact (coef.)", fontsize=6.0, labelpad=4)
    ax.set_title("Regression coef.", fontsize=6.8, color="#666666", pad=2)
    ax.tick_params(axis="x", labelsize=5.7, pad=2)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
