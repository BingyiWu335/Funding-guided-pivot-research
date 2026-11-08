import json
from pathlib import Path

import polars as pl
from common import DATA, OUTPUT

D = DATA / "figure4"
import figure4_means as panel_a
import matplotlib.pyplot as plt
import numpy as np
from apply_readable_b_axes import apply_readable_b_axes
from grid_clearance import clear_grid_under_labels
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.transforms import offset_copy


def read(n):
    return pl.scan_csv(D / Path(n).name, null_values=["NA", ""]).collect().to_dicts()


ref = json.loads((D / "interaction.json").read_text("utf8"))
line = read("plot/prediction_curve.csv")
arr = lambda k: np.array([float(r[k]) for r in line])
xx = arr("topic_similarity")
yy = arr("prediction_difference")
lo = arr("ci_low")
hi = arr("ci_high")
assert np.all(np.diff(xx) > 0), "Prediction similarities must be strictly increasing"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)
for version in ["difference"]:
    fig = plt.figure(figsize=(9.25, 3.7))
    mean = fig.add_axes([0.075, 0.23, 0.28, 0.69])
    forest = fig.add_axes([0.38, 0.23, 0.15, 0.69])
    paired = fig.add_axes([0.66, 0.23, 0.325, 0.69])
    panel_a.draw_means(mean)
    panel_a.draw_coefficients(forest)
    apply_readable_b_axes(fig, mean, forest, mean_ylim=(0.8, 1.2))
    mean.set_ylabel("Mean citation impact", fontsize=11, labelpad=8)
    for text in mean.texts:
        text.set_fontsize(9.5)
    raw_means = {
        (r["group"], r["cell"]): float(r["mean"]) for r in read("plot/rfa_means.csv")
    }
    relative = []
    for text in mean.texts:
        if text.get_text().startswith("Δ ="):
            group = "RFA" if text.get_color() == panel_a.TEAL else "Non-RFA"
            low = raw_means[group, "SA"]
            high = raw_means[group, "ID"]
            assert high > 0
            pct = 100 * (low - high) / high
            text.set_text(text.get_text() + f"\n+{pct:.1f}% relative")
            text.set_linespacing(1.2)
            if group == "RFA":
                x0, y0 = text.get_position()
                text.set_position((x0, y0 + 0.012))
            relative.append(
                {
                    "group": group,
                    "below_median_mean": low,
                    "above_median_mean": high,
                    "absolute_difference": low - high,
                    "relative_increase_percent": pct,
                    "denominator": "above-median group mean of normalized citation impact",
                }
            )
    rfa_low = next(
        (
            row
            for row in read("plot/rfa_means.csv")
            if row["group"] == "RFA" and row["cell"] == "SA"
        )
    )
    for label in mean.texts:
        if label.get_text() == "RFA":
            label.set_position(
                (0, float(rfa_low["mean"]) + float(rfa_low["ci95"]) + 0.008)
            )
    forest.set_xlabel("Reg. coef.", fontsize=11, labelpad=7)
    forest.set_title("")
    coefficients = {r["term"]: r for r in read("plot/rfa_coefficients.csv")}
    anchored = []
    for text, term, y in zip(
        forest.texts,
        ["continuous_RFA", "continuous_GrantCos", "GrantCos_RFA"],
        [2, 1, 0],
    ):
        est = float(coefficients[term]["est"])
        text.set_text(
            text.get_text().replace(
                "RFA × (1 − Grant pivot)", "RFA ×\n(1 − Grant pivot)"
            )
        )
        text.set_fontsize(9.5)
        text.set_linespacing(1.15)
        text.set_ha("center")
        text.set_va("bottom")
        text.set_position((est, y))
        text.set_transform(
            offset_copy(forest.transData, fig=fig, x=0, y=7, units="points")
        )
        anchored.append((text, est, y))
    BLUE = panel_a.BLUE
    shift = 0
    paired.fill_between(
        xx, lo + shift, hi + shift, color=BLUE, alpha=0.14, lw=0, zorder=2
    )
    paired.plot(xx, yy + shift, color=BLUE, lw=1.8, zorder=3)
    paired.axhline(0, color="#777777", lw=0.9, ls=(0, (4, 3)), zorder=2)
    paired.set_ylim(-0.01, 0.05)
    paired.set_ylabel("Predicted citation impact difference", fontsize=11, labelpad=8)
    paired.text(
        0.98,
        0.96,
        "Grant pivot below median × Similarity\n"
        + f"Coef. = {ref['interaction']:.3f}".replace("-", "−")
        + (
            "***"
            if ref["p_value"] < 0.001
            else "**"
            if ref["p_value"] < 0.01
            else "*"
            if ref["p_value"] < 0.05
            else " n.s."
        ),
        transform=paired.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=BLUE,
        linespacing=1.3,
        zorder=7,
    )
    paired.set_xlim(0, 1)
    paired.set_xticks(np.arange(0, 1.001, 0.2))
    paired.set_xlabel("Paper–grant topic similarity", fontsize=11, labelpad=7)
    paired.yaxis.set_major_locator(MultipleLocator(0.02))
    paired.yaxis.set_minor_locator(MultipleLocator(0.01))
    paired.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    paired.grid(axis="y", color=panel_a.GRID, lw=0.45, ls=(0, (3, 3)))
    paired.set_axisbelow(True)
    for ax in [mean, forest, paired]:
        ax.tick_params(labelsize=10, pad=5)
        for spine in ax.spines.values():
            spine.set_zorder(10)
    fig.text(0.015, 0.985, "a", fontsize=15, fontweight="bold", va="top")
    fig.text(0.588, 0.985, "b", fontsize=15, fontweight="bold", va="top")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    zero_x = forest.transData.transform((0, 0))[0]
    for text, est, y in anchored:
        box = text.get_window_extent(renderer)
        dx = max(0.0, (zero_x - box.x0) * 72 / fig.dpi + 2)
        text.set_transform(
            offset_copy(forest.transData, fig=fig, x=dx, y=7, units="points")
        )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clearance = []
    for text in paired.texts:
        box = text.get_window_extent(renderer)
        corners = paired.transData.inverted().transform(
            [[box.x0, box.y0], [box.x1, box.y1]]
        )
        left = max(0.0, corners[0, 0])
        right = min(1.0, corners[1, 0])
        xs = np.linspace(left, right, 1001)
        bandlo = np.interp(xs, xx, lo + shift)
        bandhi = np.interp(xs, xx, hi + shift)
        dy = (
            paired.transData.inverted().transform([[0, 0], [0, 3 * fig.dpi / 72]])[1, 1]
            - paired.transData.inverted().transform([[0, 0], [0, 3 * fig.dpi / 72]])[
                0, 1
            ]
        )
        delta = max(0.0, float(bandhi.max()) + dy - corners[0, 1])
        if delta:
            apos = np.array(text.get_position())
            data_pos = paired.transData.inverted().transform(
                text.get_transform().transform(apos)
            )
            data_pos[1] += delta
            text.set_position(
                text.get_transform()
                .inverted()
                .transform(paired.transData.transform(data_pos))
            )
        fig.canvas.draw()
        box = text.get_window_extent(fig.canvas.get_renderer())
        corners = paired.transData.inverted().transform(
            [[box.x0, box.y0], [box.x1, box.y1]]
        )
        gap = corners[0, 1] - float(bandhi.max())
        gap_pt = (
            gap
            * paired.bbox.height
            / (paired.get_ylim()[1] - paired.get_ylim()[0])
            * 72
            / fig.dpi
        )
        assert gap_pt >= 3 - 1e-07, (text.get_text(), gap_pt)
        clearance.append(
            {
                "label": text.get_text(),
                "side": "above",
                "minimum_band_clearance_pt": gap_pt,
                "axes_position": list(
                    paired.transAxes.inverted().transform(
                        text.get_transform().transform(text.get_position())
                    )
                ),
            }
        )
    for ax in [mean, forest, paired]:
        inv = ax.transAxes.inverted()
        for guide in ax.get_xgridlines() + ax.get_ygridlines():
            v = inv.transform(
                guide.get_transform().transform(guide.get_path().vertices)
            )
            fixed = v[0, 0] if abs(v[0, 0] - v[-1, 0]) < 1e-08 else v[0, 1]
            if fixed < 0 or fixed > 1:
                guide.set_visible(False)
    clear_grid_under_labels(fig, [mean, forest, paired])
    fig.savefig(
        OUTPUT / "figure4.svg", bbox_inches="tight", pad_inches=0.04, facecolor="white"
    )
    plt.close(fig)
