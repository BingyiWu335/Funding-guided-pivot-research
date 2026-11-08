import json
from io import BytesIO

import numpy as np
from common import DATA, OUTPUT, frame
from figure5_layout import compose

R = DATA / "figure5"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Polygon
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


def read(name):
    return json.loads((R / name).read_text("utf8"))


meta = read("heatmap_axes.json")
reg = read("directions.json")
cit = read("attenuation.json")
inter = read("interaction.json")
grid = frame(R / "heatmap_grid.csv")
quad = frame(R / "quadrant_summary.csv")
BLACK = "#1A1A1A"
BLUE = "#4888C8"
TEAL = "#0C8B8E"
CORAL = "#E97B7A"
GRID = "#D6DADE"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.linewidth": 0.65,
        "axes.edgecolor": BLACK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.3,
        "ytick.major.size": 2.3,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
W, H = (677, 240)
fig = plt.figure(figsize=(W / 72, H / 72))
axes = [fig.add_axes([x / W, 43 / H, 165 / W, 165 / H]) for x in [40, 274, 508]]
ab, ac, ad = axes
for ax in axes:
    ax.tick_params(labelsize=8.3, pad=3, direction="out")
    ax.xaxis.label.set_size(9.6)
    ax.yaxis.label.set_size(9.6)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 3
    ax.xaxis.set_zorder(1000)
    ax.yaxis.set_zorder(1000)
    for sp in ax.spines.values():
        sp.set_zorder(1001)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else " n.s."


def gradient_arrow(ax, start, end, pale, deep):
    fig.canvas.draw()
    p0 = ax.transData.transform(start)
    p1 = ax.transData.transform(end)
    direction = (p1 - p0) / np.linalg.norm(p1 - p0)
    normal = np.array([-direction[1], direction[0]])
    ptpx = fig.dpi / 72
    base = p1 - direction * (4.8 * ptpx)
    t = np.linspace(0, 1, 257)
    pts = ax.transData.inverted().transform(
        p0[None, :] + (base - p0)[None, :] * t[:, None]
    )
    rgb0 = np.array(to_rgb(pale))
    rgb1 = np.array(to_rgb(deep))
    colors = rgb0[None, :] + (rgb1 - rgb0)[None, :] * ((t[:-1] + t[1:]) / 2)[:, None]
    ax.add_collection(
        LineCollection(
            np.stack([pts[:-1], pts[1:]], axis=1),
            colors=colors,
            linewidths=1.8,
            capstyle="round",
            zorder=7,
        )
    )
    head = ax.transData.inverted().transform(
        np.array([p1, base + normal * 2.5 * ptpx, base - normal * 2.5 * ptpx])
    )
    ax.add_patch(Polygon(head, closed=True, facecolor=deep, edgecolor="none", zorder=7))


NB = meta["bins_per_axis"]
A = np.full((NB, NB), np.nan)
for row in grid.itertuples():
    A[row.trajectory_bin, row.grant_bin] = row.mean_expert_coauthorship
assert np.isfinite(A).all() and A.min() >= 0 and (A.max() <= 0.45)
cm = LinearSegmentedColormap.from_list(
    "expert_blue", ["#F3F6FA", "#9DBAD4", "#2C5F8A", "#173A57"]
)
im = ab.imshow(
    A,
    origin="lower",
    extent=(0, 1, 0, 1),
    aspect="equal",
    interpolation="nearest",
    cmap=cm,
    norm=Normalize(0, 0.45),
    zorder=1,
)
grant_split = ab.axvline(
    meta["grant_pivot_cutoff"], color="#303030", lw=0.7, ls=(0, (4, 3)), zorder=4
)
ab.axhline(
    meta["trajectory_pivot_cutoff"], color="#303030", lw=0.7, ls=(0, (4, 3)), zorder=4
)
for name, (x, y, ha, va) in {
    "Switching": (0.025, 0.97, "left", "top"),
    "Drift": (0.975, 0.97, "right", "top"),
    "Alignment": (0.025, 0.03, "left", "bottom"),
    "Inertia": (0.975, 0.03, "right", "bottom"),
}.items():
    mean = float(quad.loc[quad.pivot_type.eq(name), "mean_expert_coauthorship"].iloc[0])
    ab.text(
        x,
        y,
        f"{name}\nmean = {mean:.3f}",
        ha=ha,
        va=va,
        fontsize=6.8,
        fontweight="bold",
        zorder=6,
        linespacing=1.1,
        bbox={
            "boxstyle": "round,pad=.14",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
        },
    )


def direction_label(term):
    row = next((c for c in reg["coefficients"] if c["Coefficient"] == term))
    return f"Coef. = {row['Estimate']:+.3f}{stars(row['Pr(>|t|)'])}"


gradient_arrow(ab, (0.3, 0.145), (0.7, 0.145), "#F8D0CF", CORAL)
gradient_arrow(ab, (0.1, 0.4), (0.1, 0.73), "#CBE7E4", TEAL)
box = {
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.88,
    "boxstyle": "round,pad=.12",
}
red_coef_label = ab.text(
    (0.3 + 0.7) / 2,
    0.19,
    direction_label("gp"),
    ha="center",
    va="bottom",
    fontsize=7.0,
    fontweight="bold",
    color="#CE4B49",
    bbox=box,
    zorder=8,
)
ab.text(
    0.14,
    (0.4 + 0.73) / 2,
    direction_label("tp"),
    ha="left",
    va="center",
    rotation=90,
    fontsize=7.0,
    fontweight="bold",
    color=TEAL,
    bbox=box,
    zorder=8,
)
ab.set_xlim(0, 1)
ab.set_ylim(0, 1)
ab.set_xticks(np.arange(0, 1.01, 0.2))
ab.set_yticks(np.arange(0, 1.01, 0.2))
ab.set_xlabel("Grant pivot")
ab.set_ylabel("Trajectory pivot")
ab.set_title("Adjusted expert count", fontsize=9.0, fontweight="bold", pad=7)
cax = fig.add_axes([211 / W, 43 / H, 6 / W, 165 / H])
cb = fig.colorbar(im, cax=cax, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.45])
cb.outline.set_linewidth(0.65)
cb.ax.tick_params(labelsize=8.3, pad=2, length=2, width=0.6)
lines = frame(R / "predicted_lines.csv")
lo, hi = (lines.count_excess.min(), lines.count_excess.max())
for group, color in [(1, BLUE), (0, TEAL)]:
    series = lines[lines.group == group].sort_values("count_excess")
    ac.fill_between(
        series.count_excess,
        series.ci_low,
        series.ci_high,
        color=color,
        alpha=0.15,
        lw=0,
        zorder=1,
    )
    ac.plot(
        series.count_excess,
        series.predicted_citation_impact,
        color=color,
        lw=1.7,
        zorder=3,
    )
ac.set_xlim(lo, hi)
ac.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
ac.set_ylim(0.65, 0.73)
ac.set_yticks([0.65, 0.67, 0.69, 0.71, 0.73])
ac.set_xlabel("Adjusted expert count")
ac.set_ylabel("Predicted citation impact")
for g, tx, ty, color in [(1, -0.03, 0.701, BLUE), (0, 0.5, 0.679, TEAL)]:
    s = next((c for c in inter["derived_contrasts"] if c.get("aligned") == g))
    group_label = (
        "Grant pivot below\nmedian group" if g else "Grant pivot above\nmedian group"
    )
    ac.text(
        tx,
        ty,
        group_label + "\n" + f"Coef. = {s['estimate']:+.3f}{stars(s['p_value'])}",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=color,
        zorder=5,
        linespacing=1.3,
    )
term = inter["interaction"]
ac.text(
    0.025,
    0.98,
    f"Adj. expert count ×\nGrant pivot below median\nCoef. = {term['Estimate']:+.3f}{stars(term['Pr(>|t|)'])}",
    transform=ac.transAxes,
    ha="left",
    va="top",
    fontsize=7.8,
    fontweight="bold",
    linespacing=1.2,
    zorder=6,
)
before = cit["before_switching"]
after = cit["after_switching"]
ad.plot(
    [0, 1],
    [before["Estimate"], after["Estimate"]],
    color="#858585",
    lw=0.85,
    ls=(0, (3, 3)),
    zorder=2,
)
out = []
for i, row, color in [(0, before, BLUE), (1, after, TEAL)]:
    est = row["Estimate"]
    low = row["2.5%"]
    high = row["97.5%"]
    ad.errorbar(
        i,
        est,
        yerr=np.array([[est - low], [high - est]]),
        fmt="o",
        markersize=4.5,
        color=color,
        ecolor=color,
        elinewidth=1.0,
        capsize=2.5,
        capthick=0.85,
        zorder=4,
    )
    label_x = 0.12 if i == 0 else 0.88
    label_y = est + 0.0007 if i == 0 else est - 0.0008
    ad.text(
        label_x,
        label_y,
        f"{est:+.4f}{stars(row['Pr(>|t|)'])}",
        ha="left" if i == 0 else "right",
        va="center",
        fontsize=7.2,
        color=color,
        fontweight="bold",
        zorder=6,
    )
ad.set_xlim(-0.5, 1.5)
ad.set_ylim(0, 0.025)
ad.set_yticks([0, 0.01, 0.02])
ad.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ad.yaxis.set_minor_locator(MultipleLocator(0.005))
ad.tick_params(axis="y", which="minor", length=1.7, width=0.5)
ad.set_xticks([0, 1], ["Without adjusted\nexpert count", "With adjusted\nexpert count"])
ad.set_ylabel("Regression coefficient")
ad.text(
    0.035,
    0.98,
    "Switching",
    transform=ad.transAxes,
    ha="left",
    va="top",
    fontsize=8.0,
    fontweight="bold",
    zorder=6,
)
ad.plot([0, 0, 1, 1], [0.0219, 0.0227, 0.0227, 0.0219], color=BLACK, lw=0.8, zorder=5)
ad.text(
    0.5,
    0.0236,
    f"Δ = −{cit['reduction_pct']:.1f}%",
    ha="center",
    va="bottom",
    fontsize=8.0,
    fontweight="bold",
    zorder=6,
)
for ax in [ac, ad]:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.42, ls=(0, (3, 3)))
ad.axhline(0, color=BLACK, lw=0.8, zorder=1002)
ad.grid(axis="y", which="minor", color=GRID, lw=0.35, ls=(0, (3, 3)), alpha=0.7)
ai = ad.inset_axes([0.17, 0.08, 0.35, 0.24], zorder=20)
expert = cit["count_excess_coefficient"]
estimate = expert["Estimate"]
ai.spines[["top", "right"]].set_visible(False)
ai.set_axisbelow(True)
ai.grid(axis="y", color=GRID, lw=0.35, ls=(0, (3, 3)))
ai.bar(
    [0],
    [estimate],
    width=0.64,
    color=BLUE,
    edgecolor="white",
    lw=0.5,
    alpha=0.92,
    zorder=2,
)
ai.errorbar(
    [0],
    [estimate],
    yerr=np.array([[estimate - expert["2.5%"]], [expert["97.5%"] - estimate]]),
    fmt="none",
    ecolor=BLACK,
    elinewidth=0.8,
    capsize=2.0,
    capthick=0.7,
    zorder=4,
)
ai.axhline(0, color=BLACK, lw=0.65, zorder=1002)
ai.text(
    0,
    expert["97.5%"] + 0.002,
    f"{estimate:+.3f}{stars(expert['Pr(>|t|)'])}",
    ha="center",
    va="bottom",
    fontsize=6.3,
    fontweight="bold",
    color=BLUE,
)
ai.set_xlim(-0.72, 0.72)
ai.set_ylim(0, 0.075)
ai.set_yticks([0, 0.03, 0.06])
ai.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ai.set_xticks([0], ["Adj. expert count"])
ai.tick_params(axis="x", labelsize=7.0, length=0, pad=2)
ai.tick_params(axis="y", labelsize=6.5, length=2, pad=2)
for sp in ai.spines.values():
    sp.set_zorder(1001)
for ax, label in zip(axes, ["b", "c", "d"]):
    pos = ax.get_position()
    fig.text(
        pos.x0 - 27 / W,
        231 / H,
        label,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )
fig.canvas.draw()
box = red_coef_label.get_bbox_patch().get_window_extent(fig.canvas.get_renderer())
margin = 0.5 * fig.dpi / 72
y0, y1 = ab.transData.inverted().transform(
    [[box.x0, box.y0 - margin], [box.x1, box.y1 + margin]]
)[:, 1]
grant_split.set_visible(False)
for ya, yb in [(0, float(y0)), (float(y1), 1)]:
    ab.plot(
        [meta["grant_pivot_cutoff"]] * 2,
        [ya, yb],
        color="#303030",
        lw=0.7,
        ls=(ya * 165 / 0.7 % 7, (4, 3)),
        zorder=4,
    )
fig.canvas.draw()
svg = BytesIO()
fig.savefig(svg, format="svg")
svg.seek(0)
compose(R / "schematic.svg", svg, OUTPUT / "figure5.svg")
plt.close(fig)
