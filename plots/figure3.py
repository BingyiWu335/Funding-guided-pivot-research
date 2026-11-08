from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from axes_foreground import raise_axes_to_foreground
from common import DATA, OUTPUT, frame
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

SOURCE = DATA / "figure3"
DATA = SOURCE
GRANT_TYPE_CUTOFF = float(
    json.loads((SOURCE / "pivot_summary.json").read_text(encoding="utf-8"))[
        "grant_pivot_median"
    ]
)
RESEARCH_TYPE_CUTOFF = float(
    json.loads((SOURCE / "pivot_summary.json").read_text(encoding="utf-8"))[
        "research_pivot_median"
    ]
)
BLACK = "#1A1A1A"
GRID = "#D6DADE"
SCAT = "#4F6271"
BLUE = "#4888C8"
TEAL = "#0C8B8E"
CORAL = "#E97B7A"
GREEN_BG = "#DCEEE9"
POS = BLUE
NEG = CORAL
DIM_COLORS = {"Research": BLUE, "Grant": CORAL}
GP_GROUPS = ["0-25%", "25-50%", "50-75%", "75-100%"]
DISPLAY_GP_GROUPS = [GP_GROUPS[0], GP_GROUPS[3]]
GP_COLORS = {
    "0-25%": "#2D73AC",
    "25-50%": "#67A9C6",
    "50-75%": "#D9945E",
    "75-100%": "#C94B3A",
}
FE_COLORS = {"Grant FE": TEAL}
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


def fmt_n(n: int) -> str:
    if n >= 1000000:
        return f"{n / 1000000:.2f}M"
    if n >= 1000:
        return f"{n / 1000:.0f}k"
    return str(int(n))


def style_axis(ax, *, grid_y: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_y:
        ax.grid(axis="y", color=GRID, lw=0.42, ls=(0, (3, 3)))
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.0, pad=1.5)
    ax.xaxis.label.set_size(6.8)
    ax.yaxis.label.set_size(6.8)


def panel_letter(ax, label: str, dx: float = -0.15, dy: float = 1.08) -> None:
    ax.text(
        dx,
        dy,
        label,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        color=BLACK,
        ha="right",
        va="top",
    )


def shade_pivot_side(ax, median: float, left: str, right: str) -> None:
    ax.axvspan(median, 1.0, color=GREEN_BG, alpha=0.7, lw=0, zorder=0)
    ax.axvline(median, color="#858585", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(
        median * 0.48,
        0.94,
        left,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=5.6,
        color="#909090",
        style="italic",
    )
    ax.text(
        median + (1 - median) * 0.52,
        0.94,
        right,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=5.6,
        color=TEAL,
        style="italic",
    )


summary = json.loads((SOURCE / "pivot_summary.json").read_text(encoding="utf-8"))
density = frame(SOURCE / "pivot_density.csv")
pooled = frame(SOURCE / "inset_pooled.csv")
coefs = frame(SOURCE / "main2_coefs.csv")
fit_b = frame(DATA / "predictions.csv").rename(
    columns={
        "trajectory_pivot": "research_pivot",
        "ci_low": "ci_lo",
        "ci_high": "ci_hi",
    }
)
_group_map = dict(zip(["Q1", "Q2", "Q3", "Q4"], GP_GROUPS))
fit_b["gp_group"] = fit_b.gp_group.map(_group_map)
assert fit_b.gp_group.notna().all()
pred_c = frame(SOURCE / "panel_c_predictions.csv")
gmed = GRANT_TYPE_CUTOFF
rmed = RESEARCH_TYPE_CUTOFF
gl = float(summary["grant_pivot_q005"])
gh = float(summary["grant_pivot_q995"])
rl = float(summary["research_pivot_q005"])
rh = float(summary["research_pivot_q995"])
type_counts = summary["pivot_type_counts"]
fig = plt.figure(figsize=(7.2, 3.95))
gs = fig.add_gridspec(
    2,
    4,
    left=0.065,
    right=0.985,
    top=0.935,
    bottom=0.135,
    width_ratios=[1.32, 1.32, 1.0, 1.0],
    hspace=0.27,
    wspace=0.48,
)
ax_a = fig.add_subplot(gs[:, :2])
ax_d = fig.add_subplot(gs[0, 2])
ax_b = fig.add_subplot(gs[1, 2])
ax_c = fig.add_subplot(gs[0, 3])
ax_e = fig.add_subplot(gs[1, 3])
grid = np.full((40, 40), np.nan)
grid[density.trajectory_bin, density.grant_bin] = density["count"]
edges = np.linspace(0, 1, 41)
color_scale = LinearSegmentedColormap.from_list("pair_density", ["#f1f4f6", "#354c5c"])
mesh = ax_a.pcolormesh(edges, edges, grid, cmap=color_scale, norm=LogNorm(), zorder=2)
scale_axis = ax_a.inset_axes([0.72, 0.62, 0.22, 0.022], zorder=8)
scale = fig.colorbar(mesh, cax=scale_axis, orientation="horizontal")
scale.set_label("Pairs per bin", fontsize=5.5, labelpad=1)
scale.ax.tick_params(labelsize=5, length=1.5, pad=1)
ax_a.axvline(gmed, color=BLACK, lw=0.8, ls=(0, (4, 2)), alpha=0.85, zorder=4)
ax_a.axhline(rmed, color=BLACK, lw=0.8, ls=(0, (4, 2)), alpha=0.85, zorder=4)
median_box = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.55)
ax_a.text(
    gmed + (gh - gl) * 0.015,
    rl + (rh - rl) * 0.05,
    f"Median = {gmed:.2f}",
    fontsize=5.7,
    fontweight="bold",
    color=BLACK,
    bbox=median_box,
    zorder=8,
)
ax_a.text(
    gl + (gh - gl) * 0.015,
    rmed + (rh - rl) * 0.015,
    f"Median = {rmed:.2f}",
    fontsize=5.7,
    fontweight="bold",
    color=BLACK,
    bbox=median_box,
    zorder=8,
)
ax_a.set_xlim(gl, gh)
ax_a.set_ylim(rl, rh)
ax_a.set_xlabel("Grant pivot", fontsize=7.4, color=BLACK, labelpad=2)
ax_a.set_ylabel("Trajectory pivot", fontsize=7.4, color=BLACK, labelpad=2)
ax_a.tick_params(labelsize=6.4, length=2.5, pad=1.5)
quad_box = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.55)


def quadrant_label(x, y, ha, va, key, name):
    ax_a.text(
        x,
        y,
        f"{name}\nn = {fmt_n(type_counts.get(key, 0))}",
        ha=ha,
        va=va,
        fontsize=6.6,
        fontweight="bold",
        color=BLACK,
        linespacing=1.18,
        bbox=quad_box,
        zorder=6,
    )


quadrant_label(
    gl + (gh - gl) * 0.03,
    rh - (rh - rl) * 0.03,
    "left",
    "top",
    "switching",
    "Switching",
)
quadrant_label(
    gh - (gh - gl) * 0.03, rh - (rh - rl) * 0.03, "right", "top", "drift", "Drift"
)
quadrant_label(
    gl + (gh - gl) * 0.03,
    rl + (rh - rl) * 0.03,
    "left",
    "bottom",
    "alignment",
    "Alignment",
)
quadrant_label(
    gh - (gh - gl) * 0.03,
    rl + (rh - rl) * 0.03,
    "right",
    "bottom",
    "inertia",
    "Inertia",
)
for spine in ax_a.spines.values():
    spine.set_visible(True)
shade_pivot_side(ax_b, rmed, "Aligning", "Pivoting")
handles_b = []
for group in DISPLAY_GP_GROUPS:
    color = GP_COLORS[group]
    fit = fit_b[fit_b.gp_group == group].sort_values("research_pivot")
    x = fit.research_pivot.to_numpy(float)
    y = fit.predicted_citation.to_numpy(float)
    lo = fit.ci_lo.to_numpy(float)
    hi = fit.ci_hi.to_numpy(float)
    ax_b.fill_between(x, lo, hi, color=color, alpha=0.08, lw=0, zorder=2)
    handles_b.append(ax_b.plot(x, y, color=color, lw=1.45, zorder=4)[0])
_shown_d = fit_b[fit_b.gp_group.isin(DISPLAY_GP_GROUPS)]
_band_min = float(_shown_d.ci_lo.min())
_band_max = float(_shown_d.ci_hi.max())
_band_span = _band_max - _band_min
_d_ylo = float(np.floor((_band_min - 0.05 * _band_span) / 0.05) * 0.05)
_d_yhi = float(np.ceil((_band_max + 0.05 * _band_span) / 0.01) * 0.01)
ax_b.set_ylim(_d_ylo, _d_yhi)
ax_b.set_yticks(np.arange(_d_ylo, _d_yhi + 1e-10, 0.05))
assert _d_ylo < _band_min and _d_yhi > _band_max
ax_b.set_xlim(0, 1)
ax_b.set_xticks(np.linspace(0, 1, 6))
ax_b.set_xlabel("Trajectory pivot")
ax_b.set_ylabel("Mean citation impact")
ax_b.legend(
    handles_b,
    ["Q1", "Q4"],
    ncol=5,
    loc="lower left",
    bbox_to_anchor=(0.36, 1.002),
    fontsize=5.1,
    handlelength=0.55,
    handletextpad=0.12,
    columnspacing=0.13,
    borderaxespad=0,
)
ax_b.text(
    0.0,
    1.025,
    "Grant pivot:",
    transform=ax_b.transAxes,
    fontsize=5.1,
    color=BLACK,
    ha="left",
    va="bottom",
)
style_axis(ax_b)
shade_pivot_side(ax_c, gmed, "Grant aligning", "Grant pivoting")
handles_c = []
for model in ("Grant FE",):
    color = FE_COLORS[model]
    data = pred_c[pred_c.model == model].sort_values("grant_pivot")
    x = data.grant_pivot.to_numpy(float)
    y = data.predicted_citation.to_numpy(float)
    lo = data.ci_lo.to_numpy(float)
    hi = data.ci_hi.to_numpy(float)
    ax_c.fill_between(x, lo, hi, color=color, alpha=0.11, lw=0, zorder=2)
    handles_c.append(ax_c.plot(x, y, color=color, lw=1.55, zorder=4)[0])
ax_c.set_xlim(0, 1)
ax_c.set_xticks(np.linspace(0, 1, 6))
ax_c.set_xlabel("Grant pivot")
ax_c.set_ylabel("Predicted citation impact")
style_axis(ax_c)
fits = {}
for dim in ("Research", "Grant"):
    data = pooled[pooled.dim == dim].sort_values("bin_center")
    x = data.bin_center.to_numpy(float)
    y = data.cit.to_numpy(float)
    from binned_fit import fit_binned, predict

    fitted = fit_binned(x, y)
    coef, cov = (fitted["coef"], fitted["cov"])
    fits[dim] = coef
    xs = np.linspace(x.min(), x.max(), 60)
    ys, lower, upper = predict(fitted, xs)
    ax_d.fill_between(
        xs, lower, upper, color=DIM_COLORS[dim], alpha=0.18, lw=0, zorder=2
    )
    ax_d.plot(xs, ys, color=DIM_COLORS[dim], lw=1.65, zorder=4)
xr = 0.64
ax_d.text(
    xr,
    np.polyval(fits["Grant"], xr) + 0.017,
    "Grant pivot",
    color=CORAL,
    fontsize=5.4,
    fontweight="bold",
    ha="center",
    va="bottom",
)
ax_d.text(
    xr,
    np.polyval(fits["Research"], xr) - 0.016,
    "Trajectory pivot",
    color=BLUE,
    fontsize=5.4,
    fontweight="bold",
    ha="center",
    va="top",
)
ax_d.legend(
    [Line2D([0], [0], color="#B5B5B5", lw=2.2)],
    ["Best fit"],
    loc="upper right",
    fontsize=5.3,
    handlelength=1.2,
    borderaxespad=0.2,
)
ax_d.set_xlim(0, 1)
ax_d.set_xticks(np.linspace(0, 1, 6))
ax_d.set_xlabel("Pivot size")
ax_d.set_ylabel("Mean citation impact")
style_axis(ax_d)
coef_data = coefs[coefs.outcome == "Z-Cit"].set_index("pivot")
pivots = ["alignment", "switching", "inertia", "drift"]
labels = ["Alignment", "Switching", "Inertia", "Drift"]
ax_e.axhline(0, color=BLACK, lw=0.8, zorder=1002, gid="zero_baseline")
m = max(abs(coef_data.est).max(), abs(coef_data.hi).max(), abs(coef_data.lo).max())
for i, pivot in enumerate(pivots):
    row = coef_data.loc[pivot]
    color = POS if row.est >= 0 else NEG
    ax_e.bar(
        i,
        row.est,
        width=0.64,
        color=color,
        edgecolor="white",
        lw=0.5,
        alpha=0.92,
        zorder=3,
    )
    ax_e.errorbar(
        i,
        row.est,
        yerr=[[row.est - row.lo], [row.hi - row.est]],
        fmt="none",
        ecolor=BLACK,
        elinewidth=0.8,
        capsize=1.8,
        zorder=5,
    )
    stars = row.star if isinstance(row.star, str) and row.star != "+" else ""
    y_text = row.hi if row.est >= 0 else row.lo
    offset = (1 if row.est >= 0 else -1) * m * 0.035
    ax_e.text(
        i,
        y_text + offset,
        f"{row.est:+.3f}{stars}",
        ha="center",
        va="bottom" if row.est >= 0 else "top",
        fontsize=5.0,
        fontweight="bold",
        color=color,
        zorder=6,
    )
ax_e.set_ylim(-m * 1.6, m * 1.6)
ax_e.set_xticks(range(4))
ax_e.set_xticklabels(labels, rotation=28, ha="right", rotation_mode="anchor")
for tick in ax_e.get_xticklabels():
    tick.set_fontsize(5.4)
    tick.set_fontweight("bold")
ax_e.tick_params(axis="x", length=0)
ax_e.set_ylabel("Regression coefficient")
style_axis(ax_e)
for axis, label in ((ax_a, "a"), (ax_d, "b"), (ax_c, "c"), (ax_b, "d"), (ax_e, "e")):
    panel_letter(
        axis,
        label,
        dx=-0.075 if axis is ax_a else -0.16,
        dy=1.04 if axis is ax_a else 1.09,
    )
fig.set_size_inches(7.2, 3.9)
for ax, rect in [
    (ax_a, [0.058, 0.092, 0.492, 0.86]),
    (ax_d, [0.617, 0.573, 0.154, 0.38]),
    (ax_c, [0.834, 0.573, 0.154, 0.38]),
    (ax_b, [0.617, 0.092, 0.154, 0.38]),
    (ax_e, [0.834, 0.092, 0.154, 0.38]),
]:
    ax.set_position(rect)
for ax in (ax_a, ax_d, ax_c, ax_b, ax_e):
    for t in list(ax.texts):
        if t.get_text() in list("abcde"):
            t.remove()
for label, x, y in [
    ("a", 0.011, 0.987),
    ("b", 0.569, 0.987),
    ("c", 0.786, 0.987),
    ("d", 0.569, 0.511),
    ("e", 0.786, 0.511),
]:
    fig.text(
        x, y, label, fontsize=9, fontweight="bold", ha="left", va="top", color=BLACK
    )
for ax in (ax_b, ax_c):
    legend = ax.get_legend()
    if legend:
        legend.remove()
for t in list(ax_b.texts):
    if t.get_text() == "Grant pivot:":
        t.remove()
for t in ax_d.texts:
    if "Grant pivot" in t.get_text():
        t.set_text("Grant pivot")
        t.set_position((0.65, np.polyval(fits["Grant"], 0.65) + 0.04))
        t.set_fontsize(5.4)
    if "Trajectory pivot" in t.get_text() or "Research pivot" in t.get_text():
        t.set_text("Trajectory pivot")
        t.set_position((0.38, np.polyval(fits["Research"], 0.38) - 0.08))
        t.set_fontsize(5.4)
label_x = [0.48, 0.18, 0.66, 0.72]
offset_points = [8, -9, 7, -10]
for group, x, dy in zip(GP_GROUPS, label_x, offset_points):
    if group not in DISPLAY_GP_GROUPS:
        continue
    fit = fit_b[fit_b.gp_group == group].sort_values("research_pivot")
    assert np.all(np.diff(fit.research_pivot) > 0)
    y = float(np.interp(x, fit.research_pivot, fit.predicted_citation))
    label = "Grant pivot Q" + str(GP_GROUPS.index(group) + 1)
    ax_b.annotate(
        label,
        (x, y),
        xytext=(0, dy),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=5.1,
        fontweight="bold",
        color=GP_COLORS[group],
    )
slopes_c = {}
for model in ["Grant FE"]:
    data = pred_c[pred_c.model == model].sort_values("grant_pivot")
    slopes_c[model] = float(
        (data.predicted_citation.iloc[-1] - data.predicted_citation.iloc[0])
        / (data.grant_pivot.iloc[-1] - data.grant_pivot.iloc[0])
    )
    intercept = float(
        data.predicted_citation.iloc[0] - slopes_c[model] * data.grant_pivot.iloc[0]
    )
    assert np.allclose(
        data.predicted_citation,
        intercept + slopes_c[model] * data.grant_pivot,
        rtol=0,
        atol=1e-12,
    )
c_coefs = frame(DATA / "focal_coefficients.csv")
for model, x, dy in [("Grant FE", 0.27, -45)]:
    d = pred_c[pred_c.model == model].sort_values("grant_pivot")
    assert np.all(np.diff(d.grant_pivot) > 0)
    y = float(np.interp(x, d.grant_pivot, d.predicted_citation))
    key = "c_funder" if model == "Funder FE" else "c_grant_id"
    row = c_coefs[(c_coefs.model == key) & (c_coefs.se_type == "project_episode")].iloc[
        0
    ]
    star = row.star if isinstance(row.star, str) else ""
    assert abs(row.estimate - slopes_c[model]) < 1e-12
    ax_c.annotate(
        f"Coef. = {row.estimate:+.3f}{star}".replace("-", "−"),
        (x, y),
        xytext=(0, dy),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=5.0,
        fontweight="bold",
        color=FE_COLORS[model],
    )
for t in ax_c.texts:
    if t.get_text() == "Grant aligning":
        t.set_text("Aligning")
    if t.get_text() == "Grant pivoting":
        t.set_text("Pivoting")
for text in ax_d.texts:
    if text.get_text() == "Grant pivot":
        text.set_text(f"Grant pivot\nSlope = {float(fits['Grant'][0]):+.3f}")
    elif text.get_text() == "Trajectory pivot":
        text.set_text(f"Trajectory pivot\nSlope = {float(fits['Research'][0]):+.3f}")
ax_e.set_ylabel("Regression coefficient")
for t in ax_e.get_xticklabels():
    t.set_fontsize(5.2)
    t.set_rotation(25)
    t.set_rotation_mode("default")
    t.set_ha("center")
ax_e.set_ylim(-m * 1.6, m * 1.7)
ax_d.get_legend().set_bbox_to_anchor((1, 0.94))
ax_b.legend(
    handles=[Line2D([0], [0], color="#b3b3b3", lw=2.0)],
    labels=["Best fit"],
    loc="lower right",
    bbox_to_anchor=(0.98, 0.015),
    fontsize=5.1,
    handlelength=1.2,
    handletextpad=0.35,
    borderaxespad=0.2,
    frameon=False,
)
for axis in (ax_a, ax_d, ax_c, ax_b, ax_e):
    axis.xaxis.labelpad = 3.0
    axis.yaxis.labelpad = 3.0
    axis.tick_params(axis="both", pad=3.0)
for axis in (ax_d, ax_c, ax_b):
    for text in axis.texts:
        text.set_bbox(None)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
target_bottom = ax_b.xaxis.label.get_window_extent(renderer).y0
longest_tick = max(
    ax_e.xaxis.get_major_ticks(), key=lambda tick: len(tick.label1.get_text())
)
current_bottom = longest_tick.label1.get_window_extent(renderer).y0
common_pad = longest_tick.get_pad() + (current_bottom - target_bottom) * 72 / fig.dpi
ax_e.tick_params(axis="x", pad=common_pad)
fig.canvas.draw()
e_bottoms = [
    tick.label1.get_window_extent(fig.canvas.get_renderer()).y0
    for tick in ax_e.xaxis.get_major_ticks()
]
longest_index = [tick.label1.get_text() for tick in ax_e.xaxis.get_major_ticks()].index(
    longest_tick.label1.get_text()
)
assert abs(e_bottoms[longest_index] - target_bottom) * 72 / fig.dpi < 0.05
raise_axes_to_foreground(fig, ax_e)
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath

for ax in (ax_b, ax_c, ax_d, ax_e):
    paths = []
    for line in list(ax.lines) + ax.get_xgridlines() + ax.get_ygridlines():
        if line.get_visible():
            paths.append(line.get_path().transformed(line.get_transform()))
    for coll in ax.collections:
        if isinstance(coll, LineCollection):
            for segment in coll.get_segments():
                paths.append(MplPath(coll.get_transform().transform(segment)))
    for spine in ax.spines.values():
        if spine.get_visible():
            paths.append(spine.get_path().transformed(spine.get_transform()))
    targets = [t for t in ax.texts if t.get_text() not in ["Aligning", "Pivoting"]]
    for t in targets:
        base = t.get_position()
        font_size = t.get_fontsize()
        other = [
            u.get_window_extent(fig.canvas.get_renderer())
            for u in list(ax.texts)
            + ax.get_xticklabels()
            + ax.get_yticklabels()
            + (ax.get_legend().get_texts() if ax.get_legend() else [])
            if u is not t and u.get_visible()
        ]
        candidates = sorted(
            [
                (dx, dy)
                for dx in range(
                    -18 if ax is ax_c else -14 if ax is ax_b else -8,
                    19 if ax is ax_c else 15 if ax is ax_b else 9,
                    2,
                )
                for dy in range(
                    -22 if ax is ax_c else -24 if ax is ax_b else -14,
                    23 if ax is ax_c else 25 if ax is ax_b else 15,
                    1,
                )
            ],
            key=lambda q: (q[0] * q[0] * 2 + q[1] * q[1], abs(q[0]), q[1], q[0]),
        )
        chosen = None
        for size in (font_size, font_size * 0.95, font_size * 0.9, min(font_size, 5)):
            t.set_fontsize(size)
            bounds = t.get_window_extent(fig.canvas.get_renderer()).padded(
                fig.dpi / 144
            )
            for dx, dy in candidates:
                box = bounds.translated(dx * fig.dpi / 72, dy * fig.dpi / 72)
                if not ax.bbox.contains(box.x0, box.y0) or not ax.bbox.contains(
                    box.x1, box.y1
                ):
                    continue
                if any(box.overlaps(b) for b in other):
                    continue
                if any(p.intersects_bbox(box, filled=False) for p in paths):
                    continue
                chosen = (dx, dy)
                break
            if chosen is not None:
                break
        if chosen is None:
            t.set_fontsize(font_size)
            continue
        dx, dy = chosen
        if hasattr(t, "xyann"):
            t.set_position((base[0] + dx, base[1] + dy))
        else:
            xy = t.get_transform().transform(base)
            new = (
                t.get_transform()
                .inverted()
                .transform((xy[0] + dx * fig.dpi / 72, xy[1] + dy * fig.dpi / 72))
            )
            t.set_position(new)
    fig.canvas.draw()
fig.savefig(OUTPUT / "figure3.svg")
plt.close(fig)
