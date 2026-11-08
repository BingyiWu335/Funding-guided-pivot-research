import math

import matplotlib as mpl
import numpy as np
import polars as pl
from common import DATA, OUTPUT
from figure2_style import apply

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from scipy.stats import t

COLOR = {"NIH": "#4888C8", "NSF": "#E97B7A", "NSFC": "#0C8B8E"}
FUNDER_MARKER = {"NIH": "o", "NSF": "s", "NSFC": "^"}
GRID = "#D9DDE1"
REFERENCE = "#858C93"
BLACK = "#1A1A1A"
W, H = (792, 503)
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 12.5,
        "axes.linewidth": 0.85,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def axis_style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.tick_params(pad=3)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=0.5, ls=(0, (2.5, 2.5)), alpha=0.75)
    for spine in ax.spines.values():
        spine.set_zorder(30)


def ordinary_fit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    assert len(x) >= 3 and len(set(x)) == len(x)
    center = x.mean()
    d = x - center
    sxx = float(d @ d)
    slope = float(d @ (y - y.mean()) / sxx)
    intercept = float(y.mean())
    residual = y - (intercept + slope * d)
    mse = float(residual @ residual / (len(x) - 2))
    xs = np.linspace(x.min(), x.max(), 180)
    ys = intercept + slope * (xs - center)
    band = float(t.ppf(0.975, len(x) - 2)) * np.sqrt(
        mse * (1 / len(x) + (xs - center) ** 2 / sxx)
    )
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(x)), d]), y, rcond=None)[0]
    assert np.allclose(coef, [intercept, slope], rtol=1e-10, atol=1e-12)
    return (
        xs,
        ys,
        band,
        {
            "slope_per_year": slope,
            "mean_year": center,
            "intercept_at_mean_year": intercept,
            "n_annual_cells": len(x),
            "residual_sd": math.sqrt(mse),
        },
    )


def gap_simple_reference_lines(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        excluded = [child.get_tightbbox(renderer).padded(3) for child in ax.child_axes]
        excluded += [
            v.get_window_extent(renderer).padded(2.5) for v in ax.texts if v.get_text()
        ]
        if ax.get_legend():
            excluded.append(ax.get_legend().get_window_extent(renderer).padded(2))
        if hasattr(ax, "extra_legend"):
            excluded.append(ax.extra_legend.get_window_extent(renderer).padded(2))
        for line in list(ax.get_xgridlines()) + list(ax.get_ygridlines()):
            pts = line.get_transform().transform(line.get_xydata())
            horiz = abs(pts[0, 1] - pts[1, 1]) < 1e-06
            which = 0 if horiz else 1
            other = 1 - which
            lo, hi = sorted(pts[:, which])
            const = pts[0, other]
            bounds = ax.get_window_extent(renderer)
            lo = max(lo, bounds.x0 if horiz else bounds.y0)
            hi = min(hi, bounds.x1 if horiz else bounds.y1)
            cuts = []
            for b in excluded:
                a, z = (b.y0, b.y1) if horiz else (b.x0, b.x1)
                if a <= const <= z:
                    left, right = (b.x0, b.x1) if horiz else (b.y0, b.y1)
                    if right > lo and left < hi:
                        cuts.append((max(lo, left), min(hi, right)))
            if not cuts:
                continue
            line.set_visible(False)
            cursor = lo
            segments = []
            for left, right in sorted(cuts):
                if left > cursor:
                    segments.append((cursor, left))
                cursor = max(cursor, right)
            if cursor < hi:
                segments.append((cursor, hi))
            for left, right in segments:
                p = (
                    np.array([[left, const], [right, const]])
                    if horiz
                    else np.array([[const, left], [const, right]])
                )
                p = fig.transFigure.inverted().transform(p)
                ax.add_line(
                    Line2D(
                        p[:, 0],
                        p[:, 1],
                        transform=fig.transFigure,
                        color=line.get_color(),
                        alpha=line.get_alpha(),
                        lw=line.get_linewidth(),
                        ls=line._unscaled_dash_pattern,
                        zorder=0,
                        clip_on=False,
                    )
                )


annual = pl.scan_csv(DATA / "figure2/annual_unit_means.csv").filter(
    pl.col("n_units") >= 30
)
profile = pl.scan_csv(DATA / "figure2/within_unit_changes.csv")
year_bounds = (
    annual.select(
        pl.col("fund_year").min().alias("lo"), pl.col("fund_year").max().alias("hi")
    )
    .collect()
    .row(0, named=True)
)
xlo = 5 * math.floor(year_bounds["lo"] / 5) - 1
xhi = year_bounds["hi"] + 1
first_tick = 10 * math.ceil(xlo / 10)
fig = plt.figure(figsize=(11, H / 72), facecolor="none")
left, right = (43.77058585, 787.46807103)
width = 202.0
gap = (right - left - 3 * width) / 2
top, bottom = (17.9014672, 214.98675742)
axes = []
fits = []
for i in range(3):
    l = left + i * (width + gap)
    axes.append(fig.add_axes([l / W, (H - bottom) / H, width / W, (bottom - top) / H]))
for ax, metric, marker in [
    (axes[0], "Grant Pivot", "o"),
    (axes[1], "Trajectory Pivot", "^"),
]:
    axis_style(
        ax, "Funded year", "Mean " + metric.lower().replace("pivot", "pivot") + " size"
    )
    ax.set_ylabel("Mean " + metric.replace("Pivot", "pivot") + " size", labelpad=4)
    all_y = []
    for funder, col in COLOR.items():
        z = (
            annual.filter((pl.col("funder") == funder) & (pl.col("metric") == metric))
            .sort("fund_year")
            .select("fund_year", "mean")
            .collect()
        )
        x = z["fund_year"].to_numpy().astype(float)
        y = z["mean"].to_numpy()
        ax.scatter(
            x,
            y,
            s=9,
            marker=marker,
            facecolor=mpl.colors.to_rgba(col, 0.2),
            edgecolor=mpl.colors.to_rgba(col, 0.65),
            lw=0.4,
            zorder=3,
        )
        segments = [(np.ones(len(x), dtype=bool), "-", "all_years")]
        all_y.extend(y)
        for selected, linestyle, period in segments:
            xx, yy = (x[selected], y[selected])
            xs, ys, ci, fit = ordinary_fit(xx, yy)
            fits.append(
                {
                    "funder": funder,
                    "metric": metric,
                    "period": period,
                    "first_year": int(xx.min()),
                    "last_year": int(xx.max()),
                    **fit,
                }
            )
            ax.fill_between(
                xs, ys - ci, ys + ci, color=col, alpha=0.075, lw=0, zorder=1
            )
            ax.plot(xs, ys, color=col, ls=linestyle, lw=1.5, zorder=2)
            all_y.extend(ys - ci)
            all_y.extend(ys + ci)
    ax.set_xlim(xlo, xhi)
    ax.set_xticks(np.arange(first_tick, xhi, 10))
    ax.set_ylim(
        math.floor((min(all_y) - 0.02) / 0.05) * 0.05,
        0.6 if metric == "Grant Pivot" else 0.55,
    )
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend(
        handles=[
            Line2D([], [], color=col, lw=1.5, label=f) for f, col in COLOR.items()
        ],
        loc="upper left",
        bbox_to_anchor=(0.012, 1.015),
        ncol=3,
        fontsize=9.0,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.4,
    )
for ax, metric in [(axes[0], "Grant Pivot"), (axes[1], "Trajectory Pivot")]:
    for funder in COLOR:
        fit = next((f for f in fits if f["funder"] == funder and f["metric"] == metric))
        label = f"{10 * fit['slope_per_year']:+.4f}/decade"
        if funder == "NSFC":
            year = fit["last_year"]
            offset = (-2, 9)
            ha = "right"
            va = "bottom"
        elif metric == "Grant Pivot" and funder == "NIH":
            year = fit["last_year"] - 1
            offset = (-2, -20)
            ha = "right"
            va = "top"
        elif metric == "Trajectory Pivot" and funder == "NIH":
            year = fit["last_year"] - 12
            offset = (-2, 12)
            ha = "right"
            va = "bottom"
        elif metric == "Trajectory Pivot" and funder == "NSF":
            year = fit["last_year"]
            offset = (-2, -15)
            ha = "right"
            va = "top"
        else:
            year = fit["first_year"] + (metric == "Trajectory Pivot")
            offset = (2, -16)
            ha = "left"
            va = "top"
        y = (
            fit["intercept_at_mean_year"]
            + (year - fit["mean_year"]) * fit["slope_per_year"]
        )
        ax.annotate(
            label,
            (year, y),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            color=COLOR[funder],
            fontsize=9.0,
            zorder=12,
        )


def make_profile(ax, inset=False):
    for funder in ["Pooled"] if inset else COLOR:
        z = (
            profile.filter(pl.col("scope") == funder)
            .sort("lag_bin")
            .select("lag_bin", "estimate", "ci")
            .collect()
        )
        x = z["lag_bin"].to_numpy().astype(float)
        y = z["estimate"].to_numpy()
        ci = z["ci"].to_numpy()
        assert x.tolist() == [0, 1, 2, 3, 4]
        col = BLACK if inset else COLOR[funder]
        ax.fill_between(x, y - ci, y + ci, color=col, alpha=0.1, lw=0, zorder=1)
        ax.plot(
            x,
            y,
            color=col,
            lw=1.1 if inset else 1.65,
            marker="o" if inset else FUNDER_MARKER[funder],
            ms=3 if inset else 4.4,
            mec="white",
            mew=0.5,
            zorder=3,
        )


b = axes[2]
axis_style(b, "Publication year since funding", "Δ Grant Pivot (vs. first year)")
make_profile(b)
upper = profile.select((pl.col("estimate") + pl.col("ci")).max()).collect().item()
b.set_xlim(-0.25, 4.25)
b.set_ylim(-0.002, max(0.03, math.ceil((upper + 0.003) / 0.005) * 0.005))
b.set_xticks(range(5), ["1", "2", "3", "4", "5+"])
b.set_yticks(np.arange(0, b.get_ylim()[1] + 1e-10, 0.01))
b.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
b.axhline(0, color=REFERENCE, lw=0.8, zorder=1)
b.axvspan(-0.25, 0.27, facecolor="#F1F3F4", edgecolor="none", linewidth=0, zorder=-2)
b.text(
    -0.04,
    0.52,
    "Funded year",
    transform=b.get_xaxis_transform(),
    rotation=90,
    rotation_mode="anchor",
    ha="center",
    va="center",
    color="#676767",
    fontsize=8.3,
)
b.legend(
    handles=[
        Line2D(
            [],
            [],
            color=COLOR[f],
            marker=FUNDER_MARKER[f],
            ms=3.4,
            mec="white",
            mew=0.5,
            lw=1.3,
            label=f,
        )
        for f in COLOR
    ],
    loc="upper left",
    bbox_to_anchor=(0.0, 1.015),
    ncol=3,
    fontsize=8.6,
    columnspacing=0.8,
    handlelength=1.1,
    handletextpad=0.3,
    borderaxespad=0.4,
)
inset = b.inset_axes([0.5, 0.14, 0.46, 0.29], facecolor="white", zorder=8)
make_profile(inset, True)
pooled_upper = (
    profile.filter(pl.col("scope") == "Pooled")
    .select((pl.col("estimate") + pl.col("ci")).max())
    .collect()
    .item()
)
inset.set_xlim(-0.2, 4.2)
inset.set_xticks(range(5), ["1", "2", "3", "4", "5+"])
inset.set_yticks([0, 0.01, 0.02])
inset.set_ylim(-0.001, max(0.023, pooled_upper + 0.002))
inset.tick_params(labelsize=6.5, pad=1, length=2)
inset.grid(color=GRID, lw=0.4, ls=(0, (2, 2)), alpha=0.8)
inset.set_axisbelow(True)
for sp in inset.spines.values():
    sp.set_linewidth(0.65)
    sp.set_zorder(30)
inset.text(
    0.04,
    0.95,
    "Pooled",
    transform=inset.transAxes,
    fontsize=7,
    fontweight="bold",
    ha="left",
    va="top",
)
for ax, label in zip(axes, "abc"):
    ax.text(
        -39.0 / width,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )
AB_RC = mpl.rcParams.copy()
import figure2_bottom as panels_cd

CD_RC = mpl.rcParams.copy()
with mpl.rc_context(CD_RC):
    lower = panels_cd.build_panels(fig)
for ax, old, new in zip(lower["axes"], ["c", "d"], ["d", "e"]):
    found = [t for t in ax.texts if t.get_text() == old]
    assert len(found) == 1
    found[0].set_text(new)
allaxes = axes + lower["axes"]
insets = [inset] + lower["insets"]
gap_simple_reference_lines(fig, [*axes, inset])
fig.canvas.draw()
from io import BytesIO

svg = BytesIO()
fig.savefig(svg, format="svg")
svg.seek(0)
apply(svg, DATA / "figure2/style.json", OUTPUT / "figure2.svg")
plt.close(fig)
