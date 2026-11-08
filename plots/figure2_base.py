from __future__ import annotations

import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.transforms import Bbox

BLACK = "#1A1A1A"
GRID = "#D9DDE1"
REFERENCE = "#858C93"
COLOR = {"NIH": "#4888C8", "NSF": "#E97B7A", "NSFC": "#0C8B8E"}
MARKER = {"NIH": "o", "NSF": "s", "NSFC": "^"}
FUNDERS = ("NIH", "NSF", "NSFC")
TYPE_LABEL = {
    "R01": "R01",
    "R21": "R21",
    "R03": "R03",
    "R15": "R15",
    "RF1": "RF1",
    "NSF_SG": "Standard Grant",
    "NSF_CG": "Continuing Grant",
    "NSFC_GENERAL": "General Program",
    "NSFC_YOUNG": "Young Scientists Fund",
    "NSFC_REGIONAL": "Less Developed Regions",
}
TYPE_OFFSET = {
    "R01": (-0.003, -0.01, "right"),
    "R21": (0.004, 0.007, "left"),
    "R03": (-0.003, 0.009, "right"),
    "R15": (0.004, -0.009, "left"),
    "RF1": (0.004, 0.003, "left"),
    "NSF_SG": (0.004, -0.009, "left"),
    "NSF_CG": (0.004, -0.009, "left"),
    "NSFC_GENERAL": (-0.004, 0.01, "right"),
    "NSFC_YOUNG": (-0.004, 0.01, "right"),
    "NSFC_REGIONAL": (-0.004, 0.01, "right"),
}


def open_axis(ax, xlabel: str, ylabel: str):
    ax.grid(True, color=GRID, lw=0.55, ls=(0, (2.5, 2.5)), alpha=0.75)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=9.8, labelpad=3.0)
    ax.set_ylabel(ylabel, fontsize=9.8, labelpad=4.0)
    ax.tick_params(labelsize=8.6, pad=2.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_reference_frame(ax, med_grant: float, med_research: float):
    ax.axvline(med_grant, color=REFERENCE, lw=0.9, ls=(0, (5, 3)), zorder=1)
    ax.axhline(med_research, color=REFERENCE, lw=0.9, ls=(0, (5, 3)), zorder=1)


def draw_b(ax, data: pd.DataFrame):
    med_grant = float(data["median_grant"].iloc[0])
    med_research = float(data["median_research"].iloc[0])
    open_axis(ax, "Grant Pivot", "Research Pivot")
    ax.set_xlim(0.4, 0.58)
    ax.set_ylim(0.25, 0.52)
    ax.set_xticks([0.4, 0.45, 0.5, 0.55])
    ax.set_yticks([0.25, 0.35, 0.45])
    draw_reference_frame(ax, med_grant, med_research)
    qstyle = dict(color="#B8BEC4", fontsize=7.0, style="italic", zorder=1)
    ax.text(0.468, 0.515, "Switching", ha="right", va="top", **qstyle)
    ax.text(0.576, 0.515, "Drift", ha="right", va="top", **qstyle)
    ax.text(0.404, 0.254, "Alignment", ha="left", va="bottom", **qstyle)
    ax.text(0.576, 0.254, "Inertia", ha="right", va="bottom", **qstyle)
    renewable = FancyBboxPatch(
        (0.402, 0.259),
        0.059,
        0.041,
        boxstyle="round,pad=0,rounding_size=0.010",
        fill=False,
        ec="#5F6B76",
        lw=1.0,
        ls=(0, (4, 3)),
        zorder=2,
    )
    ax.add_patch(renewable)
    ax.text(
        0.403,
        0.303,
        "Renewable",
        color="#5F6B76",
        fontsize=7.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    for row in data.itertuples(index=False):
        color = COLOR[row.funder]
        ax.add_patch(
            Ellipse(
                (row.grant_pivot, row.research_pivot),
                width=2 * row.ci_grant,
                height=2 * row.ci_research,
                facecolor=to_rgba(color, 0.12),
                edgecolor=to_rgba(color, 0.5),
                linewidth=0.8,
                zorder=2.5,
            )
        )
        ax.plot(
            row.grant_pivot,
            row.research_pivot,
            marker=MARKER[row.funder],
            ms=6.0,
            mfc=color,
            mec="white",
            mew=0.7,
            ls="none",
            zorder=3,
        )
        dx, dy, ha = TYPE_OFFSET[row.ptype]
        ax.annotate(
            TYPE_LABEL[row.ptype],
            (row.grant_pivot, row.research_pivot),
            xytext=(row.grant_pivot + dx, row.research_pivot + dy),
            fontsize=7.2,
            color=BLACK,
            ha=ha,
            va="center",
            zorder=4,
        )
    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKER[funder],
            color="none",
            mfc=COLOR[funder],
            mec="white",
            mew=0.6,
            ms=5.8,
            label=funder,
        )
        for funder in FUNDERS
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        fontsize=7.2,
        ncol=1,
        handletextpad=0.35,
        labelspacing=0.2,
        borderaxespad=0.25,
    )


def overlap_area(a: Bbox, b: Bbox) -> float:
    width = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    height = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    return width * height
