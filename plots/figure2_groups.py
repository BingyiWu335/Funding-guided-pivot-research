from __future__ import annotations

import figure2_base as base
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

COLOR = base.COLOR
MARKER = base.MARKER
DOMAIN_COLORS = {
    "Health Sciences": "#D96B5F",
    "Life Sciences": "#62A875",
    "Physical Sciences": "#577FAF",
    "Social Sciences": "#D1A23E",
}
DOMAIN_SHORT = {
    "Health Sciences": "Health",
    "Life Sciences": "Life",
    "Physical Sciences": "Physical",
    "Social Sciences": "Social",
}
DOMAIN_ORDER = (
    "Health Sciences",
    "Life Sciences",
    "Physical Sciences",
    "Social Sciences",
)
DEPARTMENT_ZOOM_X = (0.44, 0.53)
DEPARTMENT_ZOOM_Y = (0.3, 0.41)


def customize_project_type(ax: plt.Axes, data: pd.DataFrame) -> None:
    quadrant_names = {"Switching", "Drift", "Alignment", "Inertia"}
    label_color = {
        base.TYPE_LABEL[row.ptype]: COLOR[row.funder]
        for row in data.itertuples(index=False)
    }
    for text in list(ax.texts):
        if text.get_text() in quadrant_names:
            text.remove()
        elif text.get_text() in label_color:
            text.set_color(label_color[text.get_text()])
    for patch in ax.patches:
        if isinstance(patch, Ellipse):
            patch.set_edgecolor("none")
            patch.set_linewidth(0)


def add_department_uncertainty(ax: plt.Axes, data: pd.DataFrame) -> None:
    for row in data.itertuples(index=False):
        ax.add_patch(
            Ellipse(
                (row.grant_pivot, row.research_pivot),
                width=2 * row.ci_grant,
                height=2 * row.ci_research,
                facecolor=mpl.colors.to_rgba(COLOR[row.funder], 0.14),
                edgecolor="none",
                linewidth=0,
                zorder=2.2,
            )
        )


def scatter_department_units(
    ax: plt.Axes,
    data: pd.DataFrame,
    point_size: float = 42,
    ellipse_alpha: float = 0.07,
) -> None:
    for row in data.itertuples(index=False):
        ax.add_patch(
            Ellipse(
                (row.grant_pivot, row.research_pivot),
                width=2 * row.ci_grant,
                height=2 * row.ci_research,
                facecolor=mpl.colors.to_rgba(COLOR[row.funder], ellipse_alpha),
                edgecolor="none",
                linewidth=0,
                zorder=2.2,
            )
        )
    for funder in ("NIH", "NSF", "NSFC"):
        subset = data.loc[data["funder"].eq(funder)]
        ax.scatter(
            subset["grant_pivot"],
            subset["research_pivot"],
            s=point_size,
            marker=MARKER[funder],
            facecolor=COLOR[funder],
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )


def place_organization_labels(
    ax: plt.Axes,
    data: pd.DataFrame,
    fontsize: float,
    candidates: list[tuple[float, float, str, str]] | None = None,
) -> None:
    if data.empty:
        return
    if candidates is None:
        candidates = [
            (5, 5, "left", "bottom"),
            (5, -5, "left", "top"),
            (-5, 5, "right", "bottom"),
            (-5, -5, "right", "top"),
            (0, 8, "center", "bottom"),
            (0, -8, "center", "top"),
            (9, 0, "left", "center"),
            (-9, 0, "right", "center"),
            (10, 7, "left", "bottom"),
            (-10, 7, "right", "bottom"),
            (10, -7, "left", "top"),
            (-10, -7, "right", "top"),
        ]
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    axes_box = ax.get_window_extent(renderer=renderer).padded(-1)
    coords = data[["grant_pivot", "research_pivot"]].to_numpy(float)
    spread = np.maximum(np.ptp(coords, axis=0), 1e-09)
    scaled = coords / spread
    distances = np.sqrt(((scaled[:, None, :] - scaled[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(np.min(distances, axis=1))
    for idx in order:
        row = data.iloc[int(idx)]
        best = None
        best_score = np.inf
        for rank, (dx, dy, ha, va) in enumerate(candidates):
            trial = ax.annotate(
                str(row["department_code"]),
                (row["grant_pivot"], row["research_pivot"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=fontsize,
                color=COLOR[row["funder"]],
                zorder=5,
            )
            box = trial.get_window_extent(renderer=renderer).padded(0.6)
            collision = sum((base.overlap_area(box, other) for other in occupied))
            outside = (
                max(0, axes_box.x0 - box.x0) * box.height
                + max(0, box.x1 - axes_box.x1) * box.height
                + max(0, axes_box.y0 - box.y0) * box.width
                + max(0, box.y1 - axes_box.y1) * box.width
            )
            score = 20 * collision + 100 * outside + rank
            trial.remove()
            if score < best_score:
                best_score = score
                best = (dx, dy, ha, va, box)
        dx, dy, ha, va, box = best
        ax.annotate(
            str(row["department_code"]),
            (row["grant_pivot"], row["research_pivot"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=COLOR[row["funder"]],
            arrowprops=dict(
                arrowstyle="-",
                color=mpl.colors.to_rgba(COLOR[row["funder"]], 0.65),
                lw=0.4,
                shrinkA=0,
                shrinkB=2,
            )
            if np.hypot(dx, dy) >= 11
            else None,
            zorder=5,
        )
        occupied.append(box)


def add_gradient_dotted_arrow(ax: plt.Axes) -> None:
    start = np.array([0.68, 0.405])
    control = np.array([0.805, 0.385])
    tip = np.array([0.815, 0.555])
    t = np.linspace(0.0, 0.88, 23)
    curve = (
        ((1 - t) ** 2)[:, None] * start
        + (2 * (1 - t) * t)[:, None] * control
        + (t**2)[:, None] * tip
    )
    rgb_start = np.array(mpl.colors.to_rgb("#C6D7DA"))
    rgb_end = np.array(mpl.colors.to_rgb("#3E7D88"))
    mix = np.linspace(0.0, 1.0, len(t))[:, None]
    rgb = rgb_start + mix * (rgb_end - rgb_start)
    alpha = np.linspace(0.38, 0.96, len(t))[:, None]
    rgba = np.concatenate([rgb, alpha], axis=1)
    ax.scatter(
        curve[:, 0],
        curve[:, 1],
        s=np.linspace(5.0, 14.0, len(t)),
        c=rgba,
        edgecolors="none",
        transform=ax.transAxes,
        clip_on=False,
        zorder=8.5,
    )
    t_head = 0.88
    tangent = 2 * (1 - t_head) * (control - start) + 2 * t_head * (tip - control)
    tip_px = ax.transAxes.transform(tip)
    tangent_px = ax.transAxes.transform(tip + tangent) - tip_px
    tangent_px /= np.linalg.norm(tangent_px)
    perpendicular_px = np.array([-tangent_px[1], tangent_px[0]])
    base_px = tip_px - 10.0 * tangent_px
    head_px = np.vstack(
        [tip_px, base_px + 4.7 * perpendicular_px, base_px - 4.7 * perpendicular_px]
    )
    head_axes = ax.transAxes.inverted().transform(head_px)
    ax.add_patch(
        mpl.patches.Polygon(
            head_axes,
            closed=True,
            transform=ax.transAxes,
            facecolor="#3E7D88",
            edgecolor="none",
            clip_on=False,
            zorder=8.7,
        )
    )


def draw_department_hierarchy(ax: plt.Axes, data: pd.DataFrame) -> None:
    med_grant = float(data["reference_grant_pivot"].iloc[0])
    med_research = float(data["reference_research_pivot"].iloc[0])
    base.open_axis(ax, "Grant Pivot", "Research Pivot")
    base.draw_reference_frame(ax, med_grant, med_research)
    ax.set_xlim(0.26, 0.66)
    ax.set_ylim(0.185, 0.57)
    ax.set_xticks(np.arange(0.3, 0.66, 0.05))
    ax.set_yticks(np.arange(0.2, 0.56, 0.05))
    scatter_department_units(ax, data, point_size=42)
    in_zoom = data["grant_pivot"].between(*DEPARTMENT_ZOOM_X) & data[
        "research_pivot"
    ].between(*DEPARTMENT_ZOOM_Y)
    place_organization_labels(ax, data.loc[~in_zoom].copy(), fontsize=6.9)
    ax.add_patch(
        mpl.patches.Rectangle(
            (DEPARTMENT_ZOOM_X[0], DEPARTMENT_ZOOM_Y[0]),
            DEPARTMENT_ZOOM_X[1] - DEPARTMENT_ZOOM_X[0],
            DEPARTMENT_ZOOM_Y[1] - DEPARTMENT_ZOOM_Y[0],
            fill=False,
            edgecolor="#71838E",
            linewidth=0.7,
            linestyle=(0, (3, 2)),
            zorder=4,
        )
    )
    add_gradient_dotted_arrow(ax)
    move_department_legend(ax)
    ax.text(
        0.985,
        0.965,
        f"Grant median {med_grant:.2f}",
        transform=ax.transAxes,
        color=base.REFERENCE,
        fontsize=7.5,
        va="top",
        ha="right",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5),
    )
    ax.text(
        0.657,
        med_research + 0.007,
        f"Research median {med_research:.2f}",
        color=base.REFERENCE,
        fontsize=7.5,
        ha="right",
        va="bottom",
    )
    zoom = ax.inset_axes([0.49, 0.56, 0.33, 0.37], zorder=9)
    zoom.set_facecolor((1, 1, 1, 0.97))
    zoom.set_xlim(*DEPARTMENT_ZOOM_X)
    zoom.set_ylim(*DEPARTMENT_ZOOM_Y)
    zoom.set_xticks([0.45, 0.48, 0.51])
    zoom.set_yticks([0.3, 0.35, 0.4])
    zoom.tick_params(labelsize=6.1, length=1.8, width=0.55, pad=0.8)
    zoom.grid(color="#DFE4E7", linewidth=0.34, linestyle=(0, (2, 2)))
    zoom.set_axisbelow(True)
    zoom.axvline(med_grant, color="#8C959C", linewidth=0.6, linestyle=(0, (4, 3)))
    zoom.axhline(med_research, color="#8C959C", linewidth=0.6, linestyle=(0, (4, 3)))
    zoom.spines["top"].set_visible(False)
    zoom.spines["right"].set_visible(False)
    zoom.spines["left"].set_linewidth(0.6)
    zoom.spines["bottom"].set_linewidth(0.6)
    zoom.set_title("Dense region", loc="left", fontsize=6.9, fontweight="bold", pad=1.0)
    dense = data.loc[in_zoom].copy()
    scatter_department_units(zoom, dense, point_size=19, ellipse_alpha=0.09)
    place_organization_labels(
        zoom,
        dense,
        fontsize=5.3,
        candidates=[
            (3, 3, "left", "bottom"),
            (3, -3, "left", "top"),
            (-3, 3, "right", "bottom"),
            (-3, -3, "right", "top"),
            (0, 5, "center", "bottom"),
            (0, -5, "center", "top"),
            (6, 0, "left", "center"),
            (-6, 0, "right", "center"),
            (7, 5, "left", "bottom"),
            (-7, 5, "right", "bottom"),
            (7, -5, "left", "top"),
            (-7, -5, "right", "top"),
        ],
    )


def place_field_labels(ax: plt.Axes, fields: pd.DataFrame) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    candidates = [
        (4, 4, "left", "bottom"),
        (4, -4, "left", "top"),
        (-4, 4, "right", "bottom"),
        (-4, -4, "right", "top"),
        (0, 6, "center", "bottom"),
        (0, -6, "center", "top"),
        (7, 0, "left", "center"),
        (-7, 0, "right", "center"),
        (8, 6, "left", "bottom"),
        (-8, 6, "right", "bottom"),
        (8, -6, "left", "top"),
        (-8, -6, "right", "top"),
    ]
    occupied = []
    axes_box = ax.get_window_extent(renderer=renderer).padded(-1)
    coords = fields[["grant_pivot", "research_pivot"]].to_numpy(float)
    spread = np.maximum(np.ptp(coords, axis=0), 1e-09)
    scaled = coords / spread
    distances = np.sqrt(((scaled[:, None, :] - scaled[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(np.min(distances, axis=1))
    for idx in order:
        row = fields.iloc[int(idx)]
        best = None
        best_score = np.inf
        for rank, (dx, dy, ha, va) in enumerate(candidates):
            trial = ax.annotate(
                row["field_abbr"],
                (row["grant_pivot"], row["research_pivot"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=8.2,
                color=DOMAIN_COLORS[row["domain"]],
                zorder=5,
            )
            box = trial.get_window_extent(renderer=renderer).padded(0.5)
            collision = sum((base.overlap_area(box, other) for other in occupied))
            outside = (
                max(0, axes_box.x0 - box.x0) * box.height
                + max(0, box.x1 - axes_box.x1) * box.height
                + max(0, axes_box.y0 - box.y0) * box.width
                + max(0, box.y1 - axes_box.y1) * box.width
            )
            score = 20 * collision + 100 * outside + rank
            trial.remove()
            if score < best_score:
                best_score = score
                best = (dx, dy, ha, va, box)
        dx, dy, ha, va, box = best
        ax.annotate(
            row["field_abbr"],
            (row["grant_pivot"], row["research_pivot"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8.2,
            color=DOMAIN_COLORS[row["domain"]],
            arrowprops=dict(
                arrowstyle="-",
                color=mpl.colors.to_rgba(DOMAIN_COLORS[row["domain"]], 0.58),
                lw=0.35,
                shrinkA=0,
                shrinkB=1.5,
            )
            if np.hypot(dx, dy) >= 9
            else None,
            zorder=5,
        )
        occupied.append(box)


def add_field_inset(ax: plt.Axes, fields: pd.DataFrame) -> None:
    inset = ax.inset_axes([0.035, 0.49, 0.44, 0.46], zorder=9)
    inset.set_facecolor((1, 1, 1, 0.965))
    inset.grid(axis="both", color="#DFE4E7", linewidth=0.36, linestyle=(0, (2, 2)))
    inset.set_axisbelow(True)
    median_grant = float(fields["grant_pivot"].median())
    median_research = float(fields["research_pivot"].median())
    inset.axvline(
        median_grant, color="#8C959C", linewidth=0.7, linestyle=(0, (4, 3)), zorder=2
    )
    inset.axhline(
        median_research, color="#8C959C", linewidth=0.7, linestyle=(0, (4, 3)), zorder=2
    )
    for row in fields.itertuples(index=False):
        inset.add_patch(
            Ellipse(
                (row.grant_pivot, row.research_pivot),
                width=2 * row.ci_grant,
                height=2 * row.ci_research,
                facecolor=mpl.colors.to_rgba(DOMAIN_COLORS[row.domain], 0.14),
                edgecolor="none",
                linewidth=0,
                zorder=3,
            )
        )
    for domain in DOMAIN_ORDER:
        subset = fields.loc[fields["domain"].eq(domain)]
        inset.scatter(
            subset["grant_pivot"],
            subset["research_pivot"],
            s=22,
            color=DOMAIN_COLORS[domain],
            edgecolor="white",
            linewidth=0.35,
            label=DOMAIN_SHORT[domain],
            zorder=4,
        )
    xpad = 0.025
    ypad = 0.03
    inset.set_xlim(
        fields["grant_pivot"].min() - xpad, fields["grant_pivot"].max() + xpad
    )
    inset.set_ylim(
        fields["research_pivot"].min() - ypad, fields["research_pivot"].max() + ypad
    )
    inset.set_title(
        "OpenAlex Fields", loc="left", fontsize=9.4, fontweight="bold", pad=1.2
    )
    inset.set_xlabel("Grant Pivot", fontsize=8.5, labelpad=0.8)
    inset.set_ylabel("Research Pivot", fontsize=8.5, labelpad=0.8)
    inset.tick_params(labelsize=8.0, length=1.9, width=0.6, pad=0.9)
    inset.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.91),
        frameon=False,
        fontsize=6.4,
        ncol=2,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.55,
        labelspacing=0.15,
        borderaxespad=0,
    )
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_linewidth(0.65)
    inset.spines["bottom"].set_linewidth(0.65)
    place_field_labels(inset, fields)


def move_department_legend(ax: plt.Axes) -> None:
    old = ax.get_legend()
    if old is not None:
        old.remove()
    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker=MARKER[funder],
            markersize=6.4,
            markerfacecolor=COLOR[funder],
            markeredgecolor="white",
            markeredgewidth=0.6,
            linestyle="none",
            label=funder,
        )
        for funder in ("NIH", "NSF", "NSFC")
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=8.2,
        ncol=3,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.25,
    )
