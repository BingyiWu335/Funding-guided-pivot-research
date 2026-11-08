import figure2_base as base
import figure2_groups as old
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D


def department_panel(ax, data, fields):
    old.draw_department_hierarchy(ax, data)
    for patch in list(ax.patches):
        if isinstance(patch, mpl.patches.Rectangle):
            x, y = patch.get_xy()
            w, h = (patch.get_width(), patch.get_height())
            patch.remove()
            for xs, ys in [
                ([x, x + w], [y, y]),
                ([x, x + w], [y + h, y + h]),
                ([x, x], [y, y + h]),
                ([x + w, x + w], [y, y + h]),
            ]:
                ax.plot(xs, ys, color="#71838E", lw=0.7, ls=(0, (3, 2)), zorder=4)
    for t in list(ax.texts):
        if "median" in t.get_text().lower():
            t.remove()
    ax.set_ylabel("Trajectory Pivot", fontsize=10.5)
    ax.set_xlabel("Grant Pivot", fontsize=10.5)
    ax.tick_params(labelsize=9)
    zoom = ax.child_axes[0]
    zoom.set_facecolor("white")
    zoom.patch.set_alpha(1)
    zoom.set_axes_locator(
        mpl.axes._base._TransformedBoundsLocator(
            [0.715, 0.08, 0.285, 0.44], ax.transAxes
        )
    )
    zoom.set_title("", loc="left")
    zoom.set_title("")
    zoom.set_xticks([0.45, 0.48, 0.51])
    zoom.set_yticks([0.3, 0.4])
    zoom.tick_params(labelsize=6.5)
    for t in list(zoom.texts):
        t.remove()
    dense = data.loc[
        data.grant_pivot.between(*old.DEPARTMENT_ZOOM_X)
        & data.research_pivot.between(*old.DEPARTMENT_ZOOM_Y)
    ]
    offsets = [
        (dx, dy, ha, va)
        for gap in [3, 5, 7, 10, 13]
        for dx, dy, ha, va in [
            (gap, gap, "left", "bottom"),
            (gap, -gap, "left", "top"),
            (-gap, gap, "right", "bottom"),
            (-gap, -gap, "right", "top"),
            (0, gap, "center", "bottom"),
            (0, -gap, "center", "top"),
        ]
    ]
    old.place_organization_labels(zoom, dense, fontsize=5.8, candidates=offsets)
    for coll in list(ax.collections):
        if len(coll.get_offsets()) == 23 and np.max(coll.get_offsets()[:, 0]) < 1:
            coll.remove()
    for patch in list(ax.patches):
        if isinstance(patch, mpl.patches.Polygon):
            patch.remove()
    ax.annotate(
        "",
        xy=(0.7, 0.2),
        xytext=(0.57, 0.36),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "connectionstyle": "arc3,rad=.25",
            "lw": 1.0,
            "color": "#6E9DA5",
            "linestyle": (0, (1, 2)),
        },
    )
    old.add_field_inset(ax, fields)
    field = ax.child_axes[-1]
    field.set_facecolor("white")
    field.patch.set_alpha(1)
    field.set_axes_locator(
        mpl.axes._base._TransformedBoundsLocator([0.04, 0.56, 0.43, 0.43], ax.transAxes)
    )
    field.set_title("", loc="left")
    field.set_title("")
    field.set_xlabel("")
    field.set_ylabel("")
    field.tick_params(labelsize=6.5)
    for line in list(field.lines):
        line.remove()
    for t in list(field.texts):
        t.remove()
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    legend = field.get_legend()
    if legend:
        legend.set_bbox_to_anchor((0, 0.98))
        legend._loc = 2
        for t in legend.get_texts():
            t.set_fontsize(6)
        fig.canvas.draw()
        occupied.append(legend.get_window_extent(renderer).padded(1))
    coords = fields[["grant_pivot", "research_pivot"]].to_numpy(float)
    span = np.maximum(np.ptp(coords, axis=0), 1e-09)
    d = np.sqrt(
        ((coords[:, None, :] / span - coords[None, :, :] / span) ** 2).sum(axis=2)
    )
    np.fill_diagonal(d, np.inf)
    candidates = [
        (3, 3, "left", "bottom"),
        (3, -3, "left", "top"),
        (-3, 3, "right", "bottom"),
        (-3, -3, "right", "top"),
        (0, 6, "center", "bottom"),
        (0, -6, "center", "top"),
        (7, 0, "left", "center"),
        (-7, 0, "right", "center"),
        (8, 5, "left", "bottom"),
        (-8, -5, "right", "top"),
    ]
    for idx in np.argsort(np.min(d, axis=1)):
        r = fields.iloc[int(idx)]
        best = None
        score = float("inf")
        box = field.get_window_extent(renderer).padded(-1)
        for rank, (dx, dy, ha, va) in enumerate(candidates):
            a = field.annotate(
                r.field_abbr,
                (r.grant_pivot, r.research_pivot),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=6.5,
                color=old.DOMAIN_COLORS[r.domain],
            )
            b = a.get_window_extent(renderer).padded(0.3)
            overflow = (
                max(0, box.x0 - b.x0)
                + max(0, b.x1 - box.x1)
                + max(0, box.y0 - b.y0)
                + max(0, b.y1 - box.y1)
            )
            value = (
                30 * sum((base.overlap_area(b, z) for z in occupied))
                + 100 * overflow
                + rank
            )
            a.remove()
            if value < score:
                score = value
                best = (dx, dy, ha, va, b)
        dx, dy, ha, va, b = best
        field.annotate(
            r.field_abbr,
            (r.grant_pivot, r.research_pivot),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=6.5,
            color=old.DOMAIN_COLORS[r.domain],
            zorder=5,
        )
        occupied.append(b)
    if ax.get_legend():
        ax.get_legend().remove()
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                ls="none",
                marker=base.MARKER[f],
                mfc=base.COLOR[f],
                mec="white",
                ms=5.8,
                label=f,
            )
            for f in base.FUNDERS
        ],
        loc="upper left",
        bbox_to_anchor=(0.53, 1),
        fontsize=8,
        ncol=1,
        handletextpad=0.35,
        labelspacing=0.2,
        borderaxespad=0.15,
    )
    main = data.loc[
        ~(
            data.grant_pivot.between(*old.DEPARTMENT_ZOOM_X)
            & data.research_pivot.between(*old.DEPARTMENT_ZOOM_Y)
        )
    ]
    replace_point_labels(
        ax,
        main,
        "department_code",
        lambda r: base.COLOR[r.funder],
        7.0,
        point_radius=4,
        point_data=data,
    )
    replace_point_labels(
        zoom,
        dense,
        "department_code",
        lambda r: base.COLOR[r.funder],
        5.8,
        point_radius=3,
    )
    replace_point_labels(
        field,
        fields,
        "field_abbr",
        lambda r: old.DOMAIN_COLORS[r.domain],
        6.5,
        point_radius=3,
    )
    return [zoom, field]


def replace_point_labels(
    ax, data, column, color, fontsize, point_radius, point_data=None
):
    for t in list(ax.texts):
        t.remove()
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox

    points = ax.transData.transform(
        (data if point_data is None else point_data)[
            ["grant_pivot", "research_pivot"]
        ].to_numpy(float)
    )
    radius = point_radius * fig.dpi / 72 + 1.5
    obstacles = [
        Bbox.from_extents(x - radius, y - radius, x + radius, y + radius)
        for x, y in points
    ]
    if ax.get_legend():
        obstacles.append(ax.get_legend().get_window_extent(renderer).padded(3))
    obstacles += [
        child.get_window_extent(renderer).padded(8) for child in ax.child_axes
    ]
    outer = ax.get_window_extent(renderer).padded(-2)
    occupied = []
    candidates = [
        (dx, dy, ha, va)
        for gap in [5, 7, 10, 13, 17, 22]
        for dx, dy, ha, va in [
            (gap, gap, "left", "bottom"),
            (gap, -gap, "left", "top"),
            (-gap, gap, "right", "bottom"),
            (-gap, -gap, "right", "top"),
            (0, gap, "center", "bottom"),
            (0, -gap, "center", "top"),
            (gap, 0, "left", "center"),
            (-gap, 0, "right", "center"),
        ]
    ]
    for _, r in data.iterrows():
        best = None
        score = float("inf")
        for rank, (dx, dy, ha, va) in enumerate(candidates):
            a = ax.annotate(
                str(r[column]),
                (r.grant_pivot, r.research_pivot),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=fontsize,
                color=color(r),
                zorder=6,
            )
            b = a.get_window_extent(renderer).padded(1.5)
            outside = (
                max(0, outer.x0 - b.x0) * b.height
                + max(0, b.x1 - outer.x1) * b.height
                + max(0, outer.y0 - b.y0) * b.width
                + max(0, b.y1 - outer.y1) * b.width
            )
            value = (
                100 * sum((base.overlap_area(b, z) for z in obstacles + occupied))
                + 150 * outside
                + rank
            )
            a.remove()
            if value < score:
                score = value
                best = (dx, dy, ha, va, b)
        dx, dy, ha, va, b = best
        ax.annotate(
            str(r[column]),
            (r.grant_pivot, r.research_pivot),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color(r),
            zorder=6,
        )
        occupied.append(b)


def programme_panel(ax, data):
    base.draw_b(ax, data)
    old.customize_project_type(ax, data)
    ax.set_ylabel("Trajectory Pivot", fontsize=10.5)
    ax.set_xlabel("Grant Pivot", fontsize=10.5)
    ax.tick_params(labelsize=9)
    for t in ax.texts:
        if t.get_text() == "Less Developed Regions":
            t.set_text("Fund for\nLess Developed Regions")
            t.set_position((0.574, 0.507))
            t.set_fontsize(7.3)
            t.set_ha("right")
            t.set_linespacing(1.15)
        if t.get_text() == "Renewable":
            t.set_fontweight("normal")
            t.set_color("#8A8A8A")
            t.set_fontsize(7)
        if t.get_text() == "Continuing Grant":
            t.set_position((0.466, 0.267))
            t.set_ha("left")


def clear_grid_under_labels(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [
        t.get_window_extent(renderer).padded(2)
        for t in fig.findobj(mpl.text.Text)
        if t.get_visible() and t.get_text().strip()
    ]
    for ax in axes:
        inset_boxes = [
            child.get_window_extent(renderer).padded(2) for child in ax.child_axes
        ]
        lines = list(ax.get_xgridlines()) + list(ax.get_ygridlines()) + list(ax.lines)
        for line in lines:
            if not line.get_visible() or line.get_linestyle() in [
                "None",
                "none",
                "",
                " ",
            ]:
                continue
            points = line.get_transform().transform(line.get_xydata())
            if len(points) != 2:
                continue
            horizontal = abs(points[0, 1] - points[1, 1]) < 1e-06
            vertical = abs(points[0, 0] - points[1, 0]) < 1e-06
            if not (horizontal or vertical):
                continue
            which = 0 if horizontal else 1
            other = 1 - which
            lo, hi = sorted(points[:, which])
            constant = points[0, other]
            bounds = ax.get_window_extent(renderer)
            lo = max(lo, bounds.x0 if horizontal else bounds.y0)
            hi = min(hi, bounds.x1 if horizontal else bounds.y1)
            excluded = []
            for b in texts + inset_boxes:
                blo, bhi = (b.y0, b.y1) if horizontal else (b.x0, b.x1)
                if blo <= constant <= bhi:
                    left, right = (b.x0, b.x1) if horizontal else (b.y0, b.y1)
                    if right > lo and left < hi:
                        excluded.append((max(left, lo), min(right, hi)))
            if not excluded:
                continue
            segments = []
            cursor = lo
            for left, right in sorted(excluded):
                if left > cursor:
                    segments.append((cursor, left))
                cursor = max(cursor, right)
            if cursor < hi:
                segments.append((cursor, hi))
            line.set_visible(False)
            for left, right in segments:
                pts = (
                    np.array([[left, constant], [right, constant]])
                    if horizontal
                    else np.array([[constant, left], [constant, right]])
                )
                fractions = fig.transFigure.inverted().transform(pts)
                ax.add_line(
                    Line2D(
                        fractions[:, 0],
                        fractions[:, 1],
                        transform=fig.transFigure,
                        color=line.get_color(),
                        alpha=line.get_alpha(),
                        lw=line.get_linewidth(),
                        linestyle=line._unscaled_dash_pattern,
                        zorder=line.get_zorder(),
                        clip_on=False,
                    )
                )
