import json
import math

import figure2_layout as draw
from common import DATA, frame

R = DATA / "figure2"
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.transforms import Bbox

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
COLOR = draw.base.COLOR
MARKER = draw.base.MARKER
LAYOUT = json.loads((R / "layout.json").read_text("utf8"))
FIELD_ABBR = {
    11: "AGR",
    12: "ART",
    13: "BGM",
    14: "BUS",
    15: "CE",
    16: "CHE",
    17: "CS",
    18: "DEC",
    19: "EAR",
    20: "ECON",
    21: "ENE",
    22: "ENG",
    23: "ENV",
    24: "IMM",
    25: "MAT",
    26: "MATH",
    27: "MED",
    28: "NEU",
    29: "NUR",
    30: "PHA",
    31: "PHY",
    32: "PSY",
    33: "SOC",
    34: "VET",
    35: "DEN",
    36: "HP",
}
TYPES = draw.base.TYPE_LABEL


def normalized_rect(frame):
    l, t, r, b = frame
    return [l / 792, (503 - b) / 503, (r - l) / 792, (b - t) / 503]


def rel_rect(frame, parent):
    l, t, r, b = frame
    pl, pt, pr, pb = parent
    return [
        (l - pl) / (pr - pl),
        (pb - b) / (pb - pt),
        (r - l) / (pr - pl),
        (b - t) / (pb - pt),
    ]


def set_inset_rect(ax, rect, parent):
    ax.set_axes_locator(
        mpl.axes._base._TransformedBoundsLocator(rect, parent.transAxes)
    )


def pixel_box(ax, row):
    return ax.transData.transform([row.grant_pivot, row.research_pivot])


def old_offset(scope, text, oldrow, limits):
    records = [r for r in LAYOUT["labels"] if r["scope"] == scope and r["text"] == text]
    if not records:
        return None
    frame = LAYOUT["frames"][
        {
            "main": "main",
            "zoom": "zoom_background",
            "fields": "fields",
            "programmes": "programmes",
        }[scope]
    ]
    l, t, r, b = frame
    xl, xr, yl, yr = limits
    oldx = l + (oldrow.grant_pivot - xl) / (xr - xl) * (r - l)
    oldy = b - (oldrow.research_pivot - yl) / (yr - yl) * (b - t)
    candidates = sorted(
        records, key=lambda a: (a["x_pt"] - oldx) ** 2 + (a["y_pt"] - oldy) ** 2
    )
    v = candidates[0]
    return (v["x_pt"] - oldx, oldy - v["y_pt"], v["anchor"], "baseline")


def place_labels(
    ax,
    data,
    column,
    color,
    fontsize,
    scope,
    olddata,
    oldlimits,
    avoid_axes=(),
    overrides=None,
    all_points=None,
):
    for t in list(ax.texts):
        t.remove()
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outer = ax.get_window_extent(renderer).padded(-1.5)
    pts = ax.transData.transform(
        data[["grant_pivot", "research_pivot"]].to_numpy(float)
    )
    radius = 4.2 * fig.dpi / 72
    protected = ax.transData.transform(
        (data if all_points is None else all_points)[
            ["grant_pivot", "research_pivot"]
        ].to_numpy(float)
    )
    obstacles = [
        Bbox.from_extents(x - radius, y - radius, x + radius, y + radius)
        for x, y in protected
    ]
    obstacles += [a.get_window_extent(renderer).padded(12) for a in avoid_axes]
    if ax.get_legend():
        obstacles.append(ax.get_legend().get_window_extent(renderer).padded(3))
    occupied = []
    records = []
    offsets = [
        (dx, dy, ha, va)
        for gap in [5, 8, 11, 15, 20, 27, 35]
        for dx, dy, ha, va in [
            (0, gap, "center", "bottom"),
            (0, -gap, "center", "top"),
            (gap, gap, "left", "bottom"),
            (-gap, gap, "right", "bottom"),
            (gap, -gap, "left", "top"),
            (-gap, -gap, "right", "top"),
            (gap, 0, "left", "center"),
            (-gap, 0, "right", "center"),
        ]
    ]
    oldkey = (
        ["funder", column] if "funder" in data and "funder" in olddata else [column]
    )
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    ordering = np.argsort(dist.min(axis=1), kind="stable")
    for idx in ordering:
        row = data.iloc[int(idx)]
        text = str(row[column])
        prior = olddata
        for k in oldkey:
            prior = prior[prior[k].astype(str).eq(str(row[k]))]
        inherited = (
            old_offset(scope, text, prior.iloc[0], oldlimits) if len(prior) else None
        )
        starts = (
            [inherited]
            if inherited and max(abs(inherited[0]), abs(inherited[1])) < 42
            else []
        )
        key = (str(row.get("funder", "")), text)
        if overrides and key in overrides:
            starts = [tuple(overrides[key])]
        candidates = starts + offsets
        best = None
        score = float("inf")
        for rank, (dx, dy, ha, va) in enumerate(candidates):
            a = ax.annotate(
                text,
                (row.grant_pivot, row.research_pivot),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=fontsize,
                color=color(row),
                zorder=15,
                clip_on=False,
            )
            box = a.get_window_extent(renderer).padded(1.2)
            overlap = sum(
                (draw.base.overlap_area(box, z) for z in obstacles + occupied)
            )
            outside = (
                max(0, outer.x0 - box.x0) * box.height
                + max(0, box.x1 - outer.x1) * box.height
                + max(0, outer.y0 - box.y0) * box.width
                + max(0, box.y1 - outer.y1) * box.width
            )
            value = (
                300 * overlap + 500 * outside + rank * 0.06 + math.hypot(dx, dy) * 0.006
            )
            a.remove()
            if value < score:
                score = value
                best = (dx, dy, ha, va, box)
            if overlap == 0 and outside == 0 and (rank == 0):
                break
        dx, dy, ha, va, box = best
        leader = (
            {
                "arrowstyle": "-",
                "color": color(row),
                "alpha": 0.5,
                "lw": 0.4,
                "shrinkA": 1.8,
                "shrinkB": 3,
            }
            if scope == "fields" and math.hypot(dx, dy) > 12
            else None
        )
        ax.annotate(
            text,
            (row.grant_pivot, row.research_pivot),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color(row),
            zorder=15,
            clip_on=False,
            arrowprops=leader,
        )
        occupied.append(box)
        records.append(
            {
                "scope": scope,
                "label": text,
                "key": key,
                "offset_points": [dx, dy],
                "ha": ha,
                "va": va,
                "position_from_Q4": [row.grant_pivot, row.research_pivot],
            }
        )
    return records


def manual_labels(ax, data, column, color, fontsize, scope):
    for t in list(ax.texts):
        t.remove()
    plan = json.loads((R / "labels.json").read_text("utf8"))[scope]
    records = []
    for row in data.itertuples():
        text = str(getattr(row, column))
        funder = str(getattr(row, "funder", ""))
        key = funder + ":" + text if funder else text
        dx, dy, ha, va = plan[key]
        leader = (
            {
                "arrowstyle": "-",
                "color": color(row),
                "alpha": 0.55,
                "lw": 0.4,
                "shrinkA": 1.2,
                "shrinkB": 2.4,
            }
            if scope == "fields"
            and key
            in ["VET", "HP", "PHA", "ECON", "DEC", "CS", "ENG", "MED", "ENV", "NUR"]
            or key in ["NIH:LM", "NIH:MD", "NIH:AA", "NIH:DE", "NIH:HL", "NIH:TR"]
            else None
        )
        ax.annotate(
            text,
            (row.grant_pivot, row.research_pivot),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color(row),
            zorder=15,
            clip_on=False,
            arrowprops=leader,
        )
        records.append(
            {
                "scope": scope,
                "label": text,
                "key": [funder, text],
                "offset_points": [dx, dy],
                "ha": ha,
                "va": va,
                "position_from_Q4": [row.grant_pivot, row.research_pivot],
            }
        )
    assert len(records) == len(plan), (scope, "unexpected missing or extra label")
    return records


def gap_field_leaders(fig, ax):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        t.get_window_extent(renderer).padded(1.8) for t in ax.texts if t.get_text()
    ]
    for annotation in ax.texts:
        arrow = getattr(annotation, "arrow_patch", None)
        if arrow is None:
            continue
        vertices = arrow.get_transform().transform(
            arrow.get_path().interpolated(20).vertices
        )
        arrow.set_visible(False)
        for p, q in zip(vertices[:-1], vertices[1:]):
            direction = q - p
            excluded = []
            for box in boxes:
                low, high = (0.0, 1.0)
                for k, lo, hi in [(0, box.x0, box.x1), (1, box.y0, box.y1)]:
                    if abs(direction[k]) < 1e-12:
                        if not lo <= p[k] <= hi:
                            low, high = (1.0, 0.0)
                            break
                    else:
                        a, b = sorted(
                            ((lo - p[k]) / direction[k], (hi - p[k]) / direction[k])
                        )
                        low = max(low, a)
                        high = min(high, b)
                if high > low:
                    excluded.append((low, high))
            cursor = 0.0
            parts = []
            for lo, hi in sorted(excluded):
                if lo > cursor:
                    parts.append((cursor, lo))
                cursor = max(cursor, hi)
            if cursor < 1:
                parts.append((cursor, 1))
            for lo, hi in parts:
                pair = fig.transFigure.inverted().transform(
                    [p + lo * direction, p + hi * direction]
                )
                ax.add_line(
                    Line2D(
                        pair[:, 0],
                        pair[:, 1],
                        transform=fig.transFigure,
                        color=arrow.get_edgecolor(),
                        alpha=0.5,
                        lw=0.4,
                        zorder=14,
                        clip_on=False,
                    )
                )


def gradient_arrow(ax, region, zoom_rect):
    start = ax.transAxes.inverted().transform(
        ax.transData.transform([region[1] - 0.018, region[2]])
    )
    tip = np.array([zoom_rect[0] - 0.025, zoom_rect[1] + zoom_rect[3] * 0.23])
    start = start + np.array([0.006, -0.012])
    control = np.array([start[0] + 0.35 * (tip[0] - start[0]), tip[1]])
    assert start[0] < control[0] < tip[0], (
        "Arrow must progress to the right without folding"
    )
    ts = np.linspace(0, 0.88, 23)
    curve = (
        (1 - ts[:, None]) ** 2 * start
        + 2 * (1 - ts[:, None]) * ts[:, None] * control
        + ts[:, None] ** 2 * tip
    )
    rgba = np.column_stack(
        [
            np.linspace(mpl.colors.to_rgb("#C6D7DA"), mpl.colors.to_rgb("#3E7D88"), 23),
            np.linspace(0.38, 0.96, 23),
        ]
    )
    ax.scatter(
        curve[:, 0],
        curve[:, 1],
        s=np.linspace(5, 14, 23),
        c=rgba,
        edgecolors="none",
        transform=ax.transAxes,
        zorder=12,
        clip_on=False,
    )
    tangent = 2 * (1 - 0.88) * (control - start) + 2 * 0.88 * (tip - control)
    tip_px = ax.transAxes.transform(tip)
    direction = ax.transAxes.transform(tip + tangent) - tip_px
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    head = np.array(
        [
            tip_px,
            tip_px - direction * 10 + normal * 4.7,
            tip_px - direction * 10 - normal * 4.7,
        ]
    )
    ax.add_patch(
        Polygon(
            ax.transAxes.inverted().transform(head),
            transform=ax.transAxes,
            facecolor="#3E7D88",
            edgecolor="none",
            zorder=13,
            clip_on=False,
        )
    )


def build_panels(fig):
    medians = json.loads((R / "medians.json").read_text("utf8"))["medians"]
    dep = frame(R / "departments.csv")
    program = frame(R / "programmes.csv")
    fields = frame(R / "fields.csv")
    oldprog = frame(R / "programme_label_anchors.csv")
    fields["field_abbr"] = fields.field_id.map(FIELD_ABBR)
    assert fields.field_abbr.notna().all()
    program["median_grant"] = medians["Grant Pivot"]
    program["median_research"] = medians["Trajectory Pivot"]
    previous = json.loads((R / "label_offsets.json").read_text("utf8"))
    zx = (0.444, 0.528)
    zy = (0.308, 0.359)

    def prior_offsets(scope):
        return {
            tuple(v["key"]): (*v["offset_points"], v["ha"], v["va"])
            for v in previous["labels"]
            if v["scope"] == scope
        }

    draw.old.DEPARTMENT_ZOOM_X = zx
    draw.old.DEPARTMENT_ZOOM_Y = zy
    original_hierarchy = draw.old.draw_department_hierarchy
    xlim = tuple(previous["main_c_limits"]["x"])
    ylim = tuple(previous["main_c_limits"]["y"])

    def adjusted_hierarchy(ax, data):
        original_hierarchy(ax, data)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    draw.old.draw_department_hierarchy = adjusted_hierarchy
    ac = fig.add_axes(normalized_rect(LAYOUT["frames"]["main"]))
    ad = fig.add_axes(normalized_rect(LAYOUT["frames"]["programmes"]))
    zoom, field = draw.department_panel(ac, dep, fields)
    zr = rel_rect([390, 384, 507.14848661, 453.47182478], LAYOUT["frames"]["main"])
    fr = rel_rect(LAYOUT["frames"]["fields"], LAYOUT["frames"]["main"])
    set_inset_rect(zoom, zr, ac)
    set_inset_rect(field, fr, ac)
    for t in list(ac.texts):
        if isinstance(t, mpl.text.Annotation) and (not t.get_text()):
            t.remove()
    gradient_arrow(ac, (*zx, *zy), zr)
    draw.programme_panel(ad, program)
    ad.set_ylim(
        0.25,
        max(
            0.56,
            math.ceil(
                (float((program.research_pivot + program.ci_research).max()) + 0.04)
                / 0.02
            )
            * 0.02,
        ),
    )
    ad.set_xlim(
        0.4,
        max(
            0.58,
            math.ceil(
                (float((program.grant_pivot + program.ci_grant).max()) + 0.015) / 0.02
            )
            * 0.02,
        ),
    )
    for p in list(ad.patches):
        if isinstance(p, FancyBboxPatch):
            p.remove()
    renew = program[program.ptype.isin(["R01", "NSF_CG"])]
    rx = min(0.402, float((renew.grant_pivot - renew.ci_grant).min()) - 0.006)
    ry = float((renew.research_pivot - renew.ci_research).min()) - 0.009
    rw = float((renew.grant_pivot + renew.ci_grant).max()) + 0.008 - rx
    rh = float((renew.research_pivot + renew.ci_research).max()) + 0.006 - ry
    renewable_frame = Rectangle(
        (rx, ry),
        rw,
        rh,
        fill=False,
        edgecolor="#BFC1C3",
        lw=0.9,
        ls=(0, (5, 2)),
        zorder=20,
    )
    ad.add_patch(renewable_frame)
    assert all(
        (
            not (rx < r.grant_pivot < rx + rw and ry < r.research_pivot < ry + rh)
            for r in program[~program.ptype.isin(["R01", "NSF_CG"])].itertuples()
        )
    ), "Retained Renewable enclosure must not acquire other types"
    for ax in [ac, ad]:
        ax.set_xlabel("Grant Pivot", fontsize=12.5, labelpad=4)
        ax.set_ylabel("Trajectory Pivot", fontsize=12.5, labelpad=4)
        ax.tick_params(labelsize=10.5, pad=3)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_zorder(30)
    ac.yaxis.set_major_locator(MultipleLocator(0.05))
    ad.set_yticks(np.arange(0.25, ad.get_ylim()[1] + 1e-05, 0.1))
    for ax in [zoom, field]:
        ax.tick_params(labelsize=7.5, pad=1.2)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        for s in ax.spines.values():
            s.set_zorder(30)
    for s in zoom.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.65)
    zoom.set_xticks([0.45, 0.48, 0.51])
    zoom.set_yticks([0.31, 0.33, 0.35])
    zoom.set_xlim(zx)
    zoom.set_ylim(0.305, 0.365)
    zoom.set_autoscale_on(False)
    field.xaxis.set_major_locator(MultipleLocator(0.1))
    field.yaxis.set_major_locator(MultipleLocator(0.1))
    field.set_xlim(
        float(fields.grant_pivot.min()) - 0.035, float(fields.grant_pivot.max()) + 0.03
    )
    field.set_ylim(
        float(fields.research_pivot.min()) - 0.035,
        float(fields.research_pivot.max()) + 0.035,
    )
    field.tick_params(labelsize=6.8)
    if field.get_legend():
        for t in field.get_legend().get_texts():
            t.set_fontsize(6.5)
    for ax in [ac, ad]:
        if ax.get_legend():
            for t in ax.get_legend().get_texts():
                t.set_fontsize(10)
    ac.get_legend().set_bbox_to_anchor((0.63, 1.0))
    labels = []
    inside = dep.grant_pivot.between(*zx) & dep.research_pivot.between(*zy)
    labels += manual_labels(
        ac, dep[~inside], "department_code", lambda r: COLOR[r.funder], 8.2, "main"
    )
    labels += manual_labels(
        zoom, dep[inside], "department_code", lambda r: COLOR[r.funder], 7.2, "zoom"
    )
    labels += manual_labels(
        field,
        fields,
        "field_abbr",
        lambda r: draw.old.DOMAIN_COLORS[r.domain],
        6.3,
        "fields",
    )
    assert len(labels) == 69 and inside.sum() == 14
    for t in list(ad.texts):
        t.remove()
    prog = program.copy()
    prog["display_label"] = prog.ptype.map(TYPES)
    oldp = oldprog.copy()
    oldp["display_label"] = oldp.ptype.map(TYPES)
    labels += place_labels(
        ad,
        prog[~prog.ptype.eq("NSFC_REGIONAL")],
        "display_label",
        lambda r: COLOR[r.funder],
        9,
        "programmes",
        oldp,
        (0.4, 0.58, 0.25, 0.52),
        overrides=prior_offsets("programmes"),
    )
    region = prog[prog.ptype.eq("NSFC_REGIONAL")].iloc[0]
    ad.annotate(
        "Fund for\nLess Developed Regions",
        (region.grant_pivot, region.research_pivot),
        xytext=(-5, 9),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color=COLOR["NSFC"],
        linespacing=1.15,
        zorder=15,
    )
    ad.text(
        rx + 0.004,
        ry + rh - 0.006,
        "Renewable",
        color="#969696",
        fontsize=8.7,
        ha="left",
        va="top",
        zorder=15,
    )
    for text in ad.texts:
        if text.get_text() == "R01":
            text.set_position((-5, -4))
            text.set_ha("right")
            text.set_va("center")
        if text.get_text() == "Continuing Grant":
            text.set_position((0, 5))
            text.set_ha("center")
            text.set_va("bottom")
    for record in labels:
        if record["scope"] == "programmes" and record["label"] == "R01":
            record.update(offset_points=[-5, -4], ha="right", va="center")
        if record["scope"] == "programmes" and record["label"] == "Continuing Grant":
            record.update(offset_points=[0, 5], ha="center", va="bottom")
    ac.text(
        -0.085,
        1.035,
        "c",
        transform=ac.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    ad.text(
        -0.19,
        1.035,
        "d",
        transform=ad.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    for text in fig.findobj(mpl.text.Text):
        text.set_bbox(None)
        text.set_path_effects([])
    assert all((t.get_bbox_patch() is None for t in fig.findobj(mpl.text.Text)))
    gap_field_leaders(fig, field)
    gap_field_leaders(fig, ac)
    gap_field_leaders(fig, zoom)
    draw.clear_grid_under_labels(fig, [ac, ad, zoom, field])
    for line in ac.lines:
        if mpl.colors.to_hex(line.get_color()).lower() == "#71838e":
            line.set_color("black")
            line.set_linewidth(0.75)
            line.set_linestyle((0, (6, 3)))
    assert np.allclose(zoom.get_xlim(), zx) and np.allclose(
        zoom.get_ylim(), [0.305, 0.365]
    )
    return {
        "axes": [ac, ad],
        "insets": [zoom, field],
        "labels": labels,
        "zoom_limits": {"x": list(zoom.get_xlim()), "y": list(zoom.get_ylim())},
    }
