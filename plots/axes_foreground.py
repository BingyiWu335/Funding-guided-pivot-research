from matplotlib.lines import Line2D


def raise_axes_to_foreground(fig, zero_axis):
    fig.canvas.draw()
    records = []
    for ax in fig.axes:
        grids = []
        for grid in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
            if not grid.get_visible():
                continue
            vertices = grid.get_transform().transform(grid.get_path().vertices)
            midpoint = (vertices[0] + vertices[-1]) / 2
            if not ax.bbox.contains(*midpoint):
                continue
            grids.append(grid)
        for grid in grids:
            clone = Line2D([], [])
            clone.update_from(grid)
            clone.set_data(grid.get_data())
            clone.set_transform(grid.get_transform())
            clone.set_zorder(0.5)
            clone.set_gid("background_grid")
            clone.set_clip_on(True)
            clone.set_clip_path(ax.patch)
            ax.add_line(clone)
        ax.grid(False, which="both", axis="both")
        ax.xaxis.set_zorder(1000)
        ax.yaxis.set_zorder(1000)
        for spine in ax.spines.values():
            spine.set_zorder(1001)
        for label in [
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
            ax.xaxis.label,
            ax.yaxis.label,
        ]:
            label.set_zorder(1001)
        for tick in [
            *ax.xaxis.get_major_ticks(),
            *ax.yaxis.get_major_ticks(),
            *ax.xaxis.get_minor_ticks(),
            *ax.yaxis.get_minor_ticks(),
        ]:
            tick.tick1line.set_zorder(1001)
            tick.tick2line.set_zorder(1001)
        foreground_spines = [
            s.get_zorder() for s in ax.spines.values() if s.get_visible()
        ]
        assert all((z == 1001 for z in foreground_spines))
        records.append(
            {
                "axis_index": len(records),
                "xaxis_zorder": ax.xaxis.get_zorder(),
                "yaxis_zorder": ax.yaxis.get_zorder(),
                "visible_spine_zorders": foreground_spines,
                "background_grids": len(grids),
            }
        )
    zero = [x for x in zero_axis.lines if x.get_gid() == "zero_baseline"]
    assert len(zero) == 1
    zero[0].set_zorder(1002)
    return {
        "axes": records,
        "e_zero_line_zorder": zero[0].get_zorder(),
        "background_grids_retained": True,
    }
