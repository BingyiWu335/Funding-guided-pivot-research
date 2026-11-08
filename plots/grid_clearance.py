from matplotlib.lines import Line2D


def clear_grid_under_labels(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        inv = ax.transAxes.inverted()
        rects = []
        texts = list(ax.texts)
        if ax.get_legend() is not None:
            texts.append(ax.get_legend())
        for text in texts:
            if not text.get_visible():
                continue
            b = text.get_window_extent(renderer).padded(1.6 * fig.dpi / 72)
            a = inv.transform([[b.x0, b.y0], [b.x1, b.y1]])
            rects.append((a[0, 0], a[0, 1], a[1, 0], a[1, 1]))
        for inset in ax.child_axes:
            b = inset.get_tightbbox(renderer).padded(1.6 * fig.dpi / 72)
            a = inv.transform([[b.x0, b.y0], [b.x1, b.y1]])
            rects.append((a[0, 0], a[0, 1], a[1, 0], a[1, 1]))
        lines = []
        for grid in ax.get_xgridlines() + ax.get_ygridlines():
            if not grid.get_visible():
                continue
            v = inv.transform(grid.get_transform().transform(grid.get_path().vertices))
            a, b = (v[0], v[-1])
            vertical = abs(a[0] - b[0]) < 1e-08
            fixed = a[0] if vertical else a[1]
            cuts = []
            for x0, y0, x1, y1 in rects:
                if vertical and x0 <= fixed <= x1:
                    cuts.append((max(0, y0), min(1, y1)))
                if not vertical and y0 <= fixed <= y1:
                    cuts.append((max(0, x0), min(1, x1)))
            intervals = []
            start = 0.0
            for lo, hi in sorted(((l, h) for l, h in cuts if h > l)):
                if lo > start:
                    intervals.append((start, lo))
                start = max(start, hi)
            if start < 1:
                intervals.append((start, 1.0))
            for lo, hi in intervals:
                x, y = (
                    ([fixed, fixed], [lo, hi])
                    if vertical
                    else ([lo, hi], [fixed, fixed])
                )
                line = Line2D(
                    x,
                    y,
                    transform=ax.transAxes,
                    color=grid.get_color(),
                    linewidth=grid.get_linewidth(),
                    linestyle=getattr(
                        grid, "_unscaled_dash_pattern", grid.get_linestyle()
                    ),
                    alpha=grid.get_alpha(),
                    zorder=0.5,
                )
                lines.append(line)
        ax.grid(False)
        for line in lines:
            ax.add_line(line)
