TICK_SIZE_PT = 6.9


def apply_readable_b_axes(fig, b, bf, mean_ylim=(0.8, 1.2)):
    b.set_xticks([0, 1], ["Grant pivot\nbelow median", "Grant pivot\nabove median"])
    b.set_ylim(*mean_ylim)
    b.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    bf.set_xlabel("Citation impact", fontsize=6.0, labelpad=4)
    for text in bf.texts:
        label = text.get_text()
        label = label.replace("RFA main", "RFA")
        label = label.replace("RFA × Grant Aligning", "RFA × Below median")
        label = label.replace("RFA × Grant Alignment", "RFA × (1 − Grant pivot)")
        text.set_text(label)
    all_axes = list(fig.axes)
    for ax in list(all_axes):
        all_axes.extend((child for child in ax.child_axes if child not in all_axes))
    for ax in all_axes:
        ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE_PT)
    fig.canvas.draw()
    record = []
    for index, ax in enumerate(all_axes):
        labels = [
            label
            for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]
            if label.get_visible() and label.get_text()
        ]
        sizes = sorted({float(label.get_fontsize()) for label in labels})
        assert all((abs(value - TICK_SIZE_PT) < 1e-09 for value in sizes))
        record.append(
            {
                "axis_index": index,
                "tick_font_sizes_pt": sizes,
                "xlabels": [x.get_text() for x in ax.get_xticklabels()],
                "ylimits": list(ax.get_ylim()),
            }
        )
    return record
