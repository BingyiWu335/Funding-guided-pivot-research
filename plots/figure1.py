import polars as pl
from common import DATA, OUTPUT
from lxml import etree

source = DATA / "figure1"
tree = etree.parse(source / "artwork.svg")
elements = {element.get("id"): element for element in tree.iter() if element.get("id")}
shares = (
    pl.scan_csv(source / "histogram_counts.csv")
    .with_columns(
        (100 * pl.col("n") / pl.col("n").sum().over("measure", "funder")).alias("share")
    )
    .sort("bin_index")
    .collect()
)


def draw_histograms(measure, left, first_bar, outlines):
    width = 84.587013 / 40
    baseline = 86.8
    for funder, outline in zip(("NSFC", "NSF", "NIH"), outlines):
        heights = shares.filter(
            (pl.col("measure") == measure) & (pl.col("funder") == funder)
        )["share"]
        steps = [f"M {left:.9f},{baseline}"]
        for index, height in enumerate(heights):
            x = left + index * width
            right = x + width
            y = baseline - 2.94 * height
            steps.append(f"V {y:.9f} H {right:.9f}")
            if funder == "NSFC":
                elements[f"patch_{first_bar + index}"][0].set(
                    "d", f"M {x:.9f},{baseline} H {right:.9f} V {y:.9f} H {x:.9f} Z"
                )
        steps.append(f"V {baseline} H {left:.9f}")
        elements[f"patch_{outline}"][0].set("d", " ".join(steps))


draw_histograms("trajectory", 28.5, 3, (45, 46, 47))
draw_histograms("grant", 139.30899, 50, (92, 93, 94))
tree.write(OUTPUT / "figure1.svg", encoding="UTF-8", xml_declaration=True)
