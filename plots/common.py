from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)


def frame(path):
    table = pl.scan_csv(path, infer_schema_length=None).collect()
    return table.to_pandas()
