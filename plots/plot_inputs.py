import polars as pl


def scan_csv(path):
    return pl.scan_csv(path, null_values=["NA", ""])


class PlotRow(dict):
    def __getattr__(self, name):
        return self[name]


def one_row(lazy):
    rows = lazy.limit(2).collect().to_dicts()
    assert len(rows) == 1
    return PlotRow(rows[0])
