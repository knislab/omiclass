from __future__ import annotations

import pandas as pd


def canonicalize_numeric_matrix(df: pd.DataFrame, *, decimals: int = 12) -> pd.DataFrame:
    """Round numeric matrix values while preserving labels and order."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        out.loc[:, numeric_cols] = out.loc[:, numeric_cols].round(int(decimals))
    return out

