import pandas as pd
from pathlib import Path
from typing import Optional, Union, Sequence

def _assert_unique(df: pd.DataFrame, name: str):
    if not df.index.is_unique:
        raise ValueError(f"{name}: duplicate row index")
    if not df.columns.is_unique:
        raise ValueError(f"{name}: duplicate column names")


def _align_samples(expr_df: pd.DataFrame, meta_df: Optional[pd.DataFrame], required_meta_cols: Optional[Sequence[str]] = None) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    required_meta_cols = [c for c in required_meta_cols if c is not None] if required_meta_cols is not None else None
    if meta_df is None:
        return expr_df, None
    # drops nans
    expr_df = expr_df.dropna(axis=1, how='all')
    if required_meta_cols is not None and len(required_meta_cols) > 0:
        missing = [c for c in required_meta_cols if c not in meta_df.columns]
        if missing:
            raise ValueError(f"required_meta_cols missing in meta_df: {missing}")
        meta_df = meta_df[required_meta_cols]
        meta_df = meta_df.dropna(axis=0, how="any")

    # Check that metadata rows are samples and expression columns are samples
    common = expr_df.columns.intersection(meta_df.index)
    if len(common) == 0:
        raise ValueError("No overlapping samples between expr_df.columns and meta_df.index")

    # Filter and maintain the expression column order
    return expr_df.loc[:, common], meta_df.loc[common]

def _write_csv(obj: Union[pd.DataFrame, pd.Series], path: Path):
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    obj.to_csv(path)