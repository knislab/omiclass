import numpy as np
import pandas as pd
from typing import Tuple
from omiclass.errors.errors import DataShapeError

def basic_checks(X: pd.DataFrame, y: pd.Series):
    print(f"[data] X shape: {X.shape}, y length: {len(y)}")
    print(f"[data] any NaNs in X: {bool(np.isnan(X.values).any())}")

    counts = y.value_counts().sort_index()
    print("[data] class counts:")
    for k, v in counts.items():
        print(f"  class {k}: {v}")
    if counts.min() < 2:
        print("[warn] some class has <2 samples — stratified CV may disable inner folds.")


def check_and_prepare_xy(X: pd.DataFrame, y: pd.Series, *, na_y: str = "error", coerce_object: bool = True, drop_all_nan_rows: bool = False, drop_constant_cols: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    # align X and y on index
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    dropped_from_X = len(X.index) - len(common_index)
    dropped_from_y = len(y.index) - len(common_index)
    if dropped_from_X or dropped_from_y:
        print(f"[info] aligned X/y on index; dropped {dropped_from_X} rows from X and {dropped_from_y} from y.")
    # drop rows where all features are NaN
    if drop_all_nan_rows:
        keep_rows = ~X.isna().all(axis=1)
        if not keep_rows.all():
            X = X.loc[keep_rows]
            y = y.loc[keep_rows]
    # Coerce object columns in X to numeric (NaN on failure)
    if coerce_object:
        obj_cols = [c for c, dt in X.dtypes.items() if dt == "object"]
        if obj_cols:
            X[obj_cols] = X[obj_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    # drop constant columns (helps tree learners / reduces spam)
    if drop_constant_cols:
        nunique = X.nunique(dropna=False)
        keep_cols = nunique[nunique > 1].index
        if len(keep_cols) != X.shape[1]:
            X = X[keep_cols]

    # Handle y missing values
    if y.isna().any():
        if na_y == "drop":
            keep = ~y.isna()
            X, y = X.loc[keep], y.loc[keep]
        elif na_y == "sentinel":
            y = y.astype("object").fillna("__MISSING__")
        elif na_y == "error":
            raise DataShapeError("y contains missing values (NaN); set na_y='drop' or 'sentinel' to handle them.")
        else:
            raise DataShapeError(f"Invalid na_y option: {na_y}")

    # Encode y to dense 0..K-1 integers; ensure plain int64
    if not pd.api.types.is_integer_dtype(y):
        y = pd.Series(pd.Categorical(y).codes, index=y.index)
    y = y.astype(np.int64, copy=False)

    # Final sanity checks
    if len(X) != len(y):
        raise RuntimeError("Post-processing mismatch: X and y length differ.")
    if not X.index.equals(y.index):
        raise RuntimeError("Post-processing mismatch: X and y indices differ.")

    return X, y

def ensure_dense_finite(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        arr = x.values
    else:
        arr = np.asarray(x)

    # If it's a masked array, fill masked values
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)

    # Replace NaN/±inf with 0.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert back to original type
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(arr, index=x.index, columns=x.columns)
    if isinstance(x, pd.Series):
        return pd.Series(arr, index=x.index, name=x.name)
    return arr