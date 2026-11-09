# omiclass/utils/pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

from omiclass.utils.splits import holdout_split, save_split, load_split, apply_split
from omiclass.utils.io import load_labels
from omiclass.errors.errors import DataShapeError
from omiclass.config.schema import Config


def expression_looks_log_transformed(X: pd.DataFrame) -> bool:
    """
    Conservative scale check used to avoid applying log1p twice.

    Limma batch correction expects log-scale expression. Raw counts and TPM-like
    values usually have a large upper tail; already log-transformed matrices
    usually do not. Negative values also imply a transformed scale.
    """
    vals = pd.to_numeric(X.stack(dropna=True), errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return False
    if (vals < 0).any():
        return True
    integer_like = np.isclose(vals.to_numpy(dtype=float), np.round(vals.to_numpy(dtype=float))).mean()
    if integer_like >= 0.95:
        return False
    q99 = float(vals.quantile(0.99))
    vmax = float(vals.max())
    return q99 <= 25.0 and vmax <= 50.0


def maybe_log1p_expression(
    X: pd.DataFrame,
    *,
    enabled: bool,
    already_normalized_or_logged: bool,
) -> tuple[pd.DataFrame, bool, str]:
    if not enabled:
        return X, False, "disabled"
    if already_normalized_or_logged:
        return X, False, "already transformed"
    if expression_looks_log_transformed(X):
        return X, False, "input appears log-scale"
    return np.log1p(X), True, "applied"



def make_or_load_split(
    X: pd.DataFrame,
    y: pd.Series,
    split_dir: Path,
    cfg: Config,
    reuse_split: bool,
    split_json: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Path, bool]:
    """
    Shared helper for creating or reusing a train/test split.
    Returns Xtr, Xte, ytr, yte, split_path, created_flag.
    """
    split_path = split_dir / "split.json"
    source = Path(split_json) if split_json is not None else split_path
    if split_json is not None or reuse_split:
        if not source.is_file():
            raise FileNotFoundError(f"Requested split does not exist: {source}")
        split = load_split(source)
        Xtr, Xte, ytr, yte = apply_split(X, y, split)
        if source.resolve() != split_path.resolve():
            save_split(split, split_path)
        created = False
    else:
        cfg_runtime = cfg.runtime

        raw_hold_out = getattr(cfg.project, "hold_out_path", None)
        hold_out_path: Optional[Path] = None
        if raw_hold_out:
            hold_out_path = Path(raw_hold_out)

        Xtr, Xte, ytr, yte, split = holdout_split(
            X,
            y,
            method=str(getattr(cfg_runtime.split, "method", "stratified")),
            train_ratio=float(getattr(cfg_runtime.split, "train_ratio", 0.8)),
            random_state=int(cfg_runtime.random_state),
            hold_out_path=hold_out_path,
        )
        save_split(split, split_path)
        created = True

    return Xtr, Xte, ytr, yte, split_path, created


def apply_batch_holdout(
    X: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Move batch hold-out samples to a separate CSV and return X with those columns removed.
    Semantics follow cfg.batch_correction.hold_out_path + your existing DataProcessor logic.
    """
    meta_path = cfg.project.meta_path
    cluster_col = cfg.project.cluster_column
    y, common_col = load_labels(meta_path, cluster_col, index_like=X.columns)
    y.dropna(inplace=True)
    y_patient = set(y.index)
    y_patient = {pat for pat in y_patient if pd.notna(pat)}
    missing_patient = [pat for pat in X.columns if pat not in y_patient]

    # Keep unlabeled samples available for external prediction; never index them
    # after they have already been removed from the matrix.
    if not missing_patient:
        return X.loc[:, common_col]
    
    # if missing hold_out_ids in X.columns, ignore
    hold_out_ids = [hid for hid in missing_patient if hid in X.columns]
    if not hold_out_ids:
        return X

    X_hold = X[hold_out_ids]
    X_filt = X.drop(columns=hold_out_ids)

    hold_out_dir = out_dir / "train_holdout"
    hold_out_dir.mkdir(parents=True, exist_ok=True)
    hold_out_file = hold_out_dir / "holdout_expression.csv"
    X_hold.to_csv(hold_out_file)

    return X_filt
