from __future__ import annotations
from typing import Tuple, Dict, Any, Union, Optional
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from omiclass.utils.io import load_list
 

Split = Dict[str, Any]

def holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    method: str = "stratified", 
    train_ratio: float = 0.8,
    random_state: int = 42,
    hold_out_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Split]:
    """Single hold-out split; returns X_train, X_test, y_train, y_test + split meta."""

    # If a hold-out file is provided and exists, use it to define the test set
    if hold_out_path is not None:
        hold_out_path = Path(hold_out_path)
        if not hold_out_path.exists():
            print(f"[holdout] hold_out_path does not exist, falling back to regular split: {hold_out_path}")
        else:
            test_ids = load_list(hold_out_path)
            all_indices = X.index
            Xte_idx = all_indices.intersection(test_ids)
            Xtr_idx = all_indices.difference(Xte_idx)

            if Xte_idx.empty:
                print("Warning: No matching IDs found for the hold-out path.")
            elif Xtr_idx.empty:
                raise ValueError(
                    "All data points were assigned to the test set based on hold_out_path. "
                    "Training set is empty."
                )
            else:
                Xtr = X.loc[Xtr_idx]
                Xte = X.loc[Xte_idx]
                ytr = y.loc[Xtr_idx]
                yte = y.loc[Xte_idx]

                test_size = len(Xte_idx) / len(all_indices)
                train_ratio = 1.0 - test_size
                method = "pre-defined"
                random_state = None  # Random state is irrelevant for pre-defined split

                split = {
                    "random_state": random_state,
                    "method": method,
                    "train_ratio": float(train_ratio),
                    "test_size": float(test_size),
                    "train_index": Xtr.index.tolist(),
                    "test_index": Xte.index.tolist(),
                }
                return Xtr, Xte, ytr, yte, split

    # Fallback: standard stratified / random split
    test_size = 1.0 - float(train_ratio)

    # align y
    y = y.reindex(X.index)
    stratify = y if method == "stratified" else None

    Xtr_idx, Xte_idx, ytr, yte = train_test_split(
        X.index, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    Xtr = X.loc[Xtr_idx]
    Xte = X.loc[Xte_idx]
    split = {
        "random_state": int(random_state),
        "method": str(method),
        "train_ratio": float(train_ratio),
        "test_size": float(test_size),
        "train_index": Xtr.index.tolist(),
        "test_index": Xte.index.tolist(),
    }
    return Xtr, Xte, ytr.loc[Xtr.index], yte.loc[Xte.index], split

def save_split(split: Split, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(split, f)

def load_split(path: Union[str, Path]) -> Split:
    with Path(path).open("r") as f:
        return json.load(f)

def apply_split(
    X: pd.DataFrame, y: pd.Series, split: Split
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    tr_idx = split["train_index"]
    te_idx = split["test_index"]
    Xtr = X.loc[tr_idx]
    Xte = X.loc[te_idx]
    ytr = y.loc[tr_idx]
    yte = y.loc[te_idx]
    return Xtr, Xte, ytr, yte
