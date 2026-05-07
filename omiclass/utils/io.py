import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
from generic_classifier.errors.errors import DataShapeError
from io import StringIO
import matplotlib.pyplot as plt
from generic_classifier.config.schema import Config


def load_matrix(path: Union[str, Path], numeric: bool = True) -> pd.DataFrame:
    p = str(path)
    if p.endswith(".parquet"):
        df = pd.read_parquet(p)
    elif p.endswith(".csv") or p.endswith(".tsv"):
        sep = "," if p.endswith(".csv") else "\t"
        df = pd.read_csv(p, sep=sep, index_col=0, low_memory=False)
        if numeric:
            df = df.apply(pd.to_numeric, errors="coerce")
    else:
        raise ValueError(f"Unsupported matrix file: {p}")
    df = df[~df.index.duplicated(keep="first")]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df

def load_meta_matrix(path: Union[str, Path], cluster_col: Optional[str] = None) -> pd.DataFrame:
    p = str(path)
    cluster_categories = None

    # ---- load ----
    if p.endswith(".parquet"):
        df = pd.read_parquet(p)
    elif p.endswith(".csv") or p.endswith(".tsv"):
        sep = "," if p.endswith(".csv") else "\t"
        df = pd.read_csv(p, sep=sep, index_col=0, low_memory=False)
    else:
        raise ValueError(f"Unsupported matrix file: {p}")

    # ---- de-duplicate (same as load_matrix) ----
    df = df[~df.index.duplicated(keep="first")]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        s = df[col]

        # Try numeric conversion
        s_num = pd.to_numeric(s, errors="coerce")

        non_na = s.notna().sum()
        numeric_ok = s_num.notna().sum()

        # If column is mostly numeric -> keep numeric
        if non_na == 0 or (numeric_ok / non_na) >= 0.9:
            out[col] = s_num
            continue

        # Otherwise: categorical -> label encode
        s_cat = s.astype("string")

        # deterministic ordering
        categories = sorted(s_cat.dropna().unique().tolist())
        cat_map = {c: i for i, c in enumerate(categories)}
        if cluster_col is not None and col == cluster_col:
            cluster_categories = categories

        out[col] = s_cat.map(cat_map).astype("Int64")

    if cluster_col is not None:
        return out, cluster_categories
    return out

def load_survival_matrix(path: Union[str, Path], index_like: pd.Index, time_col: str, event_col: str) -> pd.DataFrame:
    meta = load_meta_matrix(path)
    if time_col not in meta.columns:
        raise KeyError(f"Time column '{time_col}' not found in metadata columns: {meta.columns.tolist()}")
    if event_col not in meta.columns:
        raise KeyError(f"Event column '{event_col}' not found in metadata columns: {meta.columns.tolist()}")
    survival_df = meta.loc[index_like, [time_col, event_col]].copy()
    t = pd.to_numeric(survival_df[time_col], errors="coerce")
    e = pd.to_numeric(survival_df[event_col], errors="coerce")

    bad = t.isna() | e.isna()
    if bad.any():
        bad_rows = survival_df.loc[bad, [time_col, event_col]].head(20)
        raise ValueError(
            f"Non-numeric values found in survival columns.\n"
            f"Examples (up to 20):\n{bad_rows}"
        )

    survival_df[time_col] = t.astype(float)
    survival_df[event_col] = e.astype("Int64")
    return survival_df

def load_labels(path: Union[str, Path], label_col: str, index_like: Union[pd.Index, None] = None, cfg: Union[Config, None] = None) -> pd.Series:
    meta, cluster_categories = load_meta_matrix(path, label_col)
    meta_label = meta[label_col]
    meta_label = meta_label.dropna()

    if cfg is not None and cluster_categories is not None:
        project_cfg = getattr(cfg, "project", None)
        if project_cfg is not None:
            current = getattr(project_cfg, "classes_names", None)

            current_is_empty = (
                current is None
                or (hasattr(current, "__len__") and len(current) == 0)
            )
            new_is_valid = (
                cluster_categories is not None
                and (not hasattr(cluster_categories, "__len__") or len(cluster_categories) > 0)
            )

            if current_is_empty and new_is_valid:
                project_cfg.classes_names = [str(c).replace("_", " ") for c in cluster_categories]
    if index_like is None:
        return meta_label
    aligned_labels = meta_label.reindex(index_like)
    common_index = aligned_labels.index
    if len(aligned_labels) == 0:
        raise DataShapeError(
            f"After alignment, got 0 labels from '{path}'."
        )
    return aligned_labels, common_index


def load_list(path: Union[str, Path]) -> List[str]:
    p = str(path)
    # --- Load the file based on extension ---
    if p.endswith(".parquet"):
        df = pd.read_parquet(p)
        if df.shape[1] > 1:
            s = df.iloc[:, 0]  # use first column
        else:
            s = df.squeeze()
    elif p.endswith(".csv") or p.endswith(".tsv"):
        sep = "," if p.endswith(".csv") else "\t"
        df = pd.read_csv(p, sep=sep)
        if df.shape[1] > 1:
            s = df.iloc[:, 0]  # take first column
        else:
            s = df.squeeze()
    elif p.endswith(".txt"):
        with open(p, "r") as f:
            s = pd.Series([line.strip() for line in f if line.strip()])
    else:
        raise ValueError(f"Unsupported list file: {p}")

    # --- Clean and return ---
    s = s.dropna().astype(str).str.strip()
    s = s[~s.duplicated(keep="first")]
    return s.tolist()

def ensure_path(dirpath: Path) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath

def fig_to_svg_str(fig) -> str:
    buf = StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    svg = buf.getvalue()
    buf.close()
    return svg