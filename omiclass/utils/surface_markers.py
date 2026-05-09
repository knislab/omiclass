# utils/surface_markers.py
from __future__ import annotations
from pathlib import Path
from typing import Union, Optional, Sequence
import subprocess
import shutil
import pandas as pd
import requests
import sys
from typing import List
from omiclass.utils.io import load_list

# Update the URL list with the reliable PNAS direct link
SURFACEOME_URLS: Sequence[str] = (
    "https://www.pnas.org/doi/suppl/10.1073/pnas.1808790115/suppl/Dataset_S01.XLS",
    "https://wlab.ethz.ch/surfaceome/table_S3_surfaceome.xlsx",
    "https://www.pnas.org/lookup/suppl/doi/10.1073/pnas.1808790115/-/DCSupplemental",
)
SURFACE_MARKER_PATH = Path(__file__).resolve().parents[1] / "data" / "filtered_surface_list.txt"

def _have(cmd: str) -> bool:
    """Check if an external command is available."""
    return shutil.which(cmd) is not None

def _run(cmd: list[str]) -> bool:
    """Run an external command and suppress output."""
    try:
        subprocess.run(
            cmd, 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            timeout=120
        )
        return True
    except Exception:
        return False

def _download_with_requests(url: str, dest: Path) -> bool:
    """Download file using the Python-native 'requests' library."""
    try:
        with requests.get(url, stream=True, allow_redirects=True, timeout=300) as r:
            r.raise_for_status() 
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except requests.exceptions.RequestException:
        return False

def _download_with_wget_or_curl(url: str, dest: Path) -> bool:
    """Download file using external wget or curl commands."""
    if _have("wget"):
        return _run(["wget", "-O", str(dest), url])
    if _have("curl"):
        return _run(["curl", "-L", "-o", str(dest), url])
    return False

def load_surfaceome(surface_marker_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads the surfaceome data from a local path or downloads it if it doesn't exist.
    """
    path = Path(surface_marker_path)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = False
        
        # Prioritize the requests library download 
        if 'requests' in sys.modules:
            for url in SURFACEOME_URLS:
                if _download_with_requests(url, path):
                    ok = True
                    break
        
        # Fallback to external tools if requests failed or wasn't available
        if not ok:
            for url in SURFACEOME_URLS:
                if _download_with_wget_or_curl(url, path):
                    ok = True
                    break
                    
        # Final check for download failure
        if not ok or not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Could not download surfaceome table to {path}.\n"
                "Tried URLs:\n  " + "\n  ".join(SURFACEOME_URLS)
            )
        
    return pd.read_excel(path)

def fillter_surface_markers(df: Union[pd.DataFrame, List[str]]) -> Union[pd.DataFrame, List[str]]:
    sur_gene_list = load_list(SURFACE_MARKER_PATH)
    sur_gene_set = set(sur_gene_list)
    if isinstance(df, pd.DataFrame):
        mask = df.index.isin(sur_gene_set)
        return df[mask]
    elif isinstance(df, list):
        filtered_list = [gene for gene in df if gene in sur_gene_set]
        
        return filtered_list
    else:
        raise ValueError("Input must be a pandas DataFrame or a list of gene names.")
    

def apply_surface_marker_filter_with_min(
    obj: Union[pd.DataFrame, List[str]],
    *,
    min_features: int,
    pad_strategy: str = "variance",
    allow_noop_if_missing: bool = True,
) -> Union[pd.DataFrame, List[str]]:

    if min_features <= 0:
        return fillter_surface_markers(obj)

    filtered = fillter_surface_markers(obj)

    if isinstance(obj, list):
        filtered_list = list(dict.fromkeys(map(str, filtered)))
        if len(filtered_list) >= min_features:
            return filtered_list

        original = list(dict.fromkeys(map(str, obj)))
        have = set(filtered_list)

        for g in original:
            if g not in have:
                filtered_list.append(g)
                have.add(g)
            if len(filtered_list) >= min_features:
                break

        if len(filtered_list) < min_features and not allow_noop_if_missing:
            raise ValueError(
                f"Surface marker filtering produced {len(filtered_list)} features; "
                f"min_features={min_features}."
            )
        return filtered_list

    if not isinstance(obj, pd.DataFrame):
        return filtered

    Xf = filtered
    if Xf.shape[1] >= min_features:
        return Xf

    X = obj
    have = list(Xf.columns)
    have_set = set(have)

    need = min_features - len(have)
    if need <= 0:
        return Xf

    remaining = [c for c in X.columns if c not in have_set]
    if not remaining:
        if allow_noop_if_missing:
            return Xf
        raise ValueError(
            f"Surface marker filtering produced {Xf.shape[1]} features; "
            f"min_features={min_features}."
        )

    if pad_strategy not in ("variance", "order"):
        raise ValueError(f"pad_strategy must be 'variance' or 'order'")

    if pad_strategy == "order":
        pad_cols = remaining[:need]
    else:
        vars_ = X[remaining].var(axis=0, ddof=0)
        pad_cols = vars_.sort_values(ascending=False).head(need).index.tolist()

    out_cols = have + pad_cols
    return X.loc[:, out_cols]
