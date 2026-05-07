from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from generic_classifier.config.schema import Config
from generic_classifier.utils.io import load_matrix, load_labels, load_list, load_meta_matrix
from generic_classifier.utils.checks import check_and_prepare_xy, ensure_dense_finite
from generic_classifier.utils.surface_markers import fillter_surface_markers
from generic_classifier.errors.errors import DataError, DataShapeError

def get_file_version(version: int = 19) -> str:
        if version == 38:
            return Path(__file__).resolve().parents[1] / "data" / "gencode_v34_GRCh38_hg38.csv"
        else:
            return Path(__file__).resolve().parents[1] / "data" / "gencode_GRCh37_hg19.csv"
        
def get_gene_symbols(genes_id: List[str], cfg: Config) -> List[str]:
    version = cfg.project.version
    path = get_file_version(version=version)
    p = Path(path)

    if not p.exists():
        raise DataError(f"Gene length file not found: {path}")

    if p.suffix == ".csv":
        df = pd.read_csv(p, index_col=0)
    elif p.suffix == ".txt":
        df = pd.read_csv(p, index_col=0, sep="\t")
    elif p.suffix == ".parquet":
        df = pd.read_parquet(p)
        df = df.set_index(df.columns[0])
    else:
        raise DataError(f"Unsupported file format: {p.suffix}")

    df.index = df.index.astype(str)

    gene_name_map = df["gene_name"].to_dict()

    symbols = []
    for gid in genes_id:
        gid_str = str(gid)
        symbol = gene_name_map.get(gid_str, gid_str)  # fallback = ID
        symbols.append(symbol)

    return symbols