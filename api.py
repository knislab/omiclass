from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from omiclass.pipeline.runner import run_pipeline


def run(
    expression: Path,
    **kwargs: Any
) -> Dict[str, Any]:
    return run_pipeline(
        expression_file=expression,
        **kwargs
    )

def run_from_cli():
    parser = argparse.ArgumentParser(description="Run the pipeline via command line.")
    
    parser.add_argument("--override", nargs=2, action="append", default=[])
    args = parser.parse_args()
    expression_file = None
    active_kwargs = {}
    for k, v in args.override:
        if k == "expression_file":
            expression_file = Path(v)
        else:
            if "," in v:
                active_kwargs[k] = [x.strip() for x in v.split(",") if x.strip()]
            else:
                active_kwargs[k] = v    
    if expression_file is None:
        raise ValueError("expression_file must be provided via --override expression_file <path>")
    return run(expression=expression_file, **active_kwargs)

if __name__ == "__main__":
    run_from_cli()