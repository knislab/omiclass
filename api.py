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

def predict(expression: Path, model_dir: Path, output_dir: Path, **kwargs: Any):
    """Load a foundation run and predict into a new output directory, without fitting."""
    settings = dict(kwargs.pop("foundation", {}) or {})
    if settings.get("mode", "predict") != "predict" or any(
        kwargs.get(key, "predict") != "predict" for key in ("foundation.mode", "foundation_mode")
    ):
        raise ValueError("api.predict cannot switch to training")
    for key in ("foundation.enabled", "foundation_enabled"):
        if key in kwargs and str(kwargs[key]).strip().lower() not in {"true", "1"}:
            raise ValueError("api.predict cannot disable the saved foundation route")
    settings.update(enabled=True, mode="predict", run_dir=model_dir)
    return run(expression, foundation=settings, output_dir=output_dir, **kwargs)


def foundation_configurations():
    """Enumerate supported manuscript protocols without loading any encoder."""
    from omiclass.foundation.catalogue import configurations
    return configurations()


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