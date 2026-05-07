# generic_classifier/cli/runtime.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, List
from generic_classifier.config.schema import Config
from generic_classifier.utils.clean_and_pred import has_leaderboard



Mode = Literal["full", "train-only", "de-train", "batch-train", "clean_and_predict"]
InputMode = Literal["raw_counts", "batch_correction", "ready_for_training"]

@dataclass
class RuntimeOptions:
    mode: Mode = "full"
    input_mode: InputMode = "ready_for_training"

    # core data
    expression_path: Optional[Path] = None    # used for batch_corrected / ready_for_training
    counts_path: Optional[Path] = None        # used for raw_counts
    meta_path: Optional[Path] = None                 # labels
    gene_path: Optional[Path] = None          # optional gene list for DataProcessor
    surface_markers: bool = False

    # modules toggles
    run_batch: Optional[bool] = None
    run_de: Optional[bool] = None
    run_geneset_opt: Optional[bool] = None
    run_training: Optional[bool] = None
    run_prediction: Optional[bool] = None

    # split
    reuse_split: bool = False
    split_json: Optional[Path] = None


def create_runtime_options_from_path_conf_kwargs(
    cfg: Config,
    expression_path: Path,
    kwargs: dict,
) -> RuntimeOptions:
    """
    Helper to create RuntimeOptions from CLI-like arguments.
    """
    is_geneset = (kwargs.get("geneset_path", False) or cfg.project.geneset_path is not None)
    run_de = (kwargs.get("run_de", False) or not is_geneset)
    run_de = (kwargs.get("raw_count", False) or run_de)
    is_raw_count = kwargs.get("raw_count", False)
    run_batch = kwargs.get("run_batch", False) or (cfg.batch_correction.batch_column is not None) or (cfg.batch_correction.covariates_path is not None) or (cfg.batch_correction.n_pcs is not None)
    geneset_opt = kwargs.get("run_geneset_opt", False) or cfg.runtime.feature_selection.enabled is True
    run_prediction = kwargs.get("run_prediction", False) or (cfg.runtime.survival_prediction.enabled is True)
    is_split = kwargs.get("reuse_split", False) and kwargs.get("split_json", False)
    project_out_dir = Path(cfg.project.out_dir)

    if has_leaderboard(project_out_dir):
        mode= "clean_and_predict"
        if is_raw_count:
            input_mode= "raw_counts"
        else:
            input_mode = "batch_correction"
    elif run_de and run_batch:
        mode = "full"
        input_mode = "raw_counts"
    elif run_de and not run_batch:
        mode = "de-train"
        input_mode = "raw_counts"
    elif not run_de and run_batch:
        mode = "batch-train"
        input_mode = "batch_correction"
    else:
        mode = "train-only"
        input_mode = "ready_for_training"
    if input_mode == "raw_counts":
        expr_path = None
        counts_path = expression_path
    else:
        expr_path = expression_path
        counts_path = None

    meta_path = kwargs.get("meta_path", None)
    if meta_path is None:
        meta_path = cfg.project.meta_path

    gene_path: Optional[Path] = None
    if is_geneset:
        gene_path = kwargs.get("gene_path", None)
        if gene_path is None:
            gene_path = cfg.project.geneset_path

    out = {
        "mode" : mode,
        "input_mode" : input_mode,
        "expression_path" : expr_path,
        "counts_path" : counts_path,
        "meta_path" : meta_path,
        "gene_path" : gene_path,
        "surface_markers" : kwargs.get("surface_markers", False) or cfg.surface_markers.enabled,
        "run_batch": run_batch,
        "run_de": run_de,
        "run_geneset_opt": geneset_opt,
        "run_training": True,
        "run_prediction": run_prediction,
        "reuse_split": is_split,
        "split_json": kwargs.get("split_json", None),
    }
    return RuntimeOptions(**out)
