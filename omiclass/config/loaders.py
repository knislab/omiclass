# generic-classifier/config/loaders.py
from __future__ import annotations
import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import re

import yaml
from .schema import Config, ProjectCfg
from omiclass.errors.errors import ConfigError, SchemaValidationError
from omiclass.config.overrides import normalize_user_overrides, RUNTIME_OPTIONS
from pydantic import ValidationError
from omiclass.utils.clean_and_pred import has_leaderboard

def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def load_config(cli_overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Create config from schema defaults + env overrides + optional CLI overrides.
    """
    try:
        cli_overrides = dict(cli_overrides or {})
        cli_overrides = normalize_user_overrides(cli_overrides)

        base_data = {}
        project_cfg = cli_overrides.get("project", {}) or {}
        config_path = project_cfg.get("out_dir")
        config_path = cli_overrides.get("output_dir", config_path)
        config_path = cli_overrides.get("out_dir", config_path)
        if config_path:
            config_full_path = Path(config_path) / "config.freeze.yaml"
            if config_full_path.is_file():
                with open(config_full_path, "r", encoding="utf-8") as f:
                    base_data = yaml.safe_load(f) or {}

        foundation = cli_overrides.get("foundation", {})
        if foundation.get("enabled") and foundation.get("mode") == "predict":
            run_dir = foundation.get("run_dir")
            if run_dir is None:
                raise ValueError("Prediction requires foundation.run_dir")
            saved_path = Path(run_dir) / "config.freeze.yaml"
            with saved_path.open(encoding="utf-8") as stream:
                base_data = yaml.safe_load(stream) or {}
            if not base_data.get("foundation", {}).get("enabled"):
                raise ValueError("Saved run is not a foundation API run")
            # A TPM file describes a cohort, not the fitted encoder. Prediction
            # must use this invocation's file or normalize its supplied counts.
            base_data.get("bulkrnabert", {}).pop("tpm_input_path", None)

        merged = _deep_update(base_data, cli_overrides) if base_data else cli_overrides

        for key in RUNTIME_OPTIONS:
            merged.pop(key, None)
        raw = Config(**merged)
        prediction_only = (raw.foundation.enabled and raw.foundation.mode == "predict") or (
            not raw.foundation.enabled and has_leaderboard(raw.project.out_dir)
        )
        if not prediction_only:
            if raw.runtime.tuning.method != "bayes":
                raise ConfigError("runtime.tuning.method supports only 'bayes'; random/grid are not implemented")
            if raw.runtime.bayesian_iterations < 0:
                raise ConfigError("bayesian_iterations must be nonnegative")
            if raw.runtime.tuning.inner_cv is not None and raw.runtime.tuning.inner_cv < 2:
                raise ConfigError("runtime.tuning.inner_cv must be at least two")
            from sklearn.metrics import get_scorer
            try:
                get_scorer(raw.runtime.scoring)
            except ValueError as exc:
                raise ConfigError(str(exc)) from exc
        return Config.model_validate(raw)

    except ValidationError as e:
        raise SchemaValidationError(f"Invalid config: {e}") from e

def freeze_config(cfg: Config, out_dir: Union[str, Path]) -> Path:
    """
    Write a machine-readable, fully-resolved copy of the effective configuration.
    Useful for provenance next to artifacts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "config.freeze.yaml"
    text = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    out_path.write_text(text)
    return out_path
