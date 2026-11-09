"""Fitted state for existing preprocessing and limma marker selection."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from omiclass.utils.pipeline import maybe_log1p_expression


class TrainingPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, cfg, *, raw_expression=None, metadata=None, de=False,
                 covariates=None, already_logged=False, cptac=False, reference=False,
                 work_dir=None):
        self.cfg = cfg
        self.raw_expression = raw_expression
        self.metadata = metadata
        self.de = de
        self.covariates = covariates
        self.already_logged = already_logged
        self.cptac = cptac
        self.reference = reference
        self.work_dir = work_dir

    def __getstate__(self):
        state = self.__dict__.copy()
        # Raw training values/covariates are fit-time inputs, never deployment state.
        state['raw_expression'] = None
        state['metadata'] = None
        return state

    def _cptac_fit(self, X):
        cfg = self.cfg.cptac_ms_preprocessing
        self.cptac_columns_ = list(X.columns if cfg.max_missing_fraction is None else
                                   X.columns[X.isna().mean() <= cfg.max_missing_fraction])
        values = X.loc[:, self.cptac_columns_]
        finite = values.to_numpy()[np.isfinite(values.to_numpy())]
        minimum = float(finite.min()) if finite.size else 1.0
        self.offset_ = abs(minimum) + cfg.pseudocount if minimum <= 0 else 0.0
        if cfg.log2_transform:
            values = np.log2(values + self.offset_)
        self.medians_ = values.median()
        return self._cptac_transform(X)

    def _cptac_transform(self, X):
        cfg = self.cfg.cptac_ms_preprocessing
        values = X.loc[:, self.cptac_columns_].copy()
        if cfg.log2_transform:
            if ((values + self.offset_) <= 0).any().any():
                raise ValueError('Prediction values are incompatible with the fitted CPTAC log2 offset')
            with np.errstate(invalid='ignore', divide='ignore'):
                values = np.log2(values + self.offset_)
        if cfg.imputation == 'median_by_feature':
            values = values.fillna(self.medians_)
        return values

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame) or not X.index.is_unique:
            raise ValueError('Preprocessing requires a DataFrame with unique sample IDs')
        self.fit_ids_ = X.index.tolist()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        current = X.copy()
        self.reference_ = None
        if self.reference:
            from omiclass.batch.reference_pipeline import make_reference
            self.reference_ = make_reference(self.cfg)
            columns = list(dict.fromkeys([self.cfg.project.cluster_column] + (self.cfg.batch_correction.design_columns or [])))
            meta = self.metadata.loc[X.index].copy() if self.metadata is not None else pd.DataFrame(index=X.index)
            names = self.cfg.project.classes_names
            if meta.attrs.get('outcome_numeric'):
                values = meta.attrs.get('outcome_values', {})
                meta[self.cfg.project.cluster_column] = [values.get(int(v), v) for v in y]
            else:
                meta[self.cfg.project.cluster_column] = [names[int(v)] for v in y] if names else np.asarray(y)
            self.reference_.fit(current.T, meta.loc[:, columns] if self.cfg.batch_correction.algorithm != 'frozen_quantile' else None)
            current = self.reference_.transform(current.T).T
        if self.cptac:
            current = self._cptac_fit(current)
        current_t, self.did_log_, self.log_reason_ = maybe_log1p_expression(
            current.T, enabled=self.cfg.runtime.log_transform,
            already_normalized_or_logged=self.already_logged or self.reference or
                (self.cptac and self.cfg.cptac_ms_preprocessing.log2_transform))
        current = current_t.T
        self.selected_features_ = current.columns[current.nunique(dropna=False).gt(1)].tolist()
        self.de_fit_ids_ = []
        if self.de:
            if self.raw_expression is None:
                raise ValueError('DE requires fit-time raw expression')
            from omiclass.de.limma_voom_runner import LimmaVoomRunner
            from omiclass.utils.io import encode_meta_matrix
            # Construct outcome values solely from this fit call, never metadata outcomes.
            meta = self.metadata.loc[X.index].copy() if self.metadata is not None else pd.DataFrame(index=X.index)
            meta = encode_meta_matrix(meta)
            meta['cluster'] = np.asarray(y)
            raw = self.raw_expression.loc[:, X.index]
            if self.cfg.project.modality == 'cptac_ms':
                from omiclass.data.processor import DataProcessor
                raw, _ = DataProcessor(self.cfg).preprocess_cptac_ms(raw)
            root = Path(self.work_dir) if self.work_dir else None
            if root is not None:
                root.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix='de-fit-', dir=root) as directory:
                runner = LimmaVoomRunner(self.cfg.de, self.cfg)
                common = dict(meta_df=meta, output_dir=directory, design_col='cluster',
                    covariates=self.covariates, surface_markers=self.cfg.surface_markers.enabled,
                    min_count=self.cfg.de.min_count if self.cfg.de.min_count is not None else 10,
                    min_samples_frac=self.cfg.de.min_samples_frac,
                    report=False)
                if self.cfg.project.modality == 'cptac_ms':
                    result = runner.run_limma(expr_df=raw, **common)
                else:
                    result = runner.run_limma_voom(raw_counts_df=raw, **common)
            selected = result.get('selected_features', result.get('selected_genes', []))
            self.selected_features_ = [g for g in selected if g in self.selected_features_]
            self.de_fit_ids_ = X.index.tolist()
        if not self.selected_features_:
            raise ValueError('No usable features remain in this training partition')
        return self

    def transform(self, X):
        # Full reference/CPTAC transformations require their fitted input schemas.
        required = (list(self.feature_names_in_) if self.reference else
                    self.cptac_columns_ if self.cptac else self.selected_features_)
        missing = set(required) - set(X.columns)
        if missing:
            raise ValueError(f'Prediction lacks fitted preprocessing features: {sorted(missing)[:5]}')
        current = X.loc[:, required].copy()
        if self.reference_ is not None:
            current = self.reference_.transform(current.T).T
        if self.cptac:
            current = self._cptac_transform(current)
        if self.did_log_:
            if (current <= -1).any().any():
                raise ValueError('Prediction values are incompatible with the fitted log1p transform')
            current = np.log1p(current)
        current = current.loc[:, self.selected_features_]
        # Retain the existing core finite-value cleaning after fitted transforms.
        return current.replace([np.inf, -np.inf], np.nan).fillna(0.0)
