"""Training-fitted expression references (genes x samples, natural log1p TPM).

FrozenQuantile implements the sorted-column mean target of Bolstad et al.
(2003), interpolating the target at each tied group's average rank.
ReferenceRUV is a custom frozen control-gene SVD / joint-design regression,
not the RUVSeq implementation. See docs/reference_preprocessing.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class ExpressionReference:
    version = 1
    method = "base"

    def __init__(self, input_scale="tpm"):
        if input_scale not in {"tpm", "log1p_tpm"}:
            raise ValueError("input_scale must be tpm or log1p_tpm (natural logarithm)")
        self.input_scale = input_scale

    def _matrix(self, X, fitting=False):
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("Expression must be a nonempty genes x samples DataFrame")
        if not X.index.is_unique or not X.columns.is_unique or X.index.hasnans or X.columns.hasnans:
            raise ValueError("Gene and sample identifiers must be unique and nonmissing")
        if not all(isinstance(g, str) for g in X.index) or not all(isinstance(s, str) for s in X.columns):
            raise ValueError("Gene and sample identifiers must be strings")
        if not fitting:
            if not hasattr(self, "genes_"):
                raise ValueError("Reference has not been fitted")
            missing = set(self.genes_) - set(X.index)
            extra = set(X.index) - set(self.genes_)
            if missing or extra:
                raise ValueError(f"Incompatible gene schema: missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]}")
            X = X.loc[self.genes_]
        values = X.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Expression contains missing/nonfinite values; no implicit imputation is allowed")
        if (values < 0).any():
            raise ValueError("TPM/log1p(TPM) input must be nonnegative")
        return np.log1p(values) if self.input_scale == "tpm" else values

    def _record(self, X):
        self.genes_ = X.index.tolist()
        self.training_samples_ = X.columns.tolist()

    def fit_transform(self, X, metadata=None):
        return self.fit(X, metadata).transform(X)

    def save(self, path):
        if not hasattr(self, "genes_"):
            raise ValueError("Reference has not been fitted")
        arrays = {k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}
        state = {k: v for k, v in vars(self).items() if k not in arrays}
        state.update(version=self.version, method=self.method)
        # Single, non-pickle artifact; do not retain a training expression matrix.
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, state=np.array(json.dumps(state)), **arrays)

    @staticmethod
    def load(path):
        with np.load(path, allow_pickle=False) as archive:
            state = json.loads(str(archive["state"]))
            if state.pop("version") != ExpressionReference.version:
                raise ValueError("Unsupported expression reference version")
            kind = state.pop("method")
            classes = {"frozen_quantile": FrozenQuantile, "reference_ruv": ReferenceRUV}
            if kind == "fsva_exact":
                from omiclass.batch.fsva import FrozenSVA
                classes[kind] = FrozenSVA
            if kind not in classes:
                raise ValueError(f"Unsupported reference method: {kind}")
            obj = classes[kind].__new__(classes[kind])
            obj.__dict__.update(state)
            obj.__dict__.update({k: archive[k].copy() for k in archive.files if k != "state"})
        return obj


class FrozenQuantile(ExpressionReference):
    method = "frozen_quantile"

    def fit(self, X, metadata=None):
        values = self._matrix(X, fitting=True)
        self.target_ = np.sort(values, axis=0).mean(axis=1)
        self._record(X)
        return self

    def transform(self, X):
        values = self._matrix(X)
        result = np.empty_like(values)
        for j in range(values.shape[1]):
            order = np.argsort(values[:, j], kind="stable")
            sorted_values = values[order, j]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
            ends = np.r_[starts[1:], len(order)]
            ranks = (starts + ends - 1) / 2
            mapped = np.interp(ranks, np.arange(len(self.target_)), self.target_)
            result[order, j] = np.repeat(mapped, ends - starts)
        return pd.DataFrame(result, index=self.genes_, columns=X.columns)


class ReferenceRUV(ExpressionReference):
    method = "reference_ruv"

    def __init__(self, control_genes, control_justification, n_factors=1,
                 design_columns=(), input_scale="tpm", max_condition=1e8):
        super().__init__(input_scale)
        if not control_justification or not control_justification.strip():
            raise ValueError("Negative controls require a scientific control_justification")
        if not control_genes or len(set(control_genes)) != len(control_genes):
            raise ValueError("Supply a nonempty, unique list of justified negative-control genes")
        if isinstance(n_factors, bool) or not isinstance(n_factors, int) or n_factors < 1:
            raise ValueError("n_factors must be a positive integer")
        if not design_columns:
            raise ValueError("A training biological design including the classification target is required")
        if not np.isfinite(max_condition) or max_condition <= 1:
            raise ValueError("max_condition must be finite and > 1")
        self.control_genes = list(control_genes)
        self.control_justification = control_justification
        self.n_factors = n_factors
        self.design_columns = list(design_columns)
        self.max_condition = max_condition

    def fit(self, X, metadata=None):
        Y = self._matrix(X, fitting=True).T
        absent = set(self.control_genes) - set(X.index)
        if absent:
            raise ValueError(f"Missing negative-control genes: {sorted(absent)[:10]}")
        if metadata is None or not metadata.index.is_unique:
            raise ValueError("Unique training metadata indexed by sample is required")
        # Only the fitted sample rows and explicitly declared design columns are accessed.
        meta = metadata.loc[X.columns, self.design_columns]
        if meta.isna().any().any():
            raise ValueError("Training biological design contains missing values")
        design = [np.ones((len(meta), 1))]
        self.design_encoding_ = {}
        for name in self.design_columns:
            col = meta[name]
            if pd.api.types.is_numeric_dtype(col):
                v = col.to_numpy(dtype=float)
                sd = v.std()
                if not np.isfinite(v).all() or sd == 0:
                    raise ValueError(f"Nonfinite or constant design column: {name}")
                design.append(((v - v.mean()) / sd)[:, None])
                self.design_encoding_[name] = dict(mean=float(v.mean()), scale=float(sd))
            else:
                levels = sorted(col.astype(str).unique())
                if len(levels) < 2:
                    raise ValueError(f"Constant design column: {name}")
                design.append(np.column_stack([(col.astype(str) == level).to_numpy(dtype=float) for level in levels[1:]]))
                self.design_encoding_[name] = dict(levels=levels)
        B = np.column_stack(design)
        positions = X.index.get_indexer(self.control_genes)
        C = Y[:, positions]
        center = C.mean(axis=0)
        _, singular, vt = np.linalg.svd(C - center, full_matrices=False)
        tolerance = np.finfo(float).eps * max(C.shape) * singular[0]
        if self.n_factors > np.count_nonzero(singular > tolerance):
            raise ValueError("Negative controls have insufficient numerical rank for n_factors")
        projection = vt[:self.n_factors].T
        W = (C - center) @ projection
        # Unit-SD factors make the confounding/conditioning diagnostic scale invariant.
        factor_scale = W.std(axis=0)
        projection = projection / factor_scale
        W = (C - center) @ projection
        D = np.column_stack([B, W])
        condition = float(np.linalg.cond(D))
        if len(Y) <= D.shape[1] or np.linalg.matrix_rank(D) < D.shape[1] or condition > self.max_condition:
            raise ValueError("Biology and control factors are confounded, rank deficient, or ill-conditioned; correction is not identifiable")
        coef = np.linalg.lstsq(D, Y, rcond=None)[0]
        self.control_positions_ = positions
        self.control_center_ = center
        self.projection_ = projection
        self.coefficients_ = coef[-self.n_factors:]
        self.diagnostics_ = dict(design_condition=condition,
            control_singular_values=singular.tolist(),
            factor_biology_r2=(1 - np.sum((W - B @ np.linalg.lstsq(B, W, rcond=None)[0])**2, axis=0)
                               / np.sum(W**2, axis=0)).tolist())
        self._record(X)
        return self

    def transform(self, X):
        Y = self._matrix(X).T
        # Per-sample products also avoid batch-size-dependent BLAS rounding.
        corrected = []
        for row in Y:
            factors = (row[self.control_positions_] - self.control_center_) @ self.projection_
            corrected.append(row - factors @ self.coefficients_)
        result = np.asarray(corrected).T
        if not np.isfinite(result).all():
            raise ValueError("Correction produced nonfinite expression")
        return pd.DataFrame(result, index=self.genes_, columns=X.columns)
