"""Fold-fitted DE-guided one-vs-rest top scoring pairs.

No randomness, cohort normalization, imputation, or expression passthrough.
The DataFrame contract preserves gene identities through sklearn cloning.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class OvRTSPSelector(TransformerMixin, BaseEstimator):
    def __init__(self, g=50, k=25, max_gene_reuse=2, min_expression=0.0, min_fraction=0.1):
        self.g = g
        self.k = k
        self.max_gene_reuse = max_gene_reuse
        self.min_expression = min_expression
        self.min_fraction = min_fraction

    @staticmethod
    def _matrix(X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("pairs requires a DataFrame with gene identifiers as columns")
        if not X.columns.is_unique or X.columns.hasnans or not all(isinstance(g, str) for g in X.columns):
            raise ValueError("pairs requires unique, nonmissing string gene identifiers")
        if not X.index.is_unique:
            raise ValueError("pairs requires unique sample identifiers")
        values = X.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("pairs expression must be finite; missing expression is never zero-filled")
        return values

    def fit(self, X, y):
        values = self._matrix(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(X) or pd.isna(labels).any():
            raise ValueError("pairs requires one nonmissing label per training sample")
        if isinstance(y, pd.Series) and not X.index.equals(y.index):
            raise ValueError("pairs X and y sample indices must match")
        if any(not isinstance(v, (int, np.integer)) or v <= 0 for v in (self.g, self.k, self.max_gene_reuse)):
            raise ValueError("g, k and max_gene_reuse must be positive integers")
        if not 0 <= self.min_fraction <= 1 or not np.isfinite(self.min_expression):
            raise ValueError("Invalid expression filter")
        self.classes_, counts = np.unique(labels, return_counts=True)
        if len(counts) < 2 or counts.min() < 2:
            raise ValueError("pairs DE requires at least two training samples per class")
        self.feature_names_in_ = X.columns.to_numpy(copy=True)
        self.n_features_in_ = X.shape[1]
        masks = [labels == cls for cls in self.classes_]
        # Preserve rare-class markers: prevalence need only pass in one class.
        expressed = np.max([(values[m] > self.min_expression).mean(axis=0) for m in masks], axis=0)
        keep = (expressed >= self.min_fraction) & (np.ptp(values, axis=0) > 0)
        genes = self.feature_names_in_[keep]
        data = values[:, keep]
        if not len(genes):
            raise ValueError("No genes pass training expression filters")

        self.candidate_genes_ = {}
        self.selected_pairs_ = []
        selected_lookup = {}
        records = []
        # Binary: the reverse comparison carries no additional information.
        targets = range(1) if len(masks) == 2 else range(len(masks))
        for target in targets:
            mask = masks[target]
            other_masks = [m for i, m in enumerate(masks) if i != target]
            effect = data[mask].mean(axis=0) - data[~mask].mean(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # constant within-class genes
                pvals = ttest_ind(data[mask], data[~mask], axis=0, equal_var=False).pvalue
            pvals = np.nan_to_num(pvals, nan=1.0)
            # Stable ties: p-value, absolute mean difference, then gene identifier.
            order = np.lexsort((genes, -np.abs(effect), pvals))
            up = np.array([i for i in order if effect[i] > 0][:self.g], dtype=int)
            down = np.array([i for i in order if effect[i] < 0][:self.g], dtype=int)
            cls = self.classes_[target]
            self.candidate_genes_[cls] = dict(up=genes[up].tolist(), down=genes[down].tolist())
            if not len(up) or not len(down):
                continue
            # Bound memory to samples x 256 pairs, even with many classes/genes.
            a = np.repeat(up, len(down))
            b = np.tile(down, len(up))
            target_freq = np.empty(len(a))
            rest_freq = np.empty(len(a))
            for start in range(0, len(a), 256):
                stop = min(start + 256, len(a))
                greater = data[:, a[start:stop]] > data[:, b[start:stop]]
                target_freq[start:stop] = greater[mask].mean(axis=0)
                rest_freq[start:stop] = np.mean([greater[m].mean(axis=0) for m in other_masks], axis=0)
            scores = target_freq - rest_freq
            ranked = np.lexsort((genes[b], genes[a], -scores))
            reuse = {}
            taken = 0
            for j in ranked:
                if scores[j] <= 0 or taken >= self.k:
                    break
                ga, gb = str(genes[a[j]]), str(genes[b[j]])
                if reuse.get(ga, 0) >= self.max_gene_reuse or reuse.get(gb, 0) >= self.max_gene_reuse:
                    continue
                reuse[ga] = reuse.get(ga, 0) + 1
                reuse[gb] = reuse.get(gb, 0) + 1
                taken += 1
                key = tuple(sorted((ga, gb)))
                if key not in selected_lookup:
                    selected_lookup[key] = len(self.selected_pairs_)
                    self.selected_pairs_.append((ga, gb))
                feature_idx = selected_lookup[key]
                fa, fb = self.selected_pairs_[feature_idx]
                records.append(dict(target_class=cls, gene_a=ga, gene_b=gb,
                    score=float(scores[j]), p_target=float(target_freq[j]), p_rest=float(rest_freq[j]),
                    feature=f"pair_{feature_idx:04d}", feature_gene_a=fa, feature_gene_b=fb,
                    orientation=1 if (ga, gb) == (fa, fb) else -1))
        if not self.selected_pairs_:
            raise ValueError("No informative up/down gene pairs found in training data")
        self.selection_report_ = pd.DataFrame(records)
        self.output_features_ = np.array([f"pair_{i:04d}" for i in range(len(self.selected_pairs_))], dtype=object)
        self.required_genes_ = list(dict.fromkeys(g for pair in self.selected_pairs_ for g in pair))
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_pairs_")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("pairs requires a DataFrame with gene identifiers as columns")
        missing = [g for g in self.required_genes_ if g not in X.columns]
        if missing:
            raise ValueError(f"Missing genes required by fitted pairs: {missing[:10]}")
        self._matrix(X.loc[:, self.required_genes_])
        a = X.loc[:, [p[0] for p in self.selected_pairs_]].to_numpy(dtype=float)
        b = X.loc[:, [p[1] for p in self.selected_pairs_]].to_numpy(dtype=float)
        result = (a > b).astype(np.int8) - (a < b).astype(np.int8)
        return pd.DataFrame(result, index=X.index, columns=self.output_features_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_features_")
        return self.output_features_.copy()


def fitted_pair_selectors(model):
    """Find every fitted selector, including calibration members and stacks."""
    named = getattr(model, "named_steps", {})
    if "pairs" in named:
        return [named["pairs"]]
    out = []
    for member in getattr(model, "calibrated_classifiers_", []) or []:
        out.extend(fitted_pair_selectors(member.estimator))
    for estimator in getattr(model, "estimators_", []) or []:
        out.extend(fitted_pair_selectors(estimator))
    return out


def prepare_pair_prediction(model, X):
    """Strict external schema check, without requiring unused training genes."""
    selectors = fitted_pair_selectors(model)
    required = list(dict.fromkeys(g for selector in selectors for g in selector.required_genes_))
    if not selectors:
        raise ValueError("Model has no fitted pair selectors")
    missing = [g for g in required if g not in X.columns]
    if missing:
        raise ValueError(f"Missing genes required by fitted pairs: {missing[:10]}")
    OvRTSPSelector._matrix(X.loc[:, required])
    return X


def pair_input_expression(X, cfg):
    """Explicit scale avoids cohort-dependent automatic log-scale detection."""
    if cfg.batch_correction.input_scale == "tpm" and cfg.runtime.log_transform:
        return np.log1p(X)
    return X
