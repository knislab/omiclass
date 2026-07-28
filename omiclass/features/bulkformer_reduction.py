from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from omiclass.features.embedding_backend import extraction_signature


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    values = p[valid]
    if not len(values):
        return out
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    out[valid] = restored
    return out


def _frame(X: Any, expected=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        result = X.copy()
    else:
        arr = np.asarray(X)
        columns = list(expected) if expected is not None else [f"x{i}" for i in range(arr.shape[1])]
        result = pd.DataFrame(arr, columns=columns)
    if result.columns.duplicated().any():
        raise ValueError("Embedding columns must be unique.")
    if expected is not None:
        expected = list(expected)
        missing = [c for c in expected if c not in result.columns]
        extra = [c for c in result.columns if c not in set(expected)]
        if missing or extra:
            raise ValueError(
                f"Embedding schema mismatch; missing={missing[:5]}, extra={extra[:5]}."
            )
        result = result.loc[:, expected]
    result = result.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Embeddings contain non-finite values.")
    return result


class SupervisedEmbeddingSelector(BaseEstimator, TransformerMixin):
    """Contrast-balanced selector whose only ranking key is signed Welch t."""

    def __init__(self, n_dimensions=200, one_vs_rest_fraction=0.5,
                 close_pair_fraction=0.5, close_pair_count=None,
                 close_pair_pca_components=10, random_state=None):
        self.n_dimensions = n_dimensions
        self.one_vs_rest_fraction = one_vs_rest_fraction
        self.close_pair_fraction = close_pair_fraction
        self.close_pair_count = close_pair_count
        self.close_pair_pca_components = close_pair_pca_components
        self.random_state = random_state

    def _find_close_pairs(self, X, y, classes):
        self.close_pair_scaler_ = StandardScaler().fit(X)
        scaled = self.close_pair_scaler_.transform(X)
        n = min(int(self.close_pair_pca_components), X.shape[1], max(1, X.shape[0] - 1))
        self.close_pair_pca_ = PCA(n_components=n, random_state=self.random_state).fit(scaled)
        scores = pd.DataFrame(self.close_pair_pca_.transform(scaled), index=X.index)
        centroids = {c: scores.loc[y == c].mean().to_numpy() for c in classes}
        matrix = pd.DataFrame(0.0, index=classes, columns=classes)
        pairs = []
        for a, b in combinations(classes, 2):
            distance = float(np.linalg.norm(centroids[a] - centroids[b]))
            matrix.loc[a, b] = matrix.loc[b, a] = distance
            pairs.append((a, b, distance))
        pairs.sort(key=lambda item: (item[2], str(item[0]), str(item[1])))
        count = self.close_pair_count or max(1, len(classes) - 1)
        self.class_centroid_distances_ = matrix
        self.selected_close_pairs_ = pairs[: min(int(count), len(pairs))]
        return self.selected_close_pairs_

    @staticmethod
    def _rank(X, positive, negative, source, contrast):
        a = X.loc[positive].to_numpy(dtype=float)
        b = X.loc[negative].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            statistic, pvalue = ttest_ind(a, b, equal_var=False, axis=0)
        statistic, pvalue = np.asarray(statistic), np.asarray(pvalue)
        difference = a.mean(axis=0) - b.mean(axis=0)
        # Constant but separated groups have an infinite limiting Welch statistic.
        separated = np.isnan(statistic) & (difference != 0)
        statistic[separated] = np.sign(difference[separated]) * np.inf
        fdr = benjamini_hochberg(pvalue)
        records = [{
            "dimension": str(name), "column_index": i, "source": source,
            "contrast": contrast, "t_statistic": float(statistic[i]),
            "mean_difference": float(difference[i]), "p_value": float(pvalue[i]),
            "fdr_adjusted_p_value": float(fdr[i]),
        } for i, name in enumerate(X.columns)]
        positive_records = [r for r in records if r["t_statistic"] > 0]
        negative_records = [r for r in records if r["t_statistic"] < 0]
        positive_records.sort(key=lambda r: (-r["t_statistic"], r["column_index"]))
        negative_records.sort(key=lambda r: (r["t_statistic"], r["column_index"]))
        for direction, values in (("positive", positive_records), ("negative", negative_records)):
            for rank, record in enumerate(values, 1):
                record.update(direction=direction, directional_rank=rank)
        return positive_records, negative_records

    @staticmethod
    def _take(groups, budget, seen):
        chosen, pointers = [], [0] * len(groups)
        while len(chosen) < budget:
            progressed = False
            for i, records in enumerate(groups):
                while pointers[i] < len(records) and records[pointers[i]]["dimension"] in seen:
                    pointers[i] += 1
                if pointers[i] < len(records):
                    record = dict(records[pointers[i]])
                    pointers[i] += 1
                    seen.add(record["dimension"])
                    chosen.append(record)
                    progressed = True
                    if len(chosen) == budget:
                        break
            if not progressed:
                break
        return chosen

    def fit(self, X, y):
        X = _frame(X)
        y = pd.Series(np.asarray(y), index=X.index)
        classes = sorted(pd.unique(y), key=str)
        if len(classes) < 2:
            raise ValueError("Supervised embedding selection requires at least two classes.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        target = min(int(self.n_dimensions), X.shape[1])
        if len(classes) == 2:
            a, b = classes
            ovr = [(y == a, y == b, "binary", f"{a}_vs_{b}")]
            pair, one_budget, pair_budget = [], target, 0
            self.class_centroid_distances_, self.selected_close_pairs_ = pd.DataFrame(), []
        else:
            ovr = [(y == c, y != c, "one_vs_rest", f"{c}_vs_rest") for c in classes]
            close = self._find_close_pairs(X, y, classes)
            pair = [(y == a, y == b, "close_pair", f"{a}_vs_{b}") for a, b, _ in close]
            fraction = self.one_vs_rest_fraction + self.close_pair_fraction
            one_budget = int(round(target * self.one_vs_rest_fraction / fraction))
            pair_budget = target - one_budget
            if not pair:
                one_budget, pair_budget = target, 0
        def ranked_groups(contrasts):
            result = []
            for pos, neg, source, name in contrasts:
                result.extend(self._rank(X, pos, neg, source, name))
            return result
        ovr_groups, pair_groups = ranked_groups(ovr), ranked_groups(pair)
        seen = set()
        selected = self._take(ovr_groups, one_budget, seen)
        selected += self._take(pair_groups, pair_budget, seen)
        selected += self._take(ovr_groups + pair_groups, target - len(selected), seen)
        # Undefined/equal dimensions are a deterministic last-resort budget fill.
        for i, name in enumerate(X.columns):
            if len(selected) == target:
                break
            if str(name) not in seen:
                selected.append({"dimension": str(name), "column_index": i, "source": "fallback",
                                 "contrast": "undefined", "direction": "undefined",
                                 "directional_rank": np.nan, "t_statistic": np.nan,
                                 "mean_difference": np.nan, "p_value": np.nan,
                                 "fdr_adjusted_p_value": np.nan})
                seen.add(str(name))
        for rank, record in enumerate(selected, 1):
            record["selection_rank"] = rank
        self.selection_report_ = pd.DataFrame(selected)
        self.selected_features_ = self.selection_report_["dimension"].tolist()
        self.selected_indices_ = [int(X.columns.get_loc(c)) for c in self.selected_features_]
        self.classes_ = np.asarray(classes)
        self.effective_n_dimensions_ = len(selected)
        self.one_vs_rest_budget_, self.close_pair_budget_ = one_budget, pair_budget
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_features_")
        return _frame(X, self.feature_names_in_).loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


class EmbeddingRepresentation(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="full", supervised_dimensions=200, pca_components=50,
                 one_vs_rest_fraction=0.5, close_pair_fraction=0.5,
                 close_pair_count=None, close_pair_pca_components=10, random_state=None,
                 extraction_signature=None):
        self.strategy = strategy
        self.supervised_dimensions = supervised_dimensions
        self.pca_components = pca_components
        self.one_vs_rest_fraction = one_vs_rest_fraction
        self.close_pair_fraction = close_pair_fraction
        self.close_pair_count = close_pair_count
        self.close_pair_pca_components = close_pair_pca_components
        self.random_state = random_state
        self.extraction_signature = extraction_signature

    def fit(self, X, y=None):
        X = _frame(X)
        self.feature_names_in_, self.n_features_in_ = np.asarray(X.columns, dtype=object), X.shape[1]
        self.extraction_signature_ = dict(self.extraction_signature or {})
        self.selector_ = None
        current = X
        if self.strategy in {"supervised", "supervised_pca"}:
            self.selector_ = SupervisedEmbeddingSelector(
                self.supervised_dimensions, self.one_vs_rest_fraction,
                self.close_pair_fraction, self.close_pair_count,
                self.close_pair_pca_components, self.random_state,
            ).fit(X, y)
            current = self.selector_.transform(X)
        self.scaler_, self.pca_ = None, None
        if self.strategy in {"pca", "supervised_pca"}:
            self.scaler_ = StandardScaler().fit(current)
            scaled = self.scaler_.transform(current)
            n = min(int(self.pca_components), current.shape[1], max(1, current.shape[0] - 1))
            self.pca_ = PCA(n_components=n, random_state=self.random_state).fit(scaled)
            self.output_features_ = [f"bulkformer_pc_{i + 1:04d}" for i in range(n)]
        else:
            self.output_features_ = list(current.columns)
        self.effective_n_features_ = len(self.output_features_)
        return self

    def transform(self, X):
        check_is_fitted(self, "output_features_")
        X = _frame(X, self.feature_names_in_)
        current = self.selector_.transform(X) if self.selector_ is not None else X
        if self.pca_ is None:
            return current
        values = self.pca_.transform(self.scaler_.transform(current))
        return pd.DataFrame(values, index=X.index, columns=self.output_features_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.output_features_, dtype=object)


def build_embedding_representation(cfg, runtime_seed, backend="bulkformer"):
    reduction = cfg.reduction
    seed = runtime_seed if reduction.random_state is None else reduction.random_state
    signature = extraction_signature(backend, cfg)
    return EmbeddingRepresentation(
        reduction.strategy, reduction.supervised_dimensions, reduction.pca_components,
        reduction.one_vs_rest_fraction, reduction.close_pair_fraction,
        reduction.close_pair_count, reduction.close_pair_pca_components, seed, signature,
    )


def build_bulkformer_representation(cfg, runtime_seed):
    """Compatibility wrapper; the transformer itself is backend-independent."""
    return build_embedding_representation(cfg, runtime_seed, backend="bulkformer")


# Compatibility import for existing callers and serialized estimators.
BulkFormerRepresentation = EmbeddingRepresentation
