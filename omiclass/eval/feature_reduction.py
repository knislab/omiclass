from dataclasses import dataclass
import copy
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import get_scorer

METRICS = ["accuracy", "f1_macro", "average_precision"]


@dataclass
class FeatureReductionResult:
    ratios: List[float]
    k_values: List[int]
    mean: Dict[str, List[float]]
    std: Dict[str, List[float]]
    ranked_features: List[str]


def evaluate_feature_reduction(
    *,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    k_ratios: Sequence[float],
    min_features: int,
    n_splits: int,
    random_state: int,
    n_repeats: int = 3,
) -> FeatureReductionResult:

    n_features = X.shape[1]

    # compute actual k values safely
    k_values = [
        max(min_features, int(round(r * n_features)))
        for r in k_ratios
    ]
    k_values = sorted(set(min(k, n_features) for k in k_values))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scorers = {m: get_scorer(m) for m in METRICS}
    rank_scorer = get_scorer("f1_macro")

    curves = {m: [] for m in METRICS}
    ranked_features_global = None
    importance_sum = pd.Series(0.0, index=X.columns)
    importance_count = pd.Series(0, index=X.columns)

    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        if hasattr(estimator, "fit_honest"):
            model = copy.deepcopy(estimator)
        else:
            model = clone(estimator)
        model.fit(Xtr, ytr)

        pi = permutation_importance(
            model,
            Xva, yva,
            scoring=rank_scorer,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1,
        )

        fold_importance = pd.Series(pi.importances_mean, index=X.columns)

        importance_sum += fold_importance
        importance_count += 1

        ranked_features = fold_importance.sort_values(ascending=False).index.tolist()

        for k in k_values:
            feats = ranked_features[:k]
            if hasattr(estimator, "fit_honest"):
                model_k = copy.deepcopy(estimator)
            else:
                model_k = clone(estimator)
            model_k.fit(Xtr[feats], ytr)

            for m, scorer in scorers.items():
                curves[m].append(scorer(model_k, Xva[feats], yva))

    mean, std = {}, {}
    for m in METRICS:
        arr = np.array(curves[m]).reshape(n_splits, len(k_values))
        mean[m] = arr.mean(axis=0).tolist()
        std[m] = arr.std(axis=0, ddof=1).tolist()

    mean_importance = importance_sum / importance_count
    ranked_features_global = mean_importance.sort_values(ascending=False).index.tolist()

    return FeatureReductionResult(
        ratios=list(k_ratios),
        k_values=k_values,
        mean=mean,
        std=std,
        ranked_features=ranked_features_global,
    )