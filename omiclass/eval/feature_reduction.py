"""The historical feature-reduction diagnostic is disabled for public evaluation."""
from dataclasses import dataclass
from typing import Dict, List, Sequence
import pandas as pd


DISABLED_REASON = (
    'Feature-reduction evaluation is disabled: its feature ranking used validation '
    'labels before evaluating feature subsets on those same validation rows. '
    'No feature-reduction scores or curves are produced.'
)


@dataclass
class FeatureReductionResult:
    ratios: List[float]
    k_values: List[int]
    mean: Dict[str, List[float]]
    std: Dict[str, List[float]]
    ranked_features: List[str]


def evaluate_feature_reduction(*, estimator, X: pd.DataFrame, y: pd.Series,
        k_ratios: Sequence[float], min_features: int, n_splits: int,
        random_state: int, n_repeats: int = 3) -> FeatureReductionResult:
    raise ValueError(DISABLED_REASON)
