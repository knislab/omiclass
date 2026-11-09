"""Use the requested sklearn scorer consistently for training-CV selection."""
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer


def fold_selection_score(scoring, model, X, y, metrics):
    # Reuse existing default metrics without another prediction pass.
    if scoring in {"f1_macro", "accuracy"}:
        return float(metrics[scoring])
    return float(get_scorer(scoring)(model, X, y))


def selection_summary(folds, scoring):
    frame = pd.DataFrame(folds)
    column = "selection_score" if "selection_score" in frame else scoring
    if column not in frame:
        raise ValueError(f"No training-CV scores for requested scorer {scoring!r}")
    scores = pd.to_numeric(frame[column], errors="coerce")
    if scores.empty or not np.isfinite(scores).all():
        raise ValueError(f"Nonfinite training-CV scores for requested scorer {scoring!r}")
    return dict(selection_score_mean=float(scores.mean()),
                selection_score_std=float(scores.std()), selection_scoring=scoring)


def rank_candidates(board, available, scoring):
    """Higher sklearn scores win (including negated losses); ties use names."""
    frame = board.loc[board["model"].isin(available)].copy()
    if "status" in frame:
        frame = frame.loc[frame.status.eq("ok")]
    column = "selection_score_mean"
    if column not in frame:
        # Existing/default leaderboards and older consumers retain these columns.
        column = f"{scoring}_mean"
    if column not in frame:
        raise ValueError(f"No training-CV column for requested scorer {scoring!r}")
    if "selection_scoring" in frame:
        frame = frame.loc[frame.selection_scoring.eq(scoring)]
    scores = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[np.isfinite(scores)].copy()
    frame[column] = scores.loc[frame.index]
    if frame.empty:
        raise ValueError(f"No completed candidates with finite training-CV scores for {scoring!r}")
    return frame.sort_values([column, "model"], ascending=[False, True], kind="mergesort"), column
