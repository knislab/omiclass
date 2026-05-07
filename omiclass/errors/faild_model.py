import numpy as np

def _failed_row(model_key: str, err: Exception) -> dict:
    return {
        "model": model_key,
        "f1_macro_mean": np.nan,
        "f1_macro_std": np.nan,
        "accuracy_mean": np.nan,
        "accuracy_std": np.nan,
        "status": "failed",
        "error": f"{type(err).__name__}: {err}",
    }
