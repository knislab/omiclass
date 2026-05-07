# generic-classifier/utils/imbalance.py
from __future__ import annotations
from enum import Enum
import pandas as pd

class Severity(str, Enum):
    BALANCED = "balanced"
    MILD     = "mild"
    SEVERE   = "severe"

def assess_imbalance(y: pd.Series, *, mild_threshold: float = 1.5, severe_threshold: float = 3.0):
    counts = y.value_counts()
    maxc, minc = int(counts.max()), int(counts.min())
    ir = maxc / max(1, minc)
    if ir <= mild_threshold:         
        sev = Severity.BALANCED
    elif ir <= severe_threshold:     
        sev = Severity.MILD
    else:                            
        sev = Severity.SEVERE
    return {"severity": sev, "counts": counts.to_dict(), "imbalance_ratio": ir}
