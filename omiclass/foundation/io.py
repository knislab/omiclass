"""Run-owned arrays and token records; no implicit manuscript paths."""
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np
import pandas as pd


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), allow_nan=False) + "\n")


def save_frame(path, frame):
    np.savez(path, values=frame.to_numpy(), index=np.asarray(frame.index, dtype=str), columns=np.asarray(frame.columns, dtype=str))


def load_frame(path):
    with np.load(path, allow_pickle=False) as z:
        return pd.DataFrame(z["values"], index=z["index"].astype(str), columns=z["columns"].astype(str))


@contextmanager
def token_record(cache, record):
    cache = Path(cache).resolve()
    path = (cache / record["file"]).resolve()
    if not path.is_relative_to(cache):
        raise ValueError("Token record escapes its cache")
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            yield {"tokens": array, "mask": np.ones(array.shape[0], dtype=bool)}
        finally:
            del array
    else:
        with np.load(path, allow_pickle=False) as z:
            yield z
