"""Native, serialized exact fSVA. The fixed training database is retained."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from omiclass.batch.reference import ExpressionReference


class FrozenSVA(ExpressionReference):
    method = "fsva_exact"

    def __init__(self, design_columns, n_factors=1, input_scale="tpm", rscript_path=None,
                 r_library_path=None, random_state=11):
        super().__init__(input_scale)
        if not design_columns or n_factors < 1:
            raise ValueError("Exact fSVA requires a biological design and positive n_factors")
        self.design_columns = list(design_columns)
        self.n_factors = int(n_factors)
        self.random_state = int(random_state)
        candidate = Path(sys.executable).parent / "Rscript"
        executable = rscript_path or (candidate if candidate.is_file() else shutil.which("Rscript"))
        if not executable:
            raise ValueError("Rscript executable is required for native exact fSVA")
        self.rscript_path = str(Path(executable).resolve())
        self.r_library_path = str(Path(r_library_path).resolve()) if r_library_path else None

    def _run(self, operation, values, samples, metadata=None):
        with tempfile.TemporaryDirectory(prefix="omiclass_fsva_") as tmp:
            root = Path(tmp)
            genes = [self.genes_[p] for p in self.variable_positions_]
            pd.DataFrame(values, index=genes, columns=samples).to_csv(root / "expression.csv")
            if metadata is not None:
                metadata.to_csv(root / "metadata.csv")
            else:
                (root / "state.rds").write_bytes(self.r_state_.tobytes())
            (root / "settings.json").write_text(json.dumps(dict(n_factors=self.n_factors, random_state=self.random_state)))
            env = os.environ.copy()
            env.pop("R_HOME", None)  # Let this Rscript select its own R installation.
            if self.r_library_path:
                env["R_LIBS_USER"] = self.r_library_path
            command = [self.rscript_path, "--vanilla", str(Path(__file__).with_name("fsva_exact.R")), operation, tmp]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"Native exact fSVA {operation} failed:\n{result.stdout[-3000:]}\n{result.stderr[-5000:]}")
            corrected = pd.read_csv(root / "corrected.csv", index_col=0).to_numpy(dtype=float)
            if operation == "fit":
                self.r_state_ = np.frombuffer((root / "state.rds").read_bytes(), dtype=np.uint8).copy()
                self.diagnostics_ = json.loads((root / "diagnostics.json").read_text())
            return corrected

    @staticmethod
    def _hash(values):
        return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()

    def fit(self, X, metadata=None):
        values = self._matrix(X, fitting=True)
        if metadata is None or not metadata.index.is_unique:
            raise ValueError("Exact fSVA requires unique training metadata indexed by sample")
        meta = metadata.loc[X.columns, self.design_columns]
        if meta.isna().any().any():
            raise ValueError("Missing values in fSVA training biological design")
        self._record(X)
        # Remove only training-constant genes from SVA estimation. They retain
        # their input values in both database correction and new-sample output.
        self.variable_positions_ = np.flatnonzero(np.ptp(values, axis=1) > 0)
        if len(self.variable_positions_) <= self.n_factors:
            raise ValueError("Insufficient variable training genes for exact fSVA")
        corrected = self._run("fit", values[self.variable_positions_], X.columns, meta)
        self.training_corrected_ = values.copy()
        self.training_corrected_[self.variable_positions_] = corrected
        self.training_hashes_ = [self._hash(values[:, j]) for j in range(values.shape[1])]
        return self

    def transform(self, X):
        values = self._matrix(X)
        result = values.copy()
        known = {sample: j for j, sample in enumerate(self.training_samples_)}
        new = []
        for j, sample in enumerate(X.columns):
            if sample in known:
                p = known[sample]
                if self._hash(values[:, j]) != self.training_hashes_[p]:
                    raise ValueError(f"Training sample ID reused with changed expression: {sample}")
                result[:, j] = self.training_corrected_[:, p]
            else:
                new.append(j)
        if new:
            subset = np.ix_(self.variable_positions_, new)
            result[subset] = self._run("transform", values[subset], X.columns[new])
        if not np.isfinite(result).all():
            raise ValueError("Exact fSVA produced nonfinite output")
        return pd.DataFrame(result, index=self.genes_, columns=X.columns)
