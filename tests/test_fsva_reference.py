import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from omiclass.batch.reference import ExpressionReference
from omiclass.batch.fsva import FrozenSVA


def test_fsva_requires_training_design_and_variable_genes():
    with pytest.raises(ValueError, match="biological design"):
        FrozenSVA([], rscript_path="/usr/bin/Rscript")
    X = pd.DataFrame(np.ones((5, 12)), index=list("abcde"), columns=[str(i) for i in range(12)])
    meta = pd.DataFrame({"target": ["A", "B"] * 6}, index=X.columns)
    with pytest.raises(ValueError, match="variable training genes"):
        FrozenSVA(["target"], rscript_path="/usr/bin/Rscript").fit(X, meta)


def test_native_exact_fsva_fit_transform_reload_and_independence(tmp_path):
    library = os.environ.get("OMICLASS_TEST_SVA_LIB")
    if not library:
        pytest.skip("Set OMICLASS_TEST_SVA_LIB to an installed sva R library for native integration")
    rng = np.random.default_rng(61)
    target = np.tile([0, 1], 30)
    unwanted = rng.normal(size=60) + .3 * target
    loadings = rng.normal(size=240)
    biology = np.r_[np.ones(60), np.zeros(180)]
    Y = 12 + loadings[:, None] * unwanted + biology[:, None] * target
    Y += rng.normal(scale=.1, size=Y.shape)
    Y = np.row_stack([Y, np.ones(60) * 2])  # constant row stays uncorrected
    X = pd.DataFrame(Y, index=[f"g{i}" for i in range(241)], columns=[f"p{i}" for i in range(60)])
    meta = pd.DataFrame({"target": np.where(target, "B", "A")}, index=X.columns)
    train, external = X.iloc[:, :56], X.iloc[:, 56:]
    ref = FrozenSVA(["target"], input_scale="log1p_tpm", r_library_path=library,
        rscript_path=os.environ.get("OMICLASS_TEST_RSCRIPT", str(Path(sys.executable).parent / "Rscript")))
    corrected = ref.fit_transform(train, meta.loc[train.columns])
    np.testing.assert_array_equal(corrected.iloc[-1], train.iloc[-1])
    path = tmp_path / "reference.npz"
    ref.save(path)
    state_bytes = path.read_bytes()
    result = ref.transform(external)
    pd.testing.assert_frame_equal(result.iloc[:, :1], ref.transform(external.iloc[:, :1]))
    restored = ExpressionReference.load(path)
    pd.testing.assert_frame_equal(result, restored.transform(external.iloc[::-1]))
    pd.testing.assert_frame_equal(corrected, restored.transform(train))
    assert path.read_bytes() == state_bytes
    assert ref.diagnostics_["implementation"] == "sva::fsva(method='exact')"
    assert np.linalg.norm(corrected.values[:-1] - train.values[:-1]) > 0
    design = np.column_stack([np.ones(train.shape[1]), target[:56]])
    learned_effect = np.linalg.lstsq(design, corrected.values[:-1].T, rcond=None)[0][1]
    np.testing.assert_allclose(learned_effect[:60], biology[:60], atol=.3)
    with pytest.raises(ValueError, match="reused with changed expression"):
        ref.transform(train + 1)
    with pytest.raises(ValueError, match="Incompatible gene schema"):
        ref.transform(external.iloc[:-1])
