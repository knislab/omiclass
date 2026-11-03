import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from omiclass.batch.reference import ExpressionReference, FrozenQuantile, ReferenceRUV
from omiclass.batch.reference_pipeline import fit_reference, transform_reference, read_expression


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(11)
    n, g = 120, 30
    target = np.tile([0., 1.], n // 2)
    w = rng.normal(size=n) + target * .6  # partial biological confounding
    loading = rng.uniform(.1, .5, g)
    biology = np.r_[np.zeros(8), np.repeat(1.4, g - 8)]
    Y = 8 + w[:, None] * loading + target[:, None] * biology
    Y += rng.normal(scale=.003, size=Y.shape)
    X = pd.DataFrame(Y.T, index=[f"g{i}" for i in range(g)], columns=[f"p{i}" for i in range(n)])
    meta = pd.DataFrame({"target": target}, index=X.columns)
    return X, meta, biology, loading


def make(kind, X):
    if kind == "frozen_quantile":
        return FrozenQuantile("log1p_tpm")
    return ReferenceRUV(X.index[:8].tolist(), "Synthetic controls have exactly zero target loading",
                        design_columns=["target"], input_scale="log1p_tpm")


@pytest.mark.parametrize("kind", ["frozen_quantile", "reference_ruv"])
def test_frozen_contract(kind, synthetic, tmp_path):
    X, meta, _, _ = synthetic
    train, external = X.iloc[:, :90], X.iloc[:, 90:].copy()
    external.columns = ["unseen_cohort_" + s for s in external.columns]
    ref = make(kind, X).fit(train, meta.loc[train.columns])
    state = copy.deepcopy(vars(ref))
    training = ref.transform(train)
    together = ref.transform(external)
    for sample in external:
        pd.testing.assert_frame_equal(together[[sample]], ref.transform(external[[sample]]))
    pd.testing.assert_frame_equal(together, ref.transform(external.iloc[::-1]))
    ref.transform(external * 2)
    pd.testing.assert_frame_equal(training, ref.transform(train))
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, vars(ref)[key])
        else:
            assert value == vars(ref)[key]
    path = tmp_path / "reference.npz"
    ref.save(path)
    restored = ExpressionReference.load(path)
    pd.testing.assert_frame_equal(together, restored.transform(external))
    assert ref.training_samples_ == train.columns.tolist()
    with pytest.raises(ValueError, match="Incompatible gene schema"):
        ref.transform(external.iloc[:-1])
    with pytest.raises(ValueError, match="Incompatible gene schema"):
        ref.transform(pd.concat([external, external.iloc[:1].rename(index={"g0": "extra"})]))
    with pytest.raises(ValueError, match="unique"):
        ref.transform(pd.concat([external, external.iloc[:1]]))
    with pytest.raises(ValueError, match="nonfinite"):
        ref.transform(external.mask(external > 8))


def test_quantile_known_target_ties_and_scale():
    X = pd.DataFrame([[0., 2.], [2., 4.], [8., 10.], [20., 22.]],
                     index=list("abcd"), columns=["train1", "train2"])
    ref = FrozenQuantile("log1p_tpm").fit(X)
    np.testing.assert_array_equal(ref.target_, [1, 3, 9, 21])
    tied = pd.DataFrame([0., 0., 0., 5.], index=X.index, columns=["external"])
    np.testing.assert_array_equal(ref.transform(tied).values[:, 0], [3, 3, 3, 21])
    log_ref = FrozenQuantile("tpm").fit(np.expm1(X))
    np.testing.assert_allclose(log_ref.transform(np.expm1(tied)), ref.transform(tied))


def test_ruv_removes_nuisance_and_preserves_conditional_target(synthetic):
    X, meta, biology, loading = synthetic
    ref = make("reference_ruv", X).fit(X, meta)
    corrected = ref.transform(X).T
    B = np.column_stack([np.ones(len(meta)), meta.target])
    effect = np.linalg.lstsq(B, corrected, rcond=None)[0][1]
    np.testing.assert_allclose(effect, biology, atol=.015)
    # The fully independent cohort has a large nuisance shift, same loadings.
    external = pd.DataFrame((8 + 3 * loading + biology).reshape(-1, 1), index=X.index, columns=["DKFZ"])
    expected = 8 + biology + np.mean((X.T.to_numpy()[:, :8] - 8) / loading[:8]) * loading
    np.testing.assert_allclose(ref.transform(external).values[:, 0], expected, atol=.02)
    assert np.var(corrected.values[:60, 10]) < np.var(X.T.values[:60, 10])


def test_ruv_rejects_unjustified_controls_rank_and_confounding(synthetic):
    X, meta, _, _ = synthetic
    with pytest.raises(ValueError, match="justification"):
        ReferenceRUV(["g0"], "", design_columns=["target"])
    with pytest.raises(ValueError, match="biological design"):
        ReferenceRUV(["g0"], "valid controls")
    constant = X.copy()
    constant.iloc[:8] = 1
    with pytest.raises(ValueError, match="rank"):
        make("reference_ruv", X).fit(constant, meta)
    confounded = X.copy()
    confounded.iloc[:8] = 2 + meta.target.values
    with pytest.raises(ValueError, match="confounded"):
        make("reference_ruv", X).fit(confounded, meta)


@pytest.mark.parametrize("kind", ["frozen_quantile", "reference_ruv"])
def test_pipeline_training_boundary_and_label_free_prediction(kind, synthetic, tmp_path):
    X, meta, _, _ = synthetic
    train = X.columns[:90].tolist()
    ids = tmp_path / "train.txt"
    ids.write_text("\n".join(train))
    controls = tmp_path / "controls.txt"
    controls.write_text("\n".join(X.index[:8]))
    meta_path = tmp_path / "metadata.csv"
    meta.loc[train].to_csv(meta_path)
    bc = SimpleNamespace(algorithm=kind, input_scale="log1p_tpm", training_samples_path=ids,
        reference_path=None, control_genes_path=controls, control_justification="Known synthetic null genes",
        n_factors=1, design_columns=[])
    ctx = SimpleNamespace(cfg=SimpleNamespace(batch_correction=bc,
        project=SimpleNamespace(cluster_column="target", meta_path=meta_path)),
        opts=SimpleNamespace(input_mode="batch_correction"), out_dir=tmp_path,
        cache={"full_expr": X}, artifacts={})
    fit_reference(ctx)
    expected = make(kind, X).fit_transform(X[train], meta.loc[train])
    pd.testing.assert_frame_equal(ctx.cache["X_corrected"], expected)
    assert ctx.cache["X_corrected"].columns.tolist() == train
    path = tmp_path / "preprocessing/reference.npz"
    before = path.read_bytes()
    meta_path.unlink()  # Prediction cannot read even training or external metadata.
    ctx.cfg.project.meta_path = "DOES_NOT_EXIST"
    transform_reference(ctx)
    assert ctx.cache["X_recorrected"].columns.tolist() == X.columns[90:].tolist()
    assert before == path.read_bytes()
    assert not json.loads((tmp_path / "preprocessing_external/provenance.json").read_text())["external_labels_accessed"]


def test_strict_loader(tmp_path):
    path = tmp_path / "expression.csv"
    for contents in [",a,a\ng,1,2\n", ",a\ng,1\ng,2\n", ",a\ng,\n"]:
        path.write_text(contents)
        with pytest.raises(ValueError):
            read_expression(path)
