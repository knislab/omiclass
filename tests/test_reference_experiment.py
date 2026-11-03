import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
import json
import yaml

from experiments.reference_preprocessing.run import training_tpm, diagnostics, read_training_counts, prepare, summarize, classifier_features, predict_external, evaluate


def test_schema_and_normalization_are_training_only():
    counts = pd.DataFrame([[10., 20., 999.], [5., 0., 0.], [0., 0., 9000.]],
                          index=["a", "b", "c"], columns=["train1", "train2", "external"])
    lengths = pd.Series([1000., 2000., 500.], index=counts.index)
    result, zeros = training_tpm(counts, lengths, ["train1", "train2"])
    assert result.index.tolist() == ["a", "b"] and zeros == ["c"]
    assert result.columns.tolist() == ["train1", "train2"]
    np.testing.assert_allclose(result.sum(axis=0), 1e6)
    changed = counts.copy()
    changed.external = 1e30
    same, same_zeros = training_tpm(changed, lengths, ["train1", "train2"])
    pd.testing.assert_frame_equal(result, same)
    assert same_zeros == zeros
    with pytest.raises(ValueError, match="Invalid training counts"):
        training_tpm(-counts, lengths, ["train1", "train2"])


def test_training_diagnostics_do_not_use_external_expression():
    X = pd.DataFrame([[1., 2., 3., 4., 5., 6.]], index=["g"], columns=list("abcdef"))
    meta = pd.DataFrame({"cohort": ["one"] * 3 + ["two"] * 3,
                         "ighv_mut": ["m", "u"] * 3}, index=X.columns)
    values = diagnostics(X, X.copy(), meta)
    assert values["before"] == values["after"]


def test_original_duplicate_sample_policy_is_explicitly_audited(tmp_path):
    path = tmp_path / "counts.csv"
    path.write_text(",patient,patient,second\ng1,2,7,3\ng2,4,8,5\n")
    counts, audit = read_training_counts(path)
    assert counts.columns.tolist() == ["patient", "second"]
    np.testing.assert_array_equal(counts.patient, [2, 4])
    assert audit == [dict(sample="patient", kept_column_position=0,
                          excluded_column_position=1, identical_values=False)]


def test_prepare_uses_training_schema_and_ignores_external_outcomes(tmp_path):
    ids = ["a", "b", "c", "d"]
    counts = pd.DataFrame([[10., 20., 15., 30.], [2., 1., 4., 3.], [0., 0., 0., 0.]],
                          index=["g1", "g2", "training_zero"], columns=ids)
    counts_path = tmp_path / "counts.csv"
    counts.to_csv(counts_path)
    lengths = pd.Series([1000., 2000., 1000.], index=counts.index, name="length")
    length_path = tmp_path / "lengths.csv"
    lengths.to_frame().to_csv(length_path)
    tpm, _ = training_tpm(counts, lengths, ids)
    tpm["DKFZ1"] = [200., 500.]
    expression = tmp_path / "expression.csv"
    tpm.to_csv(expression)
    meta = pd.DataFrame(dict(cohort=["one", "two", "one", "two", "DKFZ"],
        age=[40, 50, 60, 70, 80], sex=[0, 1, 0, 1, 0],
        ighv_mut=["mutated", "unmutated", "mutated", "unmutated", "MUST_NOT_BE_USED"]),
        index=ids + ["DKFZ1"])
    meta_path, training_meta = tmp_path / "meta.csv", tmp_path / "train_meta.csv"
    meta.to_csv(meta_path)
    meta.loc[ids].to_csv(training_meta)
    source = tmp_path / "source"
    for seed in [11, 61, 73, 80, 89]:
        root = source / f"Omiclass_IGHV_corrected_seed{seed}"
        (root / "de_results").mkdir(parents=True)
        (root / "split.json").write_text(json.dumps(dict(train_index=ids[:2], test_index=ids[2:])))
        (root / "config.freeze.yaml").write_text(yaml.safe_dump(dict(project=dict(
            cluster_column="ighv_mut", meta_path=str(training_meta)))))
        (root / "de_results/limma_voom_selected_genes.txt").write_text("g1\ng2\n")
    args = SimpleNamespace(out=tmp_path / "first", expression=expression, metadata=meta_path,
        source_root=source, training_counts=counts_path, gene_lengths=length_path, input_scale="tpm")
    prepare(args)
    first = pd.read_csv(args.out / "training.csv", index_col=0)
    schema = json.loads((args.out / "gene_schema.json").read_text())
    assert schema["selected_genes"] == ["g1", "g2"]
    assert schema["excluded_training_zero_genes"] == ["training_zero"]
    assert "ighv_mut" not in pd.read_csv(args.out / "correction_covariates.csv").columns
    tpm.DKFZ1 = 1e30
    tpm.to_csv(expression)
    meta.loc["DKFZ1", "ighv_mut"] = "CHANGED_EXTERNAL_OUTCOME"
    meta.to_csv(meta_path)
    args.out = tmp_path / "second"
    prepare(args)
    pd.testing.assert_frame_equal(first, pd.read_csv(args.out / "training.csv", index_col=0))
    assert schema == json.loads((args.out / "gene_schema.json").read_text())


def test_summary_uses_cv_selected_model_holdout_and_external_results(tmp_path):
    root = tmp_path / "runs/frozen_quantile/seed11"
    root.mkdir(parents=True)
    (root / "complete.json").write_text(json.dumps(dict(method="frozen_quantile", seed=11,
        metrics=dict(accuracy=.8, f1_macro=.75), correction_fit_uses_external_expression=False)))
    (root / "selection.json").write_text(json.dumps(dict(model=str(root / "selected.joblib"))))
    pd.DataFrame([dict(model="selected", accuracy=.9, f1_macro=.89),
                  dict(model="other", accuracy=1., f1_macro=1.)]).to_csv(root / "test_metrics.csv", index=False)
    summarize(SimpleNamespace(runs_root=tmp_path / "runs", out=tmp_path / "summary"))
    row = pd.read_csv(tmp_path / "summary/per_seed.csv").iloc[0]
    assert row.model == "selected" and row.internal_accuracy == .9 and row.external_accuracy == .8


class StackingSchemaProbe:
    _feature_names_in = np.array(["g2", "g1"])

    @property
    def feature_names_in_(self):
        raise AttributeError("Stacking wrapper exposes only its private saved schema")

    def predict(self, X):
        assert X.columns.tolist() == ["g2", "g1"]
        return np.array([0, 1])

    def fit(self, *args, **kwargs):
        raise AssertionError("Evaluation-only recovery must never fit")


def test_stacking_private_schema_and_public_schema():
    assert classifier_features(StackingSchemaProbe()) == ["g2", "g1"]
    assert classifier_features(SimpleNamespace(feature_names_in_=["g1", "g2"])) == ["g1", "g2"]
    for model in [SimpleNamespace(), SimpleNamespace(_feature_names_in=["g1", "g1"]),
                  SimpleNamespace(_feature_names_in=[]), SimpleNamespace(_feature_names_in=[["g1"]])]:
        with pytest.raises(ValueError, match="schema"):
            classifier_features(model)


def test_saved_stacking_prediction_orders_genes_before_reading_outcomes(tmp_path, monkeypatch):
    import joblib
    monkeypatch.setattr(joblib, "load", lambda path: StackingSchemaProbe())
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"ighv_mut": ["mutated", "unmutated"]}, index=["e1", "e2"]).to_csv(metadata)
    read_csv = pd.read_csv
    out = tmp_path / "evaluation"
    out.mkdir()
    def guarded_read(path, *args, **kwargs):
        if path == metadata:
            assert (out / "external_predictions.csv").exists()
        return read_csv(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", guarded_read)
    external = pd.DataFrame([[1., 2.], [3., 4.]], index=["g1", "g2"], columns=["e1", "e2"])
    metrics = predict_external("unused.joblib", external, ["mutated", "unmutated"], metadata, out)
    assert metrics == dict(accuracy=1., f1_macro=1., n_scored=2)
    with pytest.raises(ValueError, match="classifier features"):
        predict_external("unused.joblib", external.iloc[:1], ["mutated", "unmutated"], metadata, out)


def test_evaluation_only_recovery_preserves_source_without_training(tmp_path, monkeypatch):
    import experiments.reference_preprocessing.run as runner
    import joblib
    monkeypatch.setattr(joblib, "load", lambda path: StackingSchemaProbe())
    source = tmp_path / "seed61"
    (source / "train_artifacts").mkdir(parents=True)
    model = source / "train_artifacts/stacked_ensemble.joblib"
    model.write_text("mock saved model; fitting is forbidden")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}")
    (source / "selection.json").write_text(json.dumps(dict(method="fsva_exact", seed=61, model=str(model))))
    config = dict(project=dict(classes_names=["mutated", "unmutated"]), runtime=dict(random_state=61),
        batch_correction=dict(algorithm="fsva_exact", n_factors=1,
            training_samples_path=str(prepared / "training_ids.txt"),
            reference_path=str(tmp_path / "shared/preprocessing/reference.npz")))
    (source / "config.freeze.yaml").write_text(yaml.safe_dump(config))
    (source / "test_metrics.csv").write_text("model,accuracy,f1_macro\nstacked_ensemble,0.9,0.89\n")
    (source / "failure.json").write_text('{"error":"previous schema failure"}')
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    external = pd.DataFrame([[1., 2.], [3., 4.]], index=["g1", "g2"], columns=["e1", "e2"])
    monkeypatch.setattr(runner, "check_inputs", lambda path: dict(training_count=625))
    monkeypatch.setattr(runner, "cached_preprocessing", lambda args: (None, external, {}))
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"ighv_mut": ["mutated", "unmutated"]}, index=["e1", "e2"]).to_csv(metadata)
    args = SimpleNamespace(run_dir=source, out=tmp_path / "retry", prepared=None,
                           preprocessing_cache=None, external_metadata=metadata)
    evaluate(args)
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    state = json.loads((args.out / "complete.json").read_text())
    assert state["metrics"]["accuracy"] == 1.
    assert not state["classifier_refitted"] and not state["preprocessing_refitted"]
    assert args.seed == 61 and args.preprocessing_cache == tmp_path / "shared"
