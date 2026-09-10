from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from final_results_runs.plots.plot_repeated_seeds import (
    discover_runs,
    scaled_dominance_matrix,
)


NOTEBOOK_PATH = (
    Path(__file__).parents[1]
    / "final_results_runs"
    / "plots"
    / "plot_repeated_seeds.ipynb"
)


def _notebook_code() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _task_output_mapping() -> dict[str, tuple[str, str | None]]:
    tree = ast.parse(_notebook_code())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "TASKS" for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Dict)
    mapping: dict[str, tuple[str, str | None]] = {}
    for task_key, task_value in zip(assignment.value.keys, assignment.value.values):
        assert isinstance(task_value, ast.Dict)
        fields = {
            ast.literal_eval(key): ast.literal_eval(value)
            for key, value in zip(task_value.keys, task_value.values)
            if isinstance(key, ast.Constant)
            and key.value in {"output_task", "output_condition"}
        }
        mapping[ast.literal_eval(task_key)] = (
            fields["output_task"],
            fields["output_condition"],
        )
    return mapping


def _write_run(run_dir: Path, *, seed: int, selected_model: str) -> None:
    manifest_dir = run_dir / "pdf_plots" / "base_files"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"classification": {"selected_model": selected_model}}),
        encoding="utf-8",
    )
    (run_dir / "config.freeze.yaml").write_text(
        f"random_state: {seed}\n", encoding="utf-8"
    )


def test_discover_runs_supports_flat_seed_run_directories(tmp_path: Path) -> None:
    run_dir = tmp_path / "Omiclass_IGHV_corrected_seed11"
    _write_run(run_dir, seed=11, selected_model="plain_random_forest")

    runs, missing = discover_runs(tmp_path)

    assert missing == []
    assert [(run.seed, run.path) for run in runs] == [("11", run_dir)]


def test_discover_runs_uses_latest_timestamped_run_in_each_seed_dir(
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "seed_11"
    older = seed_dir / "Omiclass_BRCA_no_correction-20260901-120000"
    latest = seed_dir / "Omiclass_BRCA_no_correction-20260902-090000"
    _write_run(older, seed=11, selected_model="plain_random_forest")
    _write_run(latest, seed=11, selected_model="weighted_random_forest")

    runs, missing = discover_runs(tmp_path)

    assert missing == []
    assert len(runs) == 1
    assert runs[0].seed == "11"
    assert runs[0].path == latest
    assert runs[0].selected_model == "weighted_random_forest"


def test_scaled_dominance_matrix_has_unit_diagonal_and_is_bounded() -> None:
    matrix = np.array([[4, 0, 0], [0, 3, 0], [0, 0, 2]])

    dominance = scaled_dominance_matrix(matrix)

    np.testing.assert_allclose(np.diag(dominance), 1.0)
    assert np.all(dominance >= 0.0)
    assert np.all(dominance <= 1.0)


def test_notebook_maps_runs_to_five_task_directories() -> None:
    mapping = _task_output_mapping()

    assert {task for task, _ in mapping.values()} == {
        "b-all",
        "brca-protein",
        "brca-tcga",
        "brca-tcga-apollo",
        "cll",
    }
    assert mapping["b_all_old_de"] == ("b-all", None)
    assert mapping["brca_proteomics_old_de"] == ("brca-protein", None)
    assert mapping["brca_corrected_old_test"] == ("brca-tcga", "corrected")
    assert mapping["brca_no_correction_old_test"] == ("brca-tcga", "uncorrected")
    assert mapping["brca_corrected_new_test"] == (
        "brca-tcga-apollo",
        "corrected",
    )
    assert mapping["brca_no_correction_new_test"] == (
        "brca-tcga-apollo",
        "uncorrected",
    )
    assert mapping["cll_corrected"] == ("cll", "corrected")
    assert mapping["cll_uncorrected"] == ("cll", "uncorrected")


def test_notebook_routes_shared_outputs_to_task_roots() -> None:
    source = _notebook_code()

    assert 'output_dir = task_root_dir(spec["task_name"])' in source
    assert 'DKFZ_SUMMARY_DIR = task_root_dir("cll_corrected") / "external-dkfz"' in source
    assert 'dkfz_comparison_dir = task_root_dir("cll_corrected") / "external-dkfz"' in source
    assert 'comparison_root = OVERVIEW_DIR / "correction_comparisons"' not in source
