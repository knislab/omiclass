from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path
import sys

import pandas as pd
import pytest

from omiclass.utils.io import load_matrix, load_meta_matrix

APOLLO_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "data" / "apolo_brca1"
sys.path.insert(0, str(APOLLO_SCRIPT_DIR))

from prepare_external_holdout import parse_star_counts, prepare  # noqa: E402


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _archive(path: Path, tables: dict[str, tuple[list[str], list[dict[str, str]]]]) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        for name, (fields, rows) in tables.items():
            text = io.StringIO()
            writer = csv.DictWriter(text, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            content = text.getvalue().encode()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))


def _star(path: Path, bad_count: str = "7") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# gene-model: test\n"
        "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\n"
        "N_unmapped\t\t\t10\t10\t10\n"
        "ENSG000001.5\tA\tprotein_coding\t3\t1\t2\n"
        f"ENSG000002.9\tB\tprotein_coding\t{bad_count}\t3\t4\n",
        encoding="utf-8",
    )


def test_star_parser_uses_unstranded_and_strips_versions(tmp_path: Path) -> None:
    path = tmp_path / "counts.tsv"
    _star(path)
    counts = parse_star_counts(path, "AP-ONE-AA")
    assert counts.to_dict() == {"ENSG000001": 3, "ENSG000002": 7}
    assert counts.name == "AP-ONE-AA"

    _star(path, bad_count="-1")
    with pytest.raises(ValueError, match="invalid unstranded raw counts"):
        parse_star_counts(path, "AP-ONE-AA")


def test_complete_external_holdout_merge(tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.tsv"
    metadata_json = tmp_path / "metadata.json"
    clinical_archive = tmp_path / "clinical.tar.gz"
    biospecimen_archive = tmp_path / "biospecimen.tar.gz"
    published_clinical = tmp_path / "published_clinical.xlsx"
    downloads = tmp_path / "downloads"
    tcga_counts = tmp_path / "tcga_counts.csv"
    tcga_metadata = tmp_path / "tcga_metadata.csv"
    output = tmp_path / "prepared"

    sheet_row = {
        "File ID": "file-1", "File Name": "sample.star_gene_counts.tsv",
        "Data Type": "Gene Expression Quantification", "Case ID": "AP-ONE",
        "Sample ID": "AP-ONE-AA", "Tissue Type": "Tumor",
    }
    _write_tsv(sheet, list(sheet_row), [sheet_row])
    metadata_json.write_text(
        json.dumps(
            [{
                "file_id": "file-1", "file_name": sheet_row["File Name"],
                "data_type": "Gene Expression Quantification", "access": "open",
                "state": "released", "analysis": {"workflow_type": "STAR - Counts"},
                "associated_entities": [{"entity_submitter_id": "AP-ONE-AA_aliquot"}],
            }]
        ),
        encoding="utf-8",
    )
    clinical_fields = [
        "cases.submitter_id", "diagnoses.age_at_diagnosis",
        "demographic.sex_at_birth", "demographic.race", "diagnoses.primary_diagnosis",
    ]
    _archive(
        clinical_archive,
        {
            "clinical.tsv": (clinical_fields, [{
                "cases.submitter_id": "AP-ONE", "diagnoses.age_at_diagnosis": "55",
                "demographic.sex_at_birth": "female", "demographic.race": "white",
                "diagnoses.primary_diagnosis": "Breast carcinoma",
            }]),
            "follow_up.tsv": (["cases.submitter_id"], []),
        },
    )
    pd.DataFrame(
        [{
            "APOLLO ID TUMOR": "AP-ONE-AA", "DX AGE": 55, "GENDER": "Female",
            "RACE": "White", "AJCC STAGE": "IIB", "PAM50": "Basal",
        }]
    ).to_excel(
        published_clinical,
        sheet_name="S1A. APOLLO_ClinicalData",
        index=False,
        startrow=5,
    )
    sample_row = {
        "cases.submitter_id": "AP-ONE", "samples.submitter_id": "AP-ONE-AA",
        "samples.tissue_type": "Tumor",
    }
    linked_row = {"samples.submitter_id": "AP-ONE-AA"}
    aliquot_row = {**linked_row, "aliquots.submitter_id": "AP-ONE-AA_aliquot"}
    _archive(
        biospecimen_archive,
        {
            "sample.tsv": (list(sample_row), [sample_row]),
            "portion.tsv": (list(linked_row), [linked_row]),
            "slide.tsv": (["samples.submitter_id"], []),
            "analyte.tsv": (list(linked_row), [linked_row]),
            "aliquot.tsv": (list(aliquot_row), [aliquot_row]),
        },
    )
    _star(downloads / "file-1" / sheet_row["File Name"])
    pd.DataFrame(
        {"TCGA-ONE": [2, 4], "TCGA-EXTRA": [8, 10]},
        index=["ENSG000002", "ENSG000003"],
    ).to_csv(tcga_counts)
    pd.DataFrame(
        [{
            "sample": "TCGA-ONE", "cohort": "PLATE-1", "age": "60",
            "sex": "female", "race": "white", "stage": "Stage I",
            "classification": "LumA",
        }]
    ).to_csv(tcga_metadata, index=False)
    args = argparse.Namespace(
        sample_sheet=sheet, metadata_json=metadata_json,
        clinical_archive=clinical_archive, biospecimen_archive=biospecimen_archive,
        published_clinical=published_clinical,
        download_dir=downloads, tcga_counts=tcga_counts, tcga_metadata=tcga_metadata,
        output_dir=output, expected_apollo_samples=1,
    )

    qc = prepare(args)
    assert qc["combined_samples"] == 2
    assert qc["shared_genes"] == 1
    assert qc["tcga_count_samples_excluded"] == 1
    combined_counts = pd.read_csv(output / "combined_counts.csv", index_col=0)
    assert combined_counts.index.tolist() == ["ENSG000002"]
    assert combined_counts.columns.tolist() == ["TCGA-ONE", "AP-ONE-AA"]
    combined_meta = pd.read_csv(output / "combined_metadata.csv")
    assert combined_meta["classification"].tolist() == ["LumA", "Basal"]
    assert combined_meta["stage"].tolist() == ["Stage I", "Stage IIB"]
    assert combined_meta["cohort"].tolist() == ["TCGA", "APOLLO"]
    comparison_meta = pd.read_csv(output / "combined_metadata_complete_design.csv")
    assert comparison_meta["sample"].tolist() == ["TCGA-ONE", "AP-ONE-AA"]
    full_meta = pd.read_csv(output / "apollo_metadata_full.tsv", sep="\t")
    assert "classification" in full_meta.columns
    assert full_meta["classification"].tolist() == ["Basal"]
    assert full_meta["published_clinical.PAM50"].tolist() == ["Basal"]
    assert full_meta["classification_source_doi"].tolist() == ["10.1038/s41523-025-00793-0"]
    assert (output / "organized_metadata" / "clinical" / "follow_up.tsv").exists()
    assert load_matrix(output / "combined_counts.csv").shape == (1, 2)
    assert load_meta_matrix(output / "combined_metadata.csv").shape == (2, 6)
    assert (output / "tcga_count_samples_excluded.txt").read_text().strip() == "TCGA-EXTRA"
