from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

APOLLO_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "data" / "apolo_brca1"
sys.path.insert(0, str(APOLLO_SCRIPT_DIR))

from download_expression import (  # noqa: E402
    main,
    resolve_input_path,
    resolve_output_path,
    select_expression_records,
    validate_downloads,
    write_filtered_manifest,
)


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_filter_excludes_non_expression_and_controlled_files(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    sheet = tmp_path / "sheet.tsv"
    metadata = tmp_path / "metadata.json"
    _write_tsv(
        manifest,
        ["id", "filename", "md5", "size", "state"],
        [
            {"id": "counts", "filename": "counts.tsv", "md5": "a", "size": "1", "state": "released"},
            {"id": "bam", "filename": "reads.bam", "md5": "b", "size": "2", "state": "released"},
            {"id": "controlled-counts", "filename": "other.tsv", "md5": "c", "size": "3", "state": "released"},
        ],
    )
    _write_tsv(
        sheet,
        ["File ID", "File Name", "Data Type", "Sample ID"],
        [
            {"File ID": "counts", "File Name": "counts.tsv", "Data Type": "Gene Expression Quantification", "Sample ID": "AP-ONE-AA"},
            {"File ID": "bam", "File Name": "reads.bam", "Data Type": "Aligned Reads", "Sample ID": "AP-ONE-AA"},
            {"File ID": "controlled-counts", "File Name": "other.tsv", "Data Type": "Gene Expression Quantification", "Sample ID": "AP-TWO-BB"},
        ],
    )
    metadata.write_text(
        json.dumps(
            [
                {"file_id": "counts", "data_type": "Gene Expression Quantification", "access": "open", "state": "released"},
                {"file_id": "bam", "data_type": "Aligned Reads", "access": "controlled", "state": "released"},
                {"file_id": "controlled-counts", "data_type": "Gene Expression Quantification", "access": "controlled", "state": "released"},
            ]
        ),
        encoding="utf-8",
    )

    records = select_expression_records(manifest, sheet, metadata, expected_count=1)
    assert [record["id"] for record in records] == ["counts"]
    filtered = tmp_path / "filtered.tsv"
    write_filtered_manifest(records, filtered)
    assert "reads.bam" not in filtered.read_text(encoding="utf-8")
    assert "other.tsv" not in filtered.read_text(encoding="utf-8")


def test_repo_relative_paths_resolve_when_running_inside_apollo_directory(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(APOLLO_SCRIPT_DIR)
    manifest = resolve_input_path(Path("data/apolo_brca1/gdc_manifest.2026-08-31.152657.txt"))
    output = resolve_output_path(Path("data/apolo_brca1/download"))
    assert manifest == APOLLO_SCRIPT_DIR / "gdc_manifest.2026-08-31.152657.txt"
    assert output == APOLLO_SCRIPT_DIR / "download"


def test_validation_reports_ok_missing_and_checksum_mismatch(tmp_path: Path) -> None:
    download_dir = tmp_path / "downloads"
    good_content = b"good"
    bad_content = b"bad"
    records = [
        {
            "id": "good-id", "filename": "good.tsv", "size": str(len(good_content)),
            "md5": hashlib.md5(good_content).hexdigest(), "sample_id": "AP-ONE-AA",
        },
        {
            "id": "bad-id", "filename": "bad.tsv", "size": str(len(bad_content)),
            "md5": hashlib.md5(b"different").hexdigest(), "sample_id": "AP-TWO-BB",
        },
        {
            "id": "missing-id", "filename": "missing.tsv", "size": "1",
            "md5": "0" * 32, "sample_id": "AP-THREE-CC",
        },
    ]
    for record, content in zip(records[:2], [good_content, bad_content]):
        path = download_dir / record["id"] / record["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(content)

    report = validate_downloads(records, download_dir)
    assert [row["status"] for row in report] == ["ok", "md5_mismatch", "missing"]


def test_main_resumes_valid_download_and_invokes_client_for_missing_file(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    sheet = tmp_path / "sheet.tsv"
    metadata = tmp_path / "metadata.json"
    output = tmp_path / "output"
    content = b"gene counts"
    digest = hashlib.md5(content).hexdigest()
    _write_tsv(
        manifest,
        ["id", "filename", "md5", "size", "state"],
        [{"id": "file-1", "filename": "counts.tsv", "md5": digest, "size": str(len(content)), "state": "released"}],
    )
    _write_tsv(
        sheet,
        ["File ID", "File Name", "Data Type", "Sample ID"],
        [{"File ID": "file-1", "File Name": "counts.tsv", "Data Type": "Gene Expression Quantification", "Sample ID": "AP-ONE-AA"}],
    )
    metadata.write_text(
        json.dumps([{"file_id": "file-1", "data_type": "Gene Expression Quantification", "access": "open", "state": "released"}]),
        encoding="utf-8",
    )
    args = [
        "--manifest", str(manifest), "--sample-sheet", str(sheet),
        "--metadata", str(metadata), "--output-dir", str(output),
        "--expected-count", "1",
    ]
    downloaded = output / "raw_expression" / "file-1" / "counts.tsv"
    downloaded.parent.mkdir(parents=True)
    downloaded.write_bytes(content)
    with patch("download_expression.shutil.which", return_value="/fake/gdc-client"), patch(
        "download_expression.subprocess.run"
    ) as run:
        assert main(args) == 0
        run.assert_not_called()

    downloaded.unlink()

    def fake_download(*_args, **_kwargs):
        downloaded.write_bytes(content)

    with patch("download_expression.shutil.which", return_value="/fake/gdc-client"), patch(
        "download_expression.subprocess.run", side_effect=fake_download
    ) as run:
        assert main(args) == 0
        run.assert_called_once()
