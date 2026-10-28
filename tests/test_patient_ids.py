"""Patient identity across repeated recordings and missing source metadata."""

import pandas as pd
import pytest

from src.datasets.native import metadata, patient_ids, patient_metadata_path


@pytest.mark.parametrize("cohort", ["SHDB", "IRIDIA"])
def test_metadata_groups_recordings_by_patient(tmp_path, cohort):
    source = tmp_path / "source"
    source.mkdir()
    manifests = tmp_path / "windows"
    manifests.mkdir()
    record_ids = ["001", "002", "002"] if cohort == "SHDB" else ["record_000", "record_001", "record_001"]
    rows = pd.DataFrame({
        "global_index": [0, 1, 2], "record_id": record_ids,
        "start_sample": [0, 0, 2000], "end_sample": [2000, 2000, 4000],
        "orig_fs": 200, "AFIB": [0, 1, 0], "AFL": 0, "file_index": [0, 0, 1],
    })
    rows.to_parquet(manifests / "windows.parquet", index=False)
    mapping = source / ("AdditionalData.csv" if cohort == "SHDB" else "iridia-af-metadata-v1.0.1.csv")
    record_col, patient_col = ("Data_ID", "Subject_ID") if cohort == "SHDB" else ("record_id", "patient_id")
    # SHDB mappings can omit zero padding; both recordings belong to one patient.
    keys = ["1", "2"] if cohort == "SHDB" else record_ids[:2]
    pd.DataFrame({record_col: keys, patient_col: ["0007", "0007"]}).to_csv(mapping, index=False)
    config = {"datasets": {cohort: {"source": str(source), "metadata": str(manifests)}}}
    result = metadata(cohort, config)
    assert result.patient_id.tolist() == ["0007"] * 3
    assert result.record_key.nunique() == (2 if cohort == "SHDB" else 3)


@pytest.mark.parametrize("contents,error", [
    ("Data_ID,Subject_ID\n001,p1\n001,p2\n", "duplicate"),
    ("Data_ID,Subject_ID\n001,\n", "Invalid"),
    ("Data_ID,Subject_ID\n   ,p1\n", "Invalid"),
    ("Data_ID,Subject_ID\n002,p2\n", "Missing SHDB patient IDs"),
])
def test_rejects_invalid_patient_mapping(tmp_path, contents, error):
    path = tmp_path / "patients.csv"
    path.write_text(contents)
    config = {"datasets": {"SHDB": {"patient_metadata": str(path)}}}
    with pytest.raises(ValueError, match=error):
        patient_ids("SHDB", config, pd.Series(["001"]))


def test_missing_patient_mapping_does_not_fall_back_to_recording_id(tmp_path):
    config = {"datasets": {"SHDB": {"source": str(tmp_path)}}}
    with pytest.raises(FileNotFoundError, match="patient mapping"):
        patient_ids("SHDB", config, pd.Series(["001"]))


def test_iridia_default_download_layout(tmp_path):
    source = tmp_path / "records/iridia-af-records-v1.0.1"
    source.mkdir(parents=True)
    path = tmp_path / "iridia-af-metadata-v1.0.1.csv"
    path.write_text("record_id,patient_id\nrecord_000,p1\n")
    config = {"datasets": {"IRIDIA": {"source": str(source)}}}
    assert patient_metadata_path("IRIDIA", config) == path


def test_cpsc_patient_ids_are_null():
    result = patient_ids("CPSC", {}, pd.Series(["data_1_1", "data_1_2"]))
    assert result.isna().all()
    assert str(result.dtype) == "string"
