"""Patient-time aggregation and saved operating-point selection for the figure."""

import json

import numpy as np
import pandas as pd
import pytest

from fig.scripts import patient_burden as burden


def predictions(patients, labels, probabilities, *, first_index=0, statuses=None):
    statuses = np.zeros(len(labels), dtype=int) if statuses is None else np.array(statuses)
    probabilities = np.asarray(probabilities)
    return pd.DataFrame({
        "global_index": np.arange(first_index, first_index + len(labels)),
        "patient_id": pd.Series(patients, dtype="string"),
        "label": labels, "eligible": True,
        burden.STATUS: statuses,
        burden.PROBABILITY: probabilities,
        burden.CLASS: np.where(statuses == 0, (probabilities >= 0.4).astype(int), -1),
    })


def write_run(folder, frames):
    manifest = {"models": [burden.MODEL], "implementations": ["python"],
                "cohorts": list(burden.COHORTS), "max_per_record": 0,
                "thresholds": {burden.MODEL: 0.4}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    for cohort in burden.COHORTS:
        directory = folder / cohort
        directory.mkdir()
        for i, frame in enumerate(frames):
            frame.to_parquet(directory / f"{i:05d}.parquet", index=False)


def test_combines_recordings_by_patient_and_weights_by_time(tmp_path):
    short_recording = predictions(["0007"], [1], [0.9])
    long_recording = predictions(["0007"] * 9 + ["other"], [0] * 10, [0.1] * 10,
                                 first_index=1)
    write_run(tmp_path, [short_recording, long_recording])
    results, threshold = burden.load_burdens(tmp_path)
    assert threshold == 0.4
    for table in results.values():
        assert list(table.index) == ["0007", "other"]
        patient = table.loc["0007"]
        assert patient.recorded_seconds == 100
        assert patient.true_burden_pct == patient.predicted_burden_pct == 10
        assert patient.scored_coverage_pct == 100


@pytest.mark.parametrize("denominator,true_pct,predicted_pct", [
    (None, 50, 50),
    ("recorded", 200 / 3, 100 / 3),
    ("scored", 50, 50),
])
def test_failure_policy_uses_explicit_shared_denominator(tmp_path, denominator,
                                                       true_pct, predicted_pct):
    frame = predictions(["p1"] * 3, [1, 1, 0], [0.9, np.nan, 0.2], statuses=[0, -4, 0])
    write_run(tmp_path, [frame])
    results, _ = (burden.load_burdens(tmp_path) if denominator is None
                  else burden.load_burdens(tmp_path, denominator))
    patient = results["SHDB"].loc["p1"]
    assert patient.true_burden_pct == pytest.approx(true_pct)
    assert patient.predicted_burden_pct == pytest.approx(predicted_pct)
    assert patient.recorded_seconds == 30
    assert patient.scored_seconds == 20


def test_uses_window_classes_at_saved_threshold_not_mean_probability(tmp_path):
    frame = predictions(["p1", "p1"], [1, 0], [0.45, 0.30])
    write_run(tmp_path, [frame])
    results, _ = burden.load_burdens(tmp_path)
    assert results["SHDB"].loc["p1", "predicted_burden_pct"] == 50
    frame.loc[0, burden.CLASS] = 0
    with pytest.raises(ValueError, match="threshold"):
        burden.patient_times(frame, 0.4, "recorded")


def test_rejects_missing_patient_ids_and_duplicate_windows(tmp_path):
    frame = predictions(["p1"], [0], [0.1])
    write_run(tmp_path, [frame, frame])
    with pytest.raises(ValueError, match="Duplicate SHDB"):
        burden.load_burdens(tmp_path)
    frame.loc[0, "patient_id"] = pd.NA
    with pytest.raises(ValueError, match="missing patient IDs"):
        burden.patient_times(frame, 0.4, "recorded")


def test_patient_errors_are_equally_weighted():
    table = pd.DataFrame({"true_burden_pct": [0, 100], "predicted_burden_pct": [0, 80],
                          "analyzed_seconds": [100000, 10]})
    assert burden.error_summary(table)["mae"] == 10
