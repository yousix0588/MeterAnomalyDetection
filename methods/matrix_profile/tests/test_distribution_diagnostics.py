from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared/scripts"))
import analyze_matrix_profile_distribution as diagnostics
import numpy as np
import pandas as pd
from anomalies.multiscale.scoring import same_slot_percentiles


def test_vectorized_attribution_matches_production_ecdf():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(8,288))
    reference[0,3] = np.nan
    reference[:,4] = 0
    target = rng.normal(size=288)
    target[4] = 1e-13
    target[5] = np.nan
    expected = same_slot_percentiles(
        {date(2026,7,i+1):pd.Series(values) for i,values in enumerate(reference)}, pd.Series(target))/100
    np.testing.assert_allclose(diagnostics.vector_same_slot(reference,target),expected,equal_nan=True)


def test_reporting_merge_preserves_original_ids_and_opposite_changes():
    rows = []
    for identity,start,end,power in [
        ("a","2026-09-01 23:00","2026-09-02 00:00",3),
        ("b","2026-09-02 00:00","2026-09-02 01:00",4),
        ("c","2026-09-02 00:30","2026-09-02 01:30",1)]:
        rows.append(dict(series_id="TEST_0",event_id=identity,event_type="shape_change",confidence="medium",
                         start=pd.Timestamp(start,tz="Australia/Sydney"),end=pd.Timestamp(end,tz="Australia/Sydney"),
                         observed_mean_kw=power,baseline_median_kw=2))
    result = diagnostics.merge_reporting_units(pd.DataFrame(rows),0)
    assert len(result)==2
    assert result.original_event_count.sum()==3
    assert set(result.direction)=={"up","down"}


def test_threshold_sequences_do_not_bridge_missing_data_by_default():
    frame = pd.DataFrame(dict(series_id=["A"]*4+["B"],start=pd.to_datetime([
        "2026-09-01 00:00","2026-09-01 00:30","2026-09-01 01:30","2026-09-03 00:00","2026-09-01 00:00"])))
    mask = pd.Series([True]*5)
    assert diagnostics.sequence_event_count(frame,mask,0)==4
    assert diagnostics.sequence_event_count(frame,mask,30)==3
