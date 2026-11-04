from datetime import timedelta

import pandas as pd
import pytest

from eris_ai.gladden_cohort import (
    FEATURE_COLUMNS, HORIZON, build_cohort, outcome, snapshot_features, temporal_split,
)


@pytest.mark.parametrize("kind,state", [
    ("Resignation", "POSITIVE"), ("Termination", "COMPETING_EVENT"),
])
def test_exit_endpoints(kind, state):
    t = pd.Timestamp("2021-01-03")
    with pytest.raises(ValueError, match="already left"):
        outcome(t, t, t, kind)
    assert outcome(t, t + HORIZON, t + HORIZON, kind) == state
    assert outcome(t, t + HORIZON + timedelta(days=1),
                   t + HORIZON + timedelta(days=1), kind) == "NEGATIVE"


def test_negative_requires_full_followup_and_positive_can_be_known_earlier():
    t = pd.Timestamp("2021-01-03")
    assert outcome(t, t + HORIZON - timedelta(days=1)) == "UNKNOWN_FOLLOWUP"
    assert outcome(t, t + HORIZON) == "NEGATIVE"
    exit_date = t + timedelta(days=2)
    assert outcome(t, exit_date, exit_date, "Resignation") == "POSITIVE"


def event(person, day, kind, role="Laborer"):
    return {"sub_ID": person, "date": pd.Timestamp(day), "record_comptype": kind,
            "sub_role": role, "sub_shift": "1", "sub_team": "2"}


def test_feature_window_uses_past_only_and_missing_is_not_absence():
    history = pd.DataFrame([
        event("a", "2021-01-03", "Absence"),  # Exactly t-28: outside window.
        event("a", "2021-01-04", "Presence"),
        event("a", "2021-01-04", "Efficacy"),  # Same day counts once.
        event("a", "2021-01-05", None),         # Unrecorded event, not HR evidence.
        event("a", "2021-01-31", "Absence"),
        event("a", "2021-02-01", "Absence", "Team Leader"),
    ])
    t = pd.Timestamp("2021-01-31")
    result, latest = snapshot_features(history, t, pd.Timestamp("2021-01-03"))
    assert result["role"] == "Laborer"
    assert result["presence_days_28d"] == result["absence_days_28d"] == 1
    assert result["recorded_days_28d"] == 2
    assert result["no_record_calendar_days_28d"] == 26
    assert latest == t
    altered = history.copy()
    altered.loc[altered.date > t, "sub_role"] = "Future information"
    assert snapshot_features(altered, t, pd.Timestamp("2021-01-03")) == (result, latest)
    assert list(result) == FEATURE_COLUMNS


def test_cohort_excludes_exit_day_and_no_evidence_and_applies_symmetric_admin_cutoff():
    events = pd.DataFrame([
        event("a", "2021-01-01", "Presence"),
        event("a", "2021-01-03", "Resignation"),
        event("b", "2021-01-01", "Presence"),
        event("b", "2021-04-30", "Presence"),
        event("c", "2021-01-01", None),
        event("d", "2021-01-01", "Presence"),
        event("d", "2021-04-20", "Resignation"),
    ])
    cohort, audit = build_cohort(events, pd.Timestamp("2021-04-30"))
    assert set(cohort.sub_ID) == {"b", "d"}
    assert audit["people_without_recorded_evidence_excluded"] == 1
    assert not cohort.duplicated(["sub_ID", "snapshot_date"]).any()
    tail = cohort.loc[~cohort.administratively_eligible]
    assert "POSITIVE" in set(tail.label_status)
    assert "UNKNOWN_FOLLOWUP" in set(tail.label_status)
    eligible = cohort.loc[cohort.administratively_eligible]
    assert eligible.horizon_end.le(pd.Timestamp("2021-04-30")).all()
    early = cohort.loc[(cohort.sub_ID == "b") & (cohort.snapshot_date == "2021-01-03")].iloc[0]
    assert early.observable_calendar_days_28d == 3


def test_temporal_split_rejects_touching_label_windows():
    with pytest.raises(ValueError, match="strictly before"):
        # Validation must start AFTER (not on) the final training label end.
        temporal_split(pd.DataFrame({
            "sub_ID": ["a"], "snapshot_date": [pd.Timestamp("2021-01-03")],
            "horizon_end": [pd.Timestamp("2021-04-03")], "target": [0],
            "label_status": ["NEGATIVE"],
        }), {"train": ("2021-01-03", "2021-01-03"),
             "validation": ("2021-04-03", "2021-04-04")})
