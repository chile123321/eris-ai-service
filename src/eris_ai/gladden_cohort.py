"""Build weekly simulated 90-day resignation cohorts without fitting any model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import zipfile
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

import pandas as pd

EXPECTED_SHA256 = "d1474183d7f018ef33a9c8f2d2a152d3f8cc1b5518a93bc8b831b93affe83af0"
ADMIN_END = pd.Timestamp("2022-06-30")
HORIZON = timedelta(days=90)
SOURCE_URL = "https://www.kaggle.com/datasets/gladdenme/factory-workers-daily-performance-attrition-s"
SOURCE_COLUMNS = [
    "sub_ID", "event_date", "record_comptype", "sub_role", "sub_shift", "sub_team",
]
FEATURE_COLUMNS = [
    "role", "shift", "team", "presence_days_28d", "absence_days_28d",
    "recorded_days_28d", "observable_calendar_days_28d", "no_record_calendar_days_28d",
]
STATES = ["POSITIVE", "NEGATIVE", "COMPETING_EVENT", "UNKNOWN_FOLLOWUP"]
# Fixed calendar windows, selected before looking at cohort event counts or model scores.
THREE_WAY = {
    "train": ("2021-01-03", "2021-05-30"),
    "validation": ("2021-08-29", "2021-10-31"),
    "final_test": ("2022-01-30", "2022-03-27"),
}
TWO_WAY = {
    "train": ("2021-01-03", "2021-08-29"),
    "validation": ("2021-11-28", "2022-03-27"),
}
# A screening heuristic, not a power calculation or a claim of precise evaluation.
MIN_UNIQUE_POSITIVE_PEOPLE = 20


def load_events(source: Path) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(source) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Expected exactly one CSV inside archive")
        raw = archive.read(names[0])
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("CSV SHA-256 differs from audited source")
    events = pd.read_csv(io.BytesIO(raw), encoding="cp1252", usecols=SOURCE_COLUMNS,
                         dtype={"sub_ID": "string", "sub_shift": "string", "sub_team": "string"})
    events["date"] = pd.to_datetime(events.event_date, format="%m/%d/%Y", errors="raise")
    if events[["sub_ID", "date"]].isna().any().any():
        raise ValueError("Missing employee ID or event date")
    counts = {kind: int(events.loc[events.record_comptype.eq(kind), "sub_ID"].nunique())
              for kind in ("Resignation", "Termination")}
    if (len(events), events.sub_ID.nunique(), counts, events.date.max()) != (
        411948, 687, {"Resignation": 77, "Termination": 102}, ADMIN_END
    ):
        raise ValueError("Source does not match audited row/ID/exit/date counts")
    return events, {
        "source_url": SOURCE_URL, "csv_filename": names[0], "csv_sha256": digest,
        "encoding": "Windows-1252", "rows": len(events), "people": 687,
        "exit_people": counts, "date_min": str(events.date.min().date()),
        "date_max": str(events.date.max().date()),
        "missing_record_comptype": int(events.record_comptype.isna().sum()),
        "missing_id_or_date": 0,
    }


def outcome(t, followup_end, exit_date=pd.NaT, exit_type=None):
    """Return an end-of-day snapshot outcome; both event window endpoints are explicit."""
    if pd.notna(exit_date) and exit_date <= t:
        raise ValueError("Employee already left at end-of-day snapshot")
    end = t + HORIZON
    if pd.notna(exit_date) and t < exit_date <= end:
        if exit_date > followup_end:
            raise ValueError("Exit event falls beyond follow-up evidence")
        if exit_type == "Resignation":
            return "POSITIVE"
        if exit_type == "Termination":
            return "COMPETING_EVENT"
        raise ValueError("Unknown exit type")
    return "NEGATIVE" if followup_end >= end else "UNKNOWN_FOLLOWUP"


def snapshot_features(history: pd.DataFrame, t: pd.Timestamp, first_observed: pd.Timestamp):
    """Only recorded HR events with date <= t can influence X."""
    past = history.loc[history.date.le(t) & history.record_comptype.notna()].sort_values(
        "date", kind="stable"
    )
    if past.empty:
        raise ValueError("No recorded employment evidence at snapshot")
    latest = past.iloc[-1]
    window = past.loc[past.date.gt(t - timedelta(days=28))]
    recorded_days = window.date.nunique()
    observable_days = min(28, (t - first_observed).days + 1)
    values = {
        "role": latest.sub_role, "shift": latest.sub_shift, "team": latest.sub_team,
        "presence_days_28d": int(window.loc[window.record_comptype.eq("Presence"), "date"].nunique()),
        "absence_days_28d": int(window.loc[window.record_comptype.eq("Absence"), "date"].nunique()),
        "recorded_days_28d": int(recorded_days),
        "observable_calendar_days_28d": int(observable_days),
        "no_record_calendar_days_28d": int(observable_days - recorded_days),
    }
    assert list(values) == FEATURE_COLUMNS
    assert 0 <= recorded_days <= observable_days <= 28
    return values, past.date.max()


def build_cohort(events: pd.DataFrame, admin_end=ADMIN_END) -> tuple[pd.DataFrame, dict]:
    recorded = events.loc[events.record_comptype.notna()].copy()
    exits = recorded.loc[recorded.record_comptype.isin(["Resignation", "Termination"])]
    if exits.sub_ID.duplicated().any():
        raise ValueError("Multiple exit events for an employee")
    # A day-level role must not depend on the arbitrary order of events within that day.
    if recorded.groupby(["sub_ID", "date"])[["sub_role", "sub_shift", "sub_team"]].nunique().gt(1).any().any():
        raise ValueError("Conflicting assignment within a recorded day")
    weeks = pd.date_range(events.date.min(), admin_end, freq="W-SUN")
    rows = []
    survivors_at_end = 0
    exit_lookup = exits.set_index("sub_ID")
    for person, history in recorded.groupby("sub_ID", sort=True):
        history = history.sort_values("date", kind="stable")
        first, last = history.date.min(), history.date.max()
        exit_date, exit_type = pd.NaT, None
        if person in exit_lookup.index:
            exit_date = exit_lookup.loc[person, "date"]
            exit_type = exit_lookup.loc[person, "record_comptype"]
            if history.date.gt(exit_date).any():
                raise ValueError("Recorded event after exit")
        else:
            survivors_at_end += int(last == admin_end)
        for t in weeks:
            # Last observed record bounds availability; no extrapolation beyond follow-up.
            if t < first or t > last or (pd.notna(exit_date) and exit_date <= t):
                continue
            features, latest_used = snapshot_features(history, t, first)
            state = outcome(t, last, exit_date, exit_type)
            rows.append({
                "sub_ID": person, "snapshot_date": t, "horizon_end": t + HORIZON,
                "first_observed_date": first, "last_feature_record_date": latest_used,
                "followup_end": last, "exit_date": exit_date, "exit_type": exit_type,
                "label_status": state, "target": {"POSITIVE": 1, "NEGATIVE": 0}.get(state),
                "administratively_eligible": t + HORIZON <= admin_end,
                **features,
            })
    cohort = pd.DataFrame(rows)
    cohort["target"] = cohort.target.astype("Int64")
    assert not cohort.duplicated(["sub_ID", "snapshot_date"]).any()
    assert cohort.snapshot_date.dt.dayofweek.eq(6).all()
    assert cohort.last_feature_record_date.le(cohort.snapshot_date).all()
    assert cohort.exit_date.isna().where(cohort.exit_date.isna(),
                                       cohort.exit_date.gt(cohort.snapshot_date)).all()
    audit = {
        "recorded_people": int(recorded.sub_ID.nunique()),
        "people_without_recorded_evidence_excluded": int(events.sub_ID.nunique() - recorded.sub_ID.nunique()),
        "recorded_survivors_at_admin_end": survivors_at_end,
        "recorded_survivors_total": int(recorded.sub_ID.nunique() - len(exits)),
        "no_events_after_exit": True, "snapshot_keys_unique": True,
        "features_not_after_snapshot": True,
    }
    return cohort, audit


def cohort_summary(frame: pd.DataFrame) -> dict:
    positive = frame.loc[frame.label_status.eq("POSITIVE")]
    return {
        "snapshots": len(frame), "people": int(frame.sub_ID.nunique()),
        "unique_resigning_people": int(positive.sub_ID.nunique()),
        "positive_rate": float(frame.target.mean()) if frame.target.notna().any() else None,
        "snapshot_min": str(frame.snapshot_date.min().date()) if len(frame) else None,
        "snapshot_max": str(frame.snapshot_date.max().date()) if len(frame) else None,
        "statuses": {state: int(frame.label_status.eq(state).sum()) for state in STATES},
    }


def temporal_split(binary: pd.DataFrame, windows: dict) -> tuple[pd.DataFrame, dict]:
    assigned = binary[["sub_ID", "snapshot_date", "horizon_end", "target"]].copy()
    assigned["split"] = "PURGED_OR_OUTSIDE_WINDOWS"
    summaries, sets, positive_sets = {}, {}, {}
    previous_end = None
    for name, (start, end) in windows.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if previous_end is not None and previous_end + HORIZON >= start:
            raise ValueError("Previous label window must end strictly before next split")
        previous_end = end
        mask = binary.snapshot_date.between(start, end)
        assigned.loc[mask, "split"] = name
        selected = binary.loc[mask]
        summaries[name] = cohort_summary(selected)
        sets[name] = set(selected.sub_ID)
        positive_sets[name] = set(selected.loc[selected.target.eq(1), "sub_ID"])
    names = list(windows)
    overlap = {f"{a}__{b}": len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i+1:]}
    positive_overlap = {f"{a}__{b}": len(positive_sets[a] & positive_sets[b])
                        for i, a in enumerate(names) for b in names[i+1:]}
    assert not any(positive_overlap.values()), "Same resignation counted in separate periods"
    return assigned, {
        "windows": windows, "sets": summaries, "id_overlap": overlap,
        "positive_person_overlap": positive_overlap,
        "purged_or_outside_snapshots": int(assigned.split.eq("PURGED_OR_OUTSIDE_WINDOWS").sum()),
        "screening_minimum_unique_events_per_set": MIN_UNIQUE_POSITIVE_PEOPLE,
        "passes_event_screen": all(s["unique_resigning_people"] >= MIN_UNIQUE_POSITIVE_PEOPLE
                                   for s in summaries.values()),
    }


def run(source: Path, output: Path):
    if output.exists():
        raise ValueError("Output exists; do not overwrite an established cohort")
    events, source_info = load_events(source)
    cohort, audit = build_cohort(events)
    administrative = cohort.loc[cohort.administratively_eligible]
    binary = administrative.loc[administrative.label_status.isin(["POSITIVE", "NEGATIVE"])].copy()
    three_assignments, three = temporal_split(binary, THREE_WAY)
    two_assignments, two = temporal_split(binary, TWO_WAY)
    status = ("READY_FOR_TEMPORAL_BASELINE" if three["passes_event_screen"]
              else "TEMPORAL_SPLIT_INSUFFICIENT_EVENTS")
    if audit["recorded_survivors_at_admin_end"] != audit["recorded_survivors_total"]:
        status = "FOLLOWUP_REVIEW_REQUIRED"
    metadata = {
        "status": status, "data_version": EXPECTED_SHA256,
        "feature_set_version": "gladden-recorded-weekly-28d-v1",
        "source": source_info, "audit": audit, "all_snapshots": cohort_summary(cohort),
        "administrative_cohort": cohort_summary(administrative), "binary": cohort_summary(binary),
        "after_admin_cutoff": cohort_summary(cohort.loc[~cohort.administratively_eligible]),
        "three_way": three, "two_way_exploratory": two,
        "feature_allowlist": FEATURE_COLUMNS,
        "metadata_columns_outside_X": [c for c in cohort.columns if c not in FEATURE_COLUMNS],
        "source_fields_read": SOURCE_COLUMNS,
        "followup_assumption": "SIMULATION_ONLY: recorded first observation establishes entry; "
            "employment continues until recorded exit. A later record supports intervening "
            "employment, assuming the documented simulator logs all separations and no rehire. "
            "Last recorded date bounds follow-up. No independent completeness manifest exists.",
        "record_availability_assumption": "Recorded events are available at end of their event "
            "date. No ingestion timestamp exists. Missing records are not recorded absence.",
        "efficacy_excluded": "Author describes end-of-day recording, but ingestion availability "
            "is not independently timestamped; this minimal cohort omits recorded_efficacy.",
        "window": "Calendar dates (t-28d, t]; Sunday end-of-day; outcome (t,t+90d].",
        "administrative_end": "2022-06-30",
        "initial_workers": "First recorded observation is not contract start; no tenure inferred.",
        "competing_event": "Termination on or before horizon end is excluded from binary cohort.",
        "screening_rule": "20 distinct resigning people per set is a pragmatic minimum, not "
            "a power calculation. Overlapping snapshots are not independent observations.",
        "evaluation_scope": "Future weeks for existing workforce; employee overlap permitted "
            "and disclosed. Does not establish generalization to unseen employees.",
        "versions": {"python": platform.python_version(), "pandas": version("pandas")},
        "random_seed": None, "training_performed": False,
        "legacy_final_test_accessed": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    for filename, frame in (("cohort_all.csv", cohort), ("cohort_binary.csv", binary),
                            ("split_three_way.csv", three_assignments),
                            ("split_two_way_exploratory.csv", two_assignments)):
        frame.to_csv(output / filename, index=False, date_format="%Y-%m-%d")
    metadata["output_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in output.glob("*.csv")}
    metadata["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output_dir)


if __name__ == "__main__":
    main()
