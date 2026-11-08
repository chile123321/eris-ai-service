import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mexwell_test_ui.py"
SPEC = importlib.util.spec_from_file_location("mexwell_test_ui", SCRIPT)
ui = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ui)


def test_local_schema_and_demo_profile_exclude_id_and_target():
    schema, status = ui.load_local_schema()
    profile = ui.load_demo_profile(schema)
    assert status == "NO_QUALIFIED_MEXWELL_MODEL"
    assert len(schema["required"]) == len(profile["features"]) == 12
    assert profile["employee_id"] == "demo-reference-001"
    assert "Employee_ID" not in profile["features"]
    assert "Resigned" not in profile["features"]
    assert schema["properties"]["Hire_Date"]["format"] == "date"
    assert schema["properties"]["Department"]["enum"]


def test_form_payload_coerces_numbers_and_preserves_schema_order():
    schema, _ = ui.load_local_schema()
    profile = ui.load_demo_profile(schema)
    raw = {field: str(value) for field, value in profile["features"].items()}
    payload = ui.form_payload(" demo-reference-001 ", raw, schema)
    assert payload["schema_version"] == "mexwell-12-v1"
    assert payload["employee"]["employee_id"] == "demo-reference-001"
    assert isinstance(payload["employee"]["features"]["Age"], float)
    assert list(payload["employee"]["features"]) == schema["required"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("Age", "abc", "phải là số"), ("Age", "nan", "hữu hạn"),
     ("Department", "Unknown", "phải thuộc")],
)
def test_form_payload_rejects_invalid_values(field, value, message):
    schema, _ = ui.load_local_schema()
    profile = ui.load_demo_profile(schema)
    raw = {name: str(item) for name, item in profile["features"].items()}
    raw[field] = value
    with pytest.raises(ui.UiClientError, match=message):
        ui.form_payload(profile["employee_id"], raw, schema)


def test_explanation_text_sorts_absolute_contribution_and_uses_percentage_points():
    result = {
        "score": 0.2,
        "base_value": 0.1,
        "additivity_error": 1e-17,
        "contributions": {"small": -0.01, "large": 0.11},
    }
    text = ui.explanation_text(result)
    assert text.index("large") < text.index("small")
    assert "large: +11.0000 pp" in text
    assert "small: -1.0000 pp" in text


def test_missing_development_keeps_manual_entry_available(tmp_path):
    schema, _ = ui.load_local_schema()
    with pytest.raises(ui.UiClientError, match="nhập form thủ công"):
        ui.load_demo_profile(schema, tmp_path / "missing.csv")
