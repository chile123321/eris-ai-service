"""Small Tkinter client for the explicitly unqualified Mexwell demo API."""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import joblib
from dotenv import dotenv_values

API_URL = "http://127.0.0.1:8000"
SCHEMA_VERSION = "mexwell-12-v1"
DEMO_ID = "demo-reference-001"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "artifacts" / "mexwell_rf12_demo_v1" / "bundle.joblib"
DEFAULT_DEVELOPMENT = ROOT / "data" / "processed" / "development.csv"


class UiClientError(RuntimeError):
    """A safe, user-facing API error that never includes the service key."""


def load_local_schema(bundle_path: Path = DEFAULT_BUNDLE) -> tuple[dict[str, Any], str]:
    """Read the trusted local demo bundle schema without constructing an explainer."""
    if not bundle_path.is_file():
        raise UiClientError(f"Không tìm thấy bundle local: {bundle_path}")
    try:
        bundle = joblib.load(bundle_path)
        metadata = bundle["metadata"]
        schema = metadata["schema"]
        if metadata["schema_version"] != SCHEMA_VERSION:
            raise ValueError("schema_version không khớp")
        if set(schema["required"]) != set(schema["properties"]):
            raise ValueError("schema local không đầy đủ")
        return schema, str(metadata["validation_status"])
    except UiClientError:
        raise
    except Exception as error:
        raise UiClientError("Không đọc được schema từ bundle demo local.") from error


def load_demo_profile(
    schema: dict[str, Any], development_path: Path = DEFAULT_DEVELOPMENT
) -> dict[str, Any]:
    """Read only the first valid development row and omit ID/target from features."""
    if not development_path.is_file():
        raise UiClientError(
            "Không tìm thấy development.csv; bạn vẫn có thể nhập form thủ công."
        )
    required = list(schema["required"])
    try:
        with development_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader)
    except (OSError, StopIteration) as error:
        raise UiClientError("Không đọc được hồ sơ từ development.csv.") from error
    missing = [field for field in required if field not in row]
    if missing:
        raise UiClientError("development.csv thiếu field: " + ", ".join(missing))
    features: dict[str, Any] = {}
    for field in required:
        value = row[field]
        if schema["properties"][field].get("type") == "number":
            try:
                parsed = float(value)
            except ValueError as error:
                raise UiClientError(f"Giá trị demo của {field} không phải số.") from error
            if not math.isfinite(parsed):
                raise UiClientError(f"Giá trị demo của {field} không hữu hạn.")
            features[field] = parsed
        else:
            features[field] = value
    return {"employee_id": DEMO_ID, "features": features}


def service_key() -> str:
    """Load the existing root .env without printing or returning it in an error."""
    key = str(dotenv_values(ROOT / ".env").get("ERIS_SERVICE_KEY", ""))
    if not key:
        raise UiClientError("Thiếu ERIS_SERVICE_KEY trong .env ở root repo.")
    return key


def form_payload(
    employee_id: str, raw_values: dict[str, str], schema: dict[str, Any]
) -> dict[str, Any]:
    employee_id = employee_id.strip()
    if not employee_id:
        raise UiClientError("employee_id không được để trống.")
    features: dict[str, Any] = {}
    for field in schema["required"]:
        raw = raw_values.get(field, "").strip()
        if not raw:
            raise UiClientError(f"Thiếu giá trị cho {field}.")
        property_schema = schema["properties"][field]
        if property_schema.get("type") == "number":
            try:
                value: Any = float(raw)
            except ValueError as error:
                raise UiClientError(f"{field} phải là số.") from error
            if not math.isfinite(value):
                raise UiClientError(f"{field} phải là số hữu hạn.")
        else:
            value = raw
        choices = property_schema.get("enum")
        if choices and value not in choices:
            raise UiClientError(f"{field} phải thuộc: {', '.join(map(str, choices))}.")
        features[field] = value
    return {
        "schema_version": SCHEMA_VERSION,
        "employee": {"employee_id": employee_id, "features": features},
    }


def _http_detail(error: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(error.read().decode("utf-8"))
        detail = body.get("detail", "")
        if isinstance(detail, list):
            return "; ".join(
                f"{'.'.join(map(str, item.get('loc', [])))}: {item.get('msg', '')}"
                for item in detail
            )
        return str(detail)
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return ""


def api_request(
    path: str,
    key: str,
    payload: dict[str, Any] | None = None,
    *,
    api_url: str = API_URL,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_url + path,
        data=data,
        method="GET" if data is None else "POST",
        headers={
            "Content-Type": "application/json",
            "X-ERIS-Service-Key": key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = _http_detail(error)
        if error.code == 401:
            message = "401: ERIS_SERVICE_KEY không khớp với API đang chạy."
        elif error.code == 422:
            message = "422: Kiểm tra lại trường dữ liệu"
            if detail:
                message += f" — {detail}"
        elif error.code == 503:
            message = "503: Demo Mexwell hoặc bundle chưa sẵn sàng."
            if detail:
                message += f" — {detail}"
        else:
            message = f"API trả HTTP {error.code}."
            if detail:
                message += f" — {detail}"
        raise UiClientError(message) from error
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise UiClientError(
            "Không kết nối được API. Hãy bật API local tại http://127.0.0.1:8000."
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UiClientError("API trả dữ liệu không phải JSON hợp lệ.") from error


def explanation_text(result: dict[str, Any]) -> str:
    contributions = sorted(
        result["contributions"].items(), key=lambda item: abs(item[1]), reverse=True
    )
    lines = [
        f"Điểm Resigned=1: {result['score'] * 100:.4f}%",
        f"Base value: {result['base_value'] * 100:.4f}%",
        "",
        "Đóng góp SHAP (điểm phần trăm):",
    ]
    lines.extend(f"{field}: {value * 100:+.4f} pp" for field, value in contributions)
    lines.extend(
        [
            "",
            f"Sai số cộng: {result['additivity_error']:.3e}",
            "SHAP giải thích hành vi dự đoán, không phải nguyên nhân nghỉ việc.",
        ]
    )
    return "\n".join(lines)


def smoke_test(api_url: str = API_URL) -> dict[str, Any]:
    """Exercise the exact request helpers used by both UI buttons."""
    schema, _ = load_local_schema()
    profile = load_demo_profile(schema)
    raw = {field: str(profile["features"][field]) for field in schema["required"]}
    payload = form_payload(profile["employee_id"], raw, schema)
    key = service_key()
    prediction = api_request("/v1/predict", key, payload, api_url=api_url)
    explanation = api_request("/v1/explain", key, payload, api_url=api_url)
    score_delta = abs(prediction["score"] - explanation["score"])
    reconstructed = explanation["base_value"] + sum(explanation["contributions"].values())
    additivity_delta = abs(reconstructed - prediction["score"])
    if score_delta > 1e-12 or additivity_delta > explanation["additivity_atol"]:
        raise UiClientError("Predict và SHAP không vượt qua kiểm tra nhất quán.")
    return {
        "profile": DEMO_ID,
        "model_id": prediction["model_id"],
        "validation_status": prediction["validation_status"],
        "score": prediction["score"],
        "base_value": explanation["base_value"],
        "score_delta": score_delta,
        "additivity_delta": additivity_delta,
    }


class MexwellTestUi:
    def __init__(self, root: Any, schema: dict[str, Any], local_status: str):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.schema = schema
        self.tk = tk
        self.ttk = ttk
        self.events: queue.Queue[tuple[str, Any, Any]] = queue.Queue()
        self.inputs: dict[str, Any] = {}
        self.employee_id = tk.StringVar(value=DEMO_ID)
        self.status = tk.StringVar(value=f"Validation: {local_status} (local)")
        root.title("Model demo Mexwell — Predict & SHAP")
        root.geometry("860x760")
        root.minsize(760, 650)

        container = ttk.Frame(root, padding=14)
        container.pack(fill="both", expand=True)
        ttk.Label(container, text="Model demo Mexwell", font=("Segoe UI", 17, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(container, textvariable=self.status).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(2, 12)
        )
        ttk.Label(container, text="employee_id").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(container, textvariable=self.employee_id, width=42).grid(
            row=2, column=1, sticky="ew", pady=3
        )

        for row_index, field in enumerate(schema["required"], start=3):
            property_schema = schema["properties"][field]
            ttk.Label(container, text=field).grid(row=row_index, column=0, sticky="w", pady=3)
            variable = tk.StringVar()
            choices = property_schema.get("enum")
            if choices:
                widget = ttk.Combobox(
                    container, textvariable=variable, values=choices, state="readonly", width=39
                )
            else:
                widget = ttk.Entry(container, textvariable=variable, width=42)
            widget.grid(row=row_index, column=1, sticky="ew", pady=3)
            type_text = "số" if property_schema.get("type") == "number" else "chuỗi"
            if property_schema.get("format") == "date":
                type_text = "YYYY-MM-DD"
            ttk.Label(container, text=type_text, foreground="#666").grid(
                row=row_index, column=2, sticky="w", padx=(8, 0)
            )
            self.inputs[field] = variable

        button_row = 3 + len(schema["required"])
        self.load_button = ttk.Button(container, text="Nạp hồ sơ demo", command=self.load_demo)
        self.predict_button = ttk.Button(
            container, text="Dự đoán", command=lambda: self.submit("/v1/predict")
        )
        self.explain_button = ttk.Button(
            container, text="Giải thích SHAP", command=lambda: self.submit("/v1/explain")
        )
        for column, button in enumerate(
            (self.load_button, self.predict_button, self.explain_button)
        ):
            button.grid(row=button_row, column=column, sticky="ew", padx=3, pady=(14, 8))

        self.output = tk.Text(container, height=15, wrap="word", state="disabled")
        self.output.grid(row=button_row + 1, column=0, columnspan=3, sticky="nsew")
        scrollbar = ttk.Scrollbar(container, command=self.output.yview)
        scrollbar.grid(row=button_row + 1, column=3, sticky="ns")
        self.output.configure(yscrollcommand=scrollbar.set)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(button_row + 1, weight=1)
        self.root.after(50, self._poll_events)
        self._run_async(self._fetch_model_info, self._show_model_info)

    def _set_output(self, text: str):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def _buttons(self, state: str):
        self.predict_button.configure(state=state)
        self.explain_button.configure(state=state)

    def _run_async(self, work: Any, success: Any):
        def run():
            try:
                result = work()
            except Exception as error:  # Keep worker failures inside the window.
                message = str(error) if isinstance(error, UiClientError) else (
                    "Có lỗi nội bộ khi xử lý yêu cầu."
                )
                self.events.put(("error", message, None))
            else:
                self.events.put(("success", result, success))

        threading.Thread(target=run, daemon=True).start()

    def _poll_events(self):
        try:
            while True:
                event, value, callback = self.events.get_nowait()
                if event == "error":
                    self._finish_error(value)
                else:
                    callback(value)
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(50, self._poll_events)

    def _fetch_model_info(self):
        return api_request("/v1/model-info", service_key())

    def _show_model_info(self, info: dict[str, Any]):
        self.status.set(
            f"Validation: {info['validation_status']} | Model: {info['model_id']}"
        )

    def load_demo(self):
        try:
            profile = load_demo_profile(self.schema)
            self.employee_id.set(profile["employee_id"])
            for field, value in profile["features"].items():
                self.inputs[field].set(str(value))
            self._set_output("Đã nạp hồ sơ demo; Employee_ID và Resigned không vào model input.")
        except UiClientError as error:
            self._set_output(str(error))

    def _payload(self):
        return form_payload(
            self.employee_id.get(),
            {field: variable.get() for field, variable in self.inputs.items()},
            self.schema,
        )

    def submit(self, path: str):
        try:
            payload = self._payload()
            key = service_key()
        except UiClientError as error:
            self._set_output(str(error))
            return
        self._buttons("disabled")
        self._set_output("Đang gọi API…")
        self._run_async(lambda: api_request(path, key, payload), self._show_result)

    def _show_result(self, result: dict[str, Any]):
        self._buttons("normal")
        self.status.set(
            f"Validation: {result['validation_status']} | Model: {result['model_id']}"
        )
        if "contributions" in result:
            self._set_output(explanation_text(result))
        else:
            self._set_output(
                f"Điểm Resigned=1: {result['score'] * 100:.4f}%\n"
                f"Model: {result['model_id']}\n"
                f"Validation: {result['validation_status']}"
            )

    def _finish_error(self, message: str):
        self._buttons("normal")
        self._set_output(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--api-url", default=API_URL)
    args = parser.parse_args()
    if args.smoke_test:
        print(json.dumps(smoke_test(args.api_url), indent=2, ensure_ascii=False))
        return
    import tkinter as tk

    schema, local_status = load_local_schema()
    root = tk.Tk()
    MexwellTestUi(root, schema, local_status)
    root.mainloop()


if __name__ == "__main__":
    main()
