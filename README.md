# ERIS AI Service

Dịch vụ FastAPI cho ERIS. Demo Mexwell dùng RF baseline 12 feature để phân loại `Resigned`; trạng thái đánh giá là `NO_QUALIFIED_MEXWELL_MODEL`, chưa đủ điều kiện sử dụng thực tế.

## Yêu cầu và cài đặt

- Python 3.12 hoặc 3.13
- Node.js 20+ chỉ khi kiểm tra client tích hợp backend

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pip install --no-deps -e .
```

File `.env` được nạp tự động khi chạy từ thư mục repository. Biến `ERIS_SERVICE_KEY` phải có ít nhất 24 ký tự và không được commit.

```powershell
uvicorn eris_ai.api:app --host 127.0.0.1 --port 8000
```

`GET /health` hoạt động khi service khởi động. Các endpoint khác yêu cầu header `X-ERIS-Service-Key`. Inference trả 503 nếu chưa bật demo hoặc bundle thiếu/không hợp lệ. `/v1/predict-batch` chưa được bật cho demo.

## Demo predict một người và TreeSHAP

Chạy từ thư mục repo; giữ nguyên `.env` và service key. Lệnh tạo bundle xác minh development, fit RF một lần và tái sử dụng nếu bundle đã tồn tại; không mở final test.

```powershell
.venv\Scripts\python.exe -m eris_ai.mexwell_demo
$env:ERIS_MEXWELL_DEMO_ENABLED = "true"
$env:ERIS_MEXWELL_DEMO_BUNDLE = "artifacts/mexwell_rf12_demo_v1/bundle.joblib"
.venv\Scripts\python.exe -m uvicorn eris_ai.api:app --host 127.0.0.1 --port 8000
```

`POST /v1/predict` và `POST /v1/explain` cùng nhận `schema_version="mexwell-12-v1"` và `employee` gồm `employee_id`, `features`. Schema đủ 12 feature xem tại `GET /v1/model-info`; ID chỉ để đối chiếu. Điểm là `predict_proba(Resigned=1)`, không có horizon 90 ngày. TreeSHAP dùng thống kê đường đi trên development trong RF, gộp one-hot về feature gốc; base value và đóng góp có đơn vị xác suất (0–1). Sai số cộng tối đa `1e-8`; đối chiếu hai lần suy luận dùng `1e-12` do RF cộng song song. SHAP giải thích hành vi dự đoán, không khẳng định nguyên nhân nghỉ việc. Chỉ nạp bundle joblib nội bộ đáng tin cậy.

## Kiểm tra

```powershell
pytest -q
ruff check .
node --test integration/node/ai-client.test.cjs
```

## Chuẩn bị dữ liệu và baseline

Nguồn chuẩn bị dữ liệu là CSV gốc có SHA-256 được xác minh. `Hire_Date` được parse theo format rõ ràng `YYYY-MM-DD HH:MM:SS.ffffff`, giữ nguyên ngày lịch và loại phần giờ. Không đưa dataset, dự đoán cấp nhân viên hoặc model artifact vào Git.

Chuẩn bị dữ liệu chưa scale và split cố định:

```powershell
python -m eris_ai.data_preparation `
  --source-csv docs/Extended_Employee_Performance_and_Productivity_Data.csv `
  --output-dir data/processed
```

Lệnh này không fit preprocessing và không train model. Các output trong `data/processed/` bị Git ignore.

Chạy baseline development-only với Gradient Boosting và Random Forest:

```powershell
python -m eris_ai.baseline `
  --development data/processed/development.csv `
  --output-dir artifacts/baseline-gb-rf-<run-id>
```

Baseline dùng 3-fold stratified CV và ghi metrics, OOF predictions cùng metadata vào run directory. Lệnh không đọc final test.
