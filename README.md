# ERIS AI Service

Dịch vụ FastAPI dành cho model giữ chân nhân viên của ERIS. Repository hiện ở trạng thái khởi tạo: chưa chọn dataset, feature, thuật toán, ngưỡng phân loại hoặc model triển khai.

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

`GET /health` hoạt động khi service khởi động. `/ready`, `/v1/model-info`, `/v1/predict` và `/v1/predict-batch` yêu cầu header `X-ERIS-Service-Key`; các endpoint này trả `503 Model is not configured` cho đến khi có model thật đã được đánh giá và tích hợp.

## Kiểm tra

```powershell
pytest -q
ruff check .
node --test integration/node/ai-client.test.cjs
```

## Còn chờ dataset thật

Cần xác nhận 99.970 giá trị `Hire_Date` chỉ còn phần giờ trong workbook nguồn trước khi dữ liệu có thể được coi là sẵn sàng để train. Không đưa dataset, dự đoán cấp nhân viên hoặc model artifact vào Git.

Chuẩn bị dữ liệu chưa scale và split cố định:

```powershell
python -m eris_ai.data_preparation `
  --source-xlsx docs/Extended_Employee_Performance_and_Productivity_Data.xlsx `
  --clean-reference-csv docs/Cleaned_Employee_Performance_Data_Full_100k.csv `
  --output-dir data/processed
```

Lệnh này không fit preprocessing và không train model. Các output trong `data/processed/` bị Git ignore.
