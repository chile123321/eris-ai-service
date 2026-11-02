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

Cần xác nhận nguồn và phiên bản dữ liệu, định nghĩa nhãn và thời điểm dự đoán, feature có thể thu thập từ backend, chiến lược split/evaluation và định dạng model artifact. Không đưa dataset, dự đoán cấp nhân viên hoặc model artifact vào Git.
