FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml requirements-lock.txt ./
COPY src ./src
RUN pip install --no-cache-dir -c requirements-lock.txt . && useradd --uid 10001 --create-home eris
USER eris
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "eris_ai.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
