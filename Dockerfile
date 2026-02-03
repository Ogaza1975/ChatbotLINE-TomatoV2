FROM python:3.10-slim

WORKDIR /app

# ---- system deps (จำเป็นสำหรับ torch / PIL) ----
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- app ----
COPY . .

# ---- run ----
CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "-b", ":8080", "app:app"]
