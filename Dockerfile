FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY mobilenetv2_chatbot.pth .
COPY tomato-SheetV2.json .

CMD ["python", "-m", "app.app"]
