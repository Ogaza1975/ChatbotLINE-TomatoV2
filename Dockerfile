# ใช้ Python Image ที่มีขนาดเล็ก
FROM python:3.9-slim

# ตั้งค่า Working Directory
WORKDIR /app

# คัดลอกไฟล์ทั้งหมดลงไปใน Container
COPY . .

# ติดตั้ง Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# สั่งรันแอป
CMD ["python", "app.py"]
