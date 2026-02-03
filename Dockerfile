# ใช้ Python ที่มีไลบรารีพื้นฐานครบ
FROM python:3.9-slim

# ตั้งค่าโฟลเดอร์ทำงาน
WORKDIR /app

# คัดลอกไฟล์ทั้งหมดจาก GitHub ลงไปในเครื่องจำลอง
COPY . .

# ติดตั้งไลบรารีที่ระบุไว้ใน requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# สั่งให้รัน app.py เมื่อเครื่องเริ่มทำงาน
CMD ["python", "app.py"]
