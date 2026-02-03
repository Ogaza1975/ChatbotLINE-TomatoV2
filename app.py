import os
import torch
import torchvision.models as models
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from PIL import Image
from torchvision import transforms
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage

app = Flask(__name__)

# --- ดึงค่าจาก Environment Variables ---
# หากคุณไม่ได้ตั้งค่าใน Cloud Run ให้เปลี่ยน os.environ.get(...) เป็น "ค่าจริง" ในอัญประกาศ
LINE_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_SECRET = os.environ.get("LINE_CHANNEL_SECRET")

# ตรวจสอบว่าดึงค่ามาได้จริงไหม (จะปรากฏใน Log)
print(f"DEBUG: Token loaded: {bool(LINE_ACCESS_TOKEN)}")
print(f"DEBUG: Secret loaded: {bool(LINE_SECRET)}")

line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_SECRET)

# --- โหลดโมเดล (มี Error Handling) ---
try:
    device = "cpu"
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, 9)
    
    # ดึง path ปัจจุบันเพื่อให้ชัวร์ว่าหาไฟล์เจอ
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "mobilenetv2_chatbot.pth")
    
    print(f"DEBUG: Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    class_names = checkpoint["class_names"]
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ MODEL ERROR: {str(e)}")

# (ส่วน disease_info และ transform ให้ใช้ตามโค้ดเดิมของคุณ)
# ... [ใส่โค้ดส่วน disease_info และ transform ของคุณ] ...

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("❌ Invalid Signature Error")
        abort(400)
    except Exception as e:
        print(f"❌ Callback Error: {str(e)}")
        abort(500)
    return "OK"

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    print("--- 📸 Received Image Message ---")
    try:
        # 1. ดาวน์โหลดรูป
        message_id = event.message.id
        content = line_bot_api.get_message_content(message_id)
        image_path = "/tmp/input.jpg"
        with open(image_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)
        print("✅ Step 1: Image saved to /tmp")

        # 2. ทำนายโรค
        print("🔄 Step 2: Predicting...")
        disease, confidence, detail = predict_image(image_path)
        print(f"✅ Prediction: {disease} ({confidence:.2f}%)")

        # 3. บันทึกลง Sheet (แยก Try เพื่อไม่ให้ Sheet พังแล้ว Bot ไม่ตอบ)
        try:
            if disease:
                log_to_sheet(disease)
                print("✅ Step 3: Logged to Sheet")
        except Exception as sheet_err:
            print(f"⚠️ Sheet Logging Failed: {sheet_err}")

        # 4. ส่งคำตอบกลับ
        if disease is None:
            reply_text = f"📷 ความแม่นยำต่ำเกินไป ({confidence:.2f}%) กรุณาส่งภาพใหม่"
        else:
            reply_text = f"🌱 วิเคราะห์สำเร็จ!\n🦠 โรค: {disease}\n📊 มั่นใจ: {confidence:.2f}%\n\n{detail}"
        
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        print("✅ Step 4: Reply sent")

    except Exception as e:
        error_msg = f"❌ Error in handle_image: {str(e)}"
        print(error_msg)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="ขออภัย ระบบขัดข้องระหว่างประมวลผล"))

# ... [ส่วนฟังก์ชัน log_to_sheet และ predict_image ของคุณ] ...
