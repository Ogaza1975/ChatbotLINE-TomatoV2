import os
import sys
import torch
import torchvision.models as models
# ... (import อื่นๆ เหมือนเดิม)

app = Flask(__name__)

# ดึงค่า Config (แนะนำให้ใส่ค่าจริงลงไปเลยเพื่อทดสอบในรอบนี้)
LINE_ACCESS_TOKEN = "ใส่ Token จริงของคุณที่นี่"
LINE_SECRET = "ใส่ Secret จริงของคุณที่นี่"

line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_SECRET)

# ระบุ Path ให้ชัดเจนสำหรับ Docker
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mobilenetv2_chatbot.pth")
JSON_PATH = os.path.join(BASE_DIR, "tomato-SheetV2.json")

print(f"--- System Booting ---")
print(f"Checking Model File: {os.path.exists(MODEL_PATH)}")

try:
    device = "cpu"
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, 9)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    class_names = checkpoint["class_names"]
    model.eval()
    print("✅ Model status: Ready")
except Exception as e:
    print(f"❌ Boot Error: {str(e)}")

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    # พิมพ์ body ออกมาดูว่า LINE ส่งอะไรมา
    print(f"Incoming Request: {body[:100]}...") 
    try:
        handler.handle(body, signature)
    except Exception as e:
        print(f"❌ Handler Error: {str(e)}")
    return "OK"

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    print("📸 Image received!")
    # ส่งข้อความทดสอบกลับทันทีเพื่อเช็กว่า Webhook สำเร็จไหม
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="ระบบได้รับรูปภาพแล้ว กำลังประมวลผลสักครู่ครับ...")
    )
    
    # ... (โค้ดส่วน predict และ log_to_sheet) ...
