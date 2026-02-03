from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage, TextSendMessage
import os
import json
from datetime import datetime

# ===============================
# PyTorch + Image
# ===============================
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# ===============================
# Google Sheet
# ===============================
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ===============================
# Flask App
# ===============================
app = Flask(__name__)

# ===============================
# LINE Config (ENV)
# ===============================
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ===============================
# Google Sheet Config (ENV)
# ===============================
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

service_account_info = json.loads(
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
)

creds = ServiceAccountCredentials.from_json_keyfile_dict(
    service_account_info,
    scope
)

client = gspread.authorize(creds)

sheet = client.open_by_key(
    "1LugFaHx26ozkqofcRkIHTfs9hJ8G4VDVwi11gTG9UQk"
).worksheet("Dashboard")

# ===============================
# Disease Info
# ===============================
disease_info = {
    "Tomato_Bacterial_spot": "🍂 โรคใบจุดแบคทีเรีย\nหลีกเลี่ยงน้ำกระเด็น และใช้สารคอปเปอร์",
    "Tomato_Early_blight": "🍁 โรคใบไหม้ระยะแรก\nตัดใบที่เป็นโรค พ่นสารป้องกันเชื้อรา",
    "Tomato_Late_blight": "🌧️ โรคใบไหม้ระยะท้าย\nพ่นสารป้องกันเชื้อราอย่างเร่งด่วน",
    "Tomato_Leaf_Mold": "🍃 โรคราน้ำค้างใบ\nลดความชื้น เพิ่มการระบายอากาศ",
    "Tomato_Septoria_leaf_spot": "⚫ โรคใบจุดเซพโทเรีย\nตัดใบเป็นโรคและพ่นสารป้องกันเชื้อรา",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "🕷️ ไรแดง\nฉีดน้ำใต้ใบหรือใช้สารกำจัดไร",
    "Tomato__Target_Spot": "🎯 โรคใบจุดเป้า\nหลีกเลี่ยงน้ำขัง พ่นสารป้องกันเชื้อรา",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "🌀 โรคใบหงิกเหลือง\nกำจัดแมลงหวี่ขาว และถอนต้นที่ติดเชื้อ",
    "Tomato_healthy": "✅ ต้นมะเขือเทศแข็งแรงดี"
}

# ===============================
# Load Model (ONCE)
# ===============================
device = "cpu"

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 9)

checkpoint = torch.load(
    "mobilenetv2_chatbot.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state"])
class_names = checkpoint["class_names"]
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CONF_THRESHOLD = 85  # %

# ===============================
# Helper Functions
# ===============================
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    confidence = conf.item() * 100

    if confidence < CONF_THRESHOLD:
        return None, confidence, None

    disease = class_names[pred.item()]
    detail = disease_info.get(disease, "")

    return disease, confidence, detail


def log_to_sheet(disease_name):
    now = datetime.now().strftime("%d/%m/%Y")
    row_data = [""] * 12 + [now, disease_name]
    last_row = len(sheet.get_all_values()) + 1
    sheet.insert_row(row_data, last_row)

# ===============================
# Routes
# ===============================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    handler.handle(body, signature)
    return "OK"


@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # ตอบกลับเร็ว ป้องกัน timeout
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="🔍 กำลังวิเคราะห์ภาพ กรุณารอสักครู่...")
    )

    message_id = event.message.id
    content = line_bot_api.get_message_content(message_id)

    image_path = "input.jpg"
    with open(image_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

    disease, confidence, detail = predict_image(image_path)

    if disease is None:
        reply = (
            "📷 ไม่สามารถวิเคราะห์ภาพได้อย่างแม่นยำ\n\n"
            "กรุณาส่งภาพใหม่ที่ชัดเจน เห็นใบหรืออาการผิดปกติ "
            "และถ่ายในบริเวณที่มีแสงสว่างเพียงพอ 🙏"
        )
    else:
        log_to_sheet(disease)
        reply = (
            f"🌱 ผลการวิเคราะห์โรคมะเขือเทศ\n\n"
            f"🦠 โรคที่พบ: {disease}\n"
            f"📊 ความมั่นใจ: {confidence:.2f}%\n\n"
            f"{detail}"
        )

    line_bot_api.push_message(
        event.source.user_id,
        TextSendMessage(text=reply)
    )

# ===============================
# Start App
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
