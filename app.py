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

# --- Configuration ---
LINE_CHANNEL_ACCESS_TOKEN = "YOUR_CHANNEL_ACCESS_TOKEN"
LINE_CHANNEL_SECRET = "YOUR_CHANNEL_SECRET"
SHEET_KEY = "1LugFaHx26ozkqofcRkIHTfs9hJ8G4VDVwi11gTG9UQk"
MODEL_PATH = "mobilenetv2_chatbot.pth"
GOOGLE_CHART_JSON = "tomato-SheetV2.json"

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# --- Model Setup ---
device = "cpu" # Cloud Run รันบน CPU เป็นหลัก
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 9)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
class_names = checkpoint["class_names"]
model.eval()

disease_info = {
    "Tomato_Bacterial_spot": "🍂 โรคใบจุดแบคทีเรีย\nหลีกเลี่ยงน้ำกระเด็น ใส่สารคอปเปอร์ และใช้เมล็ดพันธุ์ปลอดโรค",
    "Tomato_Early_blight": "🍁 โรคใบไหม้ระยะแรก\nตัดใบที่เป็นโรค พ่นสารป้องกันเชื้อรา และเว้นระยะปลูก",
    "Tomato_Late_blight": "🌧️ โรคใบไหม้ระยะท้าย\nพ่นสารป้องกันเชื้อราอย่างเร่งด่วน และกำจัดต้นที่ติดเชื้อ",
    "Tomato_Leaf_Mold": "🍃 โรคราน้ำค้างใบ\nลดความชื้น เพิ่มการระบายอากาศ",
    "Tomato_Septoria_leaf_spot": "⚫ โรคใบจุดเซพโทเรีย\nตัดใบเป็นโรค และพ่นสารป้องกันเชื้อรา",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "🕷️ ไรแดง\nฉีดน้ำแรง ๆ ใต้ใบ หรือใช้สารกำจัดไร",
    "Tomato__Target_Spot": "🎯 โรคใบจุดเป้า\nพ่นสารป้องกันเชื้อรา และหลีกเลี่ยงน้ำขัง",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "🌀 โรคใบหงิกเหลือง\nกำจัดแมลงหวี่ขาว และถอนต้นที่ติดเชื้อ",
    "Tomato_healthy": "✅ ต้นมะเขือเทศแข็งแรงดี"
}

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --- Functions ---
def log_to_sheet(disease_name):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CHART_JSON, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_KEY).worksheet("Dashboard")
        now = datetime.now().strftime("%d/%m/%Y")
        row_data = [""] * 12 + [now, disease_name]
        last_row = len(sheet.get_all_values()) + 1
        sheet.insert_row(row_data, last_row)
    except Exception as e:
        print(f"Error logging to sheet: {e}")

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    
    confidence = conf.item() * 100
    if confidence < 85:
        return None, confidence, None
    
    disease = class_names[pred.item()]
    detail = disease_info.get(disease, "")
    return disease, confidence, detail

# --- Routes ---
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_id = event.message.id
    content = line_bot_api.get_message_content(message_id)
    image_path = "/tmp/input.jpg" # Cloud Run ต้องเขียนไฟล์ลง /tmp
    with open(image_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

    disease, confidence, detail = predict_image(image_path)
    if disease is None:
        reply = "📷 ไม่สามารถวิเคราะห์ได้ชัดเจน กรุณาส่งภาพใหม่ในที่แสงสว่างเพียงพอ"
    else:
        log_to_sheet(disease)
        reply = f"🌱 ผลการวิเคราะห์\n🦠 โรค: {disease}\n📊 ความมั่นใจ: {confidence:.2f}%\n\n{detail}"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    # Google Cloud Run จะกำหนด PORT มาให้ผ่าน Env Var
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
