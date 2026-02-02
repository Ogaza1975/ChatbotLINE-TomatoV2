import os
from datetime import datetime

from flask import Flask, request, abort

# ===== LINE BOT =====
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage

# ===== AI / ML =====
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image


# ===============================
# Flask App
# ===============================
app = Flask(__name__)


# ===============================
# HEALTH CHECK (Cloud Run)
# ===============================
@app.route("/")
def health():
    return "OK", 200


# ===============================
# LINE CONFIG (ENV)
# ===============================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# ===============================
# GOOGLE SHEET (lazy init)
# ===============================
def log_to_sheet(disease_name):
    from google.auth import default
    import gspread

    credentials, project = default()
    client = gspread.authorize(credentials)

    SPREADSHEET_ID = "1VhCs76yNRjb_voXbPDJu4uP9NHNXcCLzeJV3xnrSnFw"
    sheet = client.open_by_key(SPREADSHEET_ID).sheet1

    today = datetime.now().strftime("%Y-%m-%d")

    sheet.append_row(
        ["" for _ in range(12)] + [today, disease_name]
    )

    print("✅ บันทึกลง Google Sheet:", disease_name)


# ===============================
# LOAD AI MODEL
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

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CONF_THRESHOLD = 85


def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    confidence = conf.item() * 100

    if confidence < CONF_THRESHOLD:
        return None, confidence

    disease = class_names[pred.item()]
    return disease, confidence


# ===============================
# WEBHOOK
# ===============================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
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

    image_path = "input.jpg"
    with open(image_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

    disease, confidence = predict_image(image_path)

    if disease is None:
        reply = (
            "📷 ไม่สามารถวิเคราะห์ภาพได้อย่างแม่นยำ\n"
            "กรุณาส่งภาพมะเขือเทศใหม่อีกครั้ง\n"
            "ให้เห็นใบหรืออาการชัดเจน 🙏"
        )
    else:
        log_to_sheet(disease)
        reply = (
            f"🌱 ผลการวิเคราะห์โรคมะเขือเทศ\n\n"
            f"🦠 โรคที่พบ: {disease}\n"
            f"📊 ความมั่นใจ: {confidence:.2f}%"
        )

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# ===============================
# RUN (Cloud Run)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
