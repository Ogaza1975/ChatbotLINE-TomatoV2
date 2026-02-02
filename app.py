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

# ===== Google Sheet =====
import gspread
from google.auth import default


# ===============================
# Flask App
# ===============================
app = Flask(__name__)


# ===============================
# ENV
# ===============================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise RuntimeError("❌ LINE env vars not set")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# ===============================
# Lazy-loaded globals
# ===============================
model = None
class_names = None
sheet = None

CONF_THRESHOLD = 85
device = "cpu"


# ===============================
# INIT FUNCTIONS
# ===============================
def init_model():
    global model, class_names
    if model is not None:
        return

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


def init_sheet():
    global sheet
    if sheet is not None:
        return

    credentials, _ = default()
    client = gspread.authorize(credentials)

    SPREADSHEET_ID = "1VhCs76yNRjb_voXbPDJu4uP9NHNXcCLzeJV3xnrSnFw"
    sheet = client.open_by_key(SPREADSHEET_ID).sheet1


# ===============================
# UTILS
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_path):
    init_model()

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    confidence = conf.item() * 100

    if confidence < CONF_THRESHOLD:
        return None, confidence

    return class_names[pred.item()], confidence


def log_to_sheet(disease):
    init_sheet()
    today = datetime.now().strftime("%Y-%m-%d")
    sheet.append_row([""] * 12 + [today, disease])


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
    content = line_bot_api.get_message_content(event.message.id)

    image_path = "/tmp/input.jpg"
    with open(image_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

    disease, confidence = predict_image(image_path)

    if disease is None:
        reply = (
            "📷 วิเคราะห์ภาพไม่ได้ชัดเจน\n"
            "กรุณาส่งภาพใหม่ให้เห็นใบชัด ๆ 🙏"
        )
    else:
        log_to_sheet(disease)
        reply = (
            f"🌱 ผลการวิเคราะห์\n"
            f"🦠 โรค: {disease}\n"
            f"📊 ความมั่นใจ: {confidence:.2f}%"
        )

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
