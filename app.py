from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage, TextSendMessage
import os

from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms

# ---------------- Flask ----------------
app = Flask(__name__)

line_bot_api = LineBotApi(os.environ.get("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.environ.get("LINE_CHANNEL_SECRET"))

# ---------------- AI (Lazy Load) ----------------
device = "cpu"
model = None
class_names = None

CONF_THRESHOLD = 85

disease_info = {
    "Tomato_Bacterial_spot": "🍂 โรคใบจุดแบคทีเรีย\nหลีกเลี่ยงน้ำกระเด็น ใช้สารคอปเปอร์",
    "Tomato_Early_blight": "🍁 โรคใบไหม้ระยะแรก\nตัดใบเป็นโรค พ่นสารป้องกันเชื้อรา",
    "Tomato_Late_blight": "🌧️ โรคใบไหม้ระยะท้าย\nพ่นสารป้องกันเชื้อราเร่งด่วน",
    "Tomato_Leaf_Mold": "🍃 โรคราน้ำค้างใบ\nลดความชื้น เพิ่มอากาศถ่ายเท",
    "Tomato_Septoria_leaf_spot": "⚫ โรคใบจุดเซพโทเรีย\nตัดใบและพ่นสารป้องกันเชื้อรา",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "🕷️ ไรแดง\nฉีดน้ำใต้ใบ หรือใช้สารกำจัดไร",
    "Tomato__Target_Spot": "🎯 โรคใบจุดเป้า\nหลีกเลี่ยงน้ำขัง",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "🌀 โรคใบหงิกเหลือง\nกำจัดแมลงหวี่ขาว",
    "Tomato_healthy": "✅ ต้นมะเขือเทศแข็งแรงดี"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model():
    global model, class_names
    if model is not None:
        return

    print("🔄 Loading AI model...")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, 9)

    checkpoint = torch.load(
        "mobilenetv2_chatbot.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state"])
    class_names = checkpoint["class_names"]
    model.eval()

    print("✅ Model loaded")

def predict_image(image_path):
    load_model()

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

# ---------------- LINE Webhook ----------------
@app.route("/callback", methods=["POST"])
def callback():
    try:
        signature = request.headers.get("X-Line-Signature")
        body = request.get_data(as_text=True)
        handler.handle(body, signature)
    except Exception as e:
        print("❌ Webhook error:", e)

    return "OK", 200   # สำคัญมาก

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    try:
        message_id = event.message.id
        content = line_bot_api.get_message_content(message_id)

        image_path = "/tmp/input.jpg"
        with open(image_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        disease, confidence, detail = predict_image(image_path)

        if disease is None:
            reply = "📷 ภาพไม่ชัดเจน กรุณาถ่ายใหม่ให้เห็นอาการชัดเจน"
        else:
            reply = (
                f"🌱 ผลการวิเคราะห์\n\n"
                f"🦠 โรค: {disease}\n"
                f"📊 ความมั่นใจ: {confidence:.2f}%\n\n"
                f"{detail}"
            )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply)
        )

    except Exception as e:
        print("❌ Image handler error:", e)

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
