import os
import sys
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    ImageMessage,
    TextSendMessage
)

# ===============================
# ENV
# ===============================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    print("❌ LINE ENV NOT SET")
    sys.exit(1)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

app = Flask(__name__)

# ===============================
# CALLBACK
# ===============================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    print("📩 Webhook received")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("❌ Invalid signature")
        abort(400)
    except Exception as e:
        print("❌ Handler error:", e)

    return "OK"


# ===============================
# TEXT MESSAGE
# ===============================
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_text = event.message.text
    print("💬 Text:", user_text)

    reply = (
        "🌱 ส่งรูปมะเขือเทศมาได้เลยครับ\n"
        "ผมจะช่วยวิเคราะห์โรคให้ 😊"
    )

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# ===============================
# IMAGE MESSAGE
# ===============================
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    print("📸 Image received")

    # ดาวน์โหลดรูปจาก LINE
    message_id = event.message.id
    content = line_bot_api.get_message_content(message_id)

    image_path = "/tmp/input.jpg"
    with open(image_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

    print("✅ Image saved:", image_path)

    # -------------------------------
    # MOCK AI RESULT (แทน AI จริงก่อน)
    # -------------------------------
    disease = "Early Blight"
    confidence = 92.45
    detail = (
        "🔎 อาการ: ใบมีจุดสีน้ำตาลเข้ม\n"
        "🧪 วิธีรักษา: ใช้สารป้องกันเชื้อรา\n"
        "🛡️ ป้องกัน: หลีกเลี่ยงความชื้นสูง"
    )

    reply = (
        "🌱 ผลการวิเคราะห์โรคมะเขือเทศ\n\n"
        f"🦠 โรคที่พบ: {disease}\n"
        f"📊 ความมั่นใจ: {confidence:.2f}%\n\n"
        f"{detail}"
    )

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# ===============================
# HEALTH CHECK
# ===============================
@app.route("/")
def health():
    return "Tomato LINE Bot is running 🍅"


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
