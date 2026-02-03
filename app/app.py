from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage, TextSendMessage
import os

from app.model_loader import predict_image
from app.sheet_logger import log_to_sheet

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    handler.handle(body, signature)
    return "OK"


@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # ตอบกลับทันที ป้องกัน timeout
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
