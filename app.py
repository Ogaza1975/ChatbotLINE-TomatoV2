import os
from flask import Flask, request, abort
from datetime import datetime

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    ImageMessage,
    TextSendMessage
)

import gspread
from oauth2client.service_account import ServiceAccountCredentials


# ==================================================
# ENV (ต้องตั้งใน Cloud Run)
# ==================================================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ==================================================
# Flask
# ==================================================
app = Flask(__name__)

# ==================================================
# Google Sheet
# ==================================================
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds = ServiceAccountCredentials.from_json_keyfile_name(
    "Tomato-Sheet.json", scope
)
client = gspread.authorize(creds)

sheet = client.open_by_key(
    "1irin8ZPdTb5VX0pnFH9S4zz6RSl_chfjppxZLZ5Y2-Q"
).sheet1


def log_to_sheet(disease_name):
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    sheet.append_row(
        [""] * 12 + [now, disease_name],
        value_input_option="USER_ENTERED"
    )
    print("✅ บันทึก Google Sheet:", disease_name)


# ==================================================
# AI PREDICT (เอาโมเดลจริงมาแทนได้)
# ==================================================
def predict_image(image_path):
    """
    ตัวอย่าง mock
    เปลี่ยนเป็นโมเดลจริงได้โดยไม่กระทบ webhook
    """
    return (
        "Tomato Early Blight",
        91.87,
        "🍂 แนวทางเบื้องต้น:\n- ตัดใบที่เป็นโรค\n- พ่นสารป้องกันเชื้อรา\n- ลดความชื้น"
    )


# ==================================================
# CALLBACK
# ==================================================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        print("❌ CALLBACK ERROR:", e)

    return "OK"


# ==================================================
# TEXT MESSAGE
# ==================================================
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text="🍅 ส่งรูปมะเขือเทศมาได้เลยครับ ผมจะช่วยวิเคราะห์โรคให้ 😊"
        )
    )


# ==================================================
# IMAGE MESSAGE (แก้ timeout 100%)
# ==================================================
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    print("📸 Image received")

    # ✅ 1. ตอบ LINE ทันที (สำคัญที่สุด)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text="🔍 ได้รับรูปแล้ว กำลังวิเคราะห์ กรุณารอสักครู่ครับ…"
        )
    )

    # ==================================================
    # หลังจากนี้ LINE ไม่รอแล้ว (ทำงานหนักได้)
    # ==================================================
    try:
        message_id = event.message.id
        content = line_bot_api.get_message_content(message_id)

        image_path = "/tmp/input.jpg"
        with open(image_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

        print("✅ Image saved")

        disease, confidence, detail = predict_image(image_path)

        if disease:
            log_to_sheet(disease)
            result = (
                f"🌱 ผลการวิเคราะห์โรคมะเขือเทศ\n\n"
                f"🦠 โรคที่พบ: {disease}\n"
                f"📊 ความมั่นใจ: {confidence:.2f}%\n\n"
                f"{detail}"
            )
        else:
            result = (
                "📷 ไม่สามารถวิเคราะห์ภาพได้\n"
                "กรุณาส่งภาพใหม่ที่ชัดเจนขึ้นครับ 🙏"
            )

        # ✅ 2. ส่งผลลัพธ์รอบสอง
        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(text=result)
        )

    except Exception as e:
        print("❌ IMAGE ERROR:", e)


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
