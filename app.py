import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    ImageMessage,
    TextSendMessage
)

from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ==============================
# LINE CONFIG (ENV)
# ==============================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ==============================
# Flask
# ==============================
app = Flask(__name__)

# ==============================
# Google Sheet
# ==============================
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

    # ต่อแถวใหม่เสมอ
    sheet.append_row(
        ["" for _ in range(12)] + [now, disease_name],
        value_input_option="USER_ENTERED"
    )

    print("✅ บันทึกลง Google Sheet:", disease_name)


# ==============================
# AI PREDICT (ตัวอย่าง)
# ==============================
def predict_image(image_path):
    """
    ตัวอย่าง mock
    ถ้าใช้โมเดลจริง ค่อยเอามาแทนตรงนี้
    """
    return (
        "Tomato Early Blight",
        92.35,
        "🩺 แนวทางเบื้องต้น:\n- ตัดใบที่เป็นโรค\n- ใช้สารป้องกันเชื้อรา\n- หลีกเลี่ยงความชื้นสูง"
    )


# ==============================
# CALLBACK
# ==============================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        print("❌ ERROR:", e)

    return "OK"


# ==============================
# TEXT MESSAGE
# ==============================
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    reply = (
        "🍅 ส่งรูปมะเขือเทศมาได้เลยครับ\n"
        "ผมจะช่วยวิเคราะห์โรคให้ 😊"
    )

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# ==============================
# IMAGE MESSAGE (สำคัญที่สุด)
# ==============================
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    print("📸 Image received")

    # ✅ 1. ตอบ LINE ทันที (กัน timeout)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text="🔍 ได้รับรูปแล้ว กำลังวิเคราะห์โรคมะเขือเทศ กรุณารอสักครู่ครับ…"
        )
    )

    # ✅ 2. งานหนัก ทำทีหลัง
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
            "กรุณาถ่ายภาพใหม่ให้ชัดเจน เห็นใบหรืออาการผิดปกติครับ 🙏"
        )

    # ✅ 3. ส่งผลลัพธ์รอบสอง (push)
    line_bot_api.push_message(
        event.source.user_id,
        TextSendMessage(text=result)
    )


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

