import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds = ServiceAccountCredentials.from_json_keyfile_name(
    "tomato-SheetV2.json",
    scope
)

client = gspread.authorize(creds)

sheet = client.open_by_key(
    "1LugFaHx26ozkqofcRkIHTfs9hJ8G4VDVwi11gTG9UQk"
).worksheet("Dashboard")


def log_to_sheet(disease_name):
    now = datetime.now().strftime("%d/%m/%Y")
    row_data = [""] * 12 + [now, disease_name]
    last_row = len(sheet.get_all_values()) + 1
    sheet.insert_row(row_data, last_row)
