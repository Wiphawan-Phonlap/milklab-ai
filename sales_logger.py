"""MilkLab Sales Logger (S2).

Usage:
    python sales_logger.py --menu "นมหมีฮอกไกโด" --qty 2 --price 65

Reads GOOGLE_SHEETS_CREDENTIALS and TELEGRAM_BOT_TOKEN (or LINE_CHANNEL_TOKEN) from env.
Appends row [timestamp, menu, qty, price, total] to a Google Sheet,
then sends a notification via Telegram or LINE bot.

นักศึกษาต้องเติม TODO ใน 4 จุดด้านล่างใน Session 2 Lab 1.3
"""

import argparse
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import gspread
import requests
from dotenv import load_dotenv

load_dotenv()


def append_to_sheet(menu: str, qty: int, price: float) -> dict:
    """TODO 1: ใช้ gspread เปิด Sheet ของตัวเอง แล้ว append_row ด้วย [timestamp, menu, qty, price, total]

    Returns dict {timestamp, menu, qty, price, total} ที่ append แล้ว
    Raises RuntimeError ถ้า credentials ไม่มี หรือ Sheet ไม่ accessible
    """
    if not menu or not menu.strip():
        raise ValueError("menu must not be empty")
    if qty <= 0:
        raise ValueError("qty must be positive")
    if price <= 0:
        raise ValueError("price must be positive")

    credentials_raw = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    spreadsheet_id = os.environ.get("SPREADSHEET_ID")

    if not credentials_raw:
        raise RuntimeError("GOOGLE_SHEETS_CREDENTIALS is missing")
    if not spreadsheet_id:
        raise RuntimeError("SPREADSHEET_ID is missing")

    try:
        credentials_info = json.loads(credentials_raw)
        client = gspread.service_account_from_dict(credentials_info)
        sheet = client.open_by_key(spreadsheet_id).sheet1

        timestamp = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(
            timespec="seconds"
        )
        total = qty * price
        row = [timestamp, menu, qty, price, total]
        sheet.append_row(row)

        return {
            "timestamp": timestamp,
            "menu": menu,
            "qty": qty,
            "price": price,
            "total": total,
        }
    except Exception as exc:
        raise RuntimeError(
            f"Failed to append row to Google Sheet: {exc}") from exc


def send_notification(message: str) -> str:
    """TODO 2: ส่ง message ไปยัง Telegram bot (ใช้ TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
    หรือ LINE bot (ใช้ LINE_CHANNEL_TOKEN) เลือกตัวใดตัวหนึ่ง

    Returns: provider name ที่ใช้ ("telegram" หรือ "line")
    Raises RuntimeError ถ้า no credentials
    """
    if not message or not message.strip():
        raise ValueError("message must not be empty")

    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    line_token = os.environ.get("LINE_CHANNEL_TOKEN")

    try:
        if telegram_token and telegram_chat_id:
            response = requests.post(
                f"https://api.telegram.org/bot{telegram_token}/sendMessage",
                json={"chat_id": telegram_chat_id, "text": message},
                timeout=10,
            )
            response.raise_for_status()
            return "telegram"

        if line_token:
            response = requests.post(
                "https://notify-api.line.me/api/notify",
                headers={"Authorization": f"Bearer {line_token}"},
                data={"message": message},
                timeout=10,
            )
            response.raise_for_status()
            return "line"

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID or LINE_CHANNEL_TOKEN is missing"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to send notification: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="MilkLab Sales Logger")
    parser.add_argument("--menu", required=True, help="ชื่อเมนู")
    parser.add_argument("--qty", type=int, required=True, help="จำนวนขวด")
    parser.add_argument("--price", type=float,
                        required=True, help="ราคาต่อขวด")
    args = parser.parse_args()

    try:
        # TODO 3: เรียก append_to_sheet แล้ว extract total
        row = append_to_sheet(args.menu, args.qty, args.price)
        total = row["total"]
    except Exception as exc:
        print(f"[ERROR] บันทึก Sheet ล้มเหลว: {exc}", file=sys.stderr)
        print("[HINT] ตรวจ GOOGLE_SHEETS_CREDENTIALS และ share Sheet กับ service account email", file=sys.stderr)
        return 1

    try:
        # TODO 4: เรียก send_notification ด้วย message ที่บอกยอดที่บันทึก
        provider = send_notification(
            f"บันทึก {args.menu} x{args.qty} = {total} บาท")
    except Exception as exc:
        print(
            f"[WARN] บันทึก Sheet สำเร็จแต่ส่งแจ้งเตือนล้มเหลว: {exc}", file=sys.stderr)
        return 0

    print(f"[OK] บันทึกและแจ้งเตือนผ่าน {provider} เรียบร้อย ยอด {total} บาท")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Session 2 Lab 1.3 complete: Sheets + Telegram notification implemented
