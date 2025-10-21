
from dotenv import load_dotenv
load_dotenv()
import os
import json

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext

from data_provider import update_data

# Account keys from .env
ACCOUNTS = {
    
}

TOKEN = os.getenv("TELEGRAM_COINEX_REPORTER")

# Load menu data
def load_data(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Show user selection menu
def show_user_selection(update, context):
    keyboard = [
        [InlineKeyboardButton(name, callback_data=f"SELECT_{name}")]
        for name in ACCOUNTS.keys()
    ]
    update.message.reply_text("👥 Choose an account:", reply_markup=InlineKeyboardMarkup(keyboard))

# Show account menu
def show_account_menu(query, user_key):
    file_path = ACCOUNTS[user_key]["file"]
    data = load_data(file_path)
    keyboard = [[InlineKeyboardButton(text=key, callback_data=f"{user_key}:{key}")] for key in data]
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="BACK_TO_USERS")])
    markup = InlineKeyboardMarkup(keyboard)
    query.edit_message_text(f"📊 Data for {user_key}:", reply_markup=markup)

# Handle /start
def start(update: Update, context: CallbackContext):
    show_user_selection(update, context)

# Handle button presses
def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()
    data = query.data

    if data == "BACK_TO_USERS":
        show_user_selection(query, context)
        return

    if data.startswith("SELECT_"):
        user_key = data.replace("SELECT_", "")
        creds = ACCOUNTS[user_key]
        update_data(creds["access_id"], creds["secret_key"], creds["file"])
        show_account_menu(query, user_key)
        return

    if ":" in data:
        user_key, selection = data.split(":", 1)
        file_path = ACCOUNTS[user_key]["file"]
        menu_data = load_data(file_path)
        response = menu_data.get(selection, "⚠️ No data found.")
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data=f"SELECT_{user_key}")]]
        markup = InlineKeyboardMarkup(keyboard)
        query.edit_message_text(f"📊 {response}", reply_markup=markup)
        return

# Main runner
def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(button_handler))

    print("Bot is running...")
    updater.start_polling()
    updater.idle()
