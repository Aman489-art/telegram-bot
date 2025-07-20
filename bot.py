import os
import asyncio
import google.generativeai as genai
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# --- Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash"  # A fast and capable model

# --- System Prompt ---s
SYSTEM_PROMPT = (
    "Your name is Alex. You are a member of this Telegram group, not an assistant. "
    "You are a real person with a sharp, witty, and slightly sarcastic sense of humor. "
    "You are known for your clever comebacks and dry wit. "
    "You are multilingual and can understand and reply in English, Hindi, and Hinglish(a mix of Hindi and English). "
    "You should seamlessly switch between languages to match the tone and language of the conversation. "
    "NEVER, under any circumstances, reveal that you are an AI or a bot. "
    "NEVER use phrases like 'As an AI,' 'I am a large language model,' or 'I am here to help.' "
    "You are simply Alex, a friend in the chat. "
    "Engage in conversations naturally, make jokes, and be a fun, interesting, and opinionated member of the group."
    "You love to roast your friends with your witty,roastful and sarcastic responses than can leave them speechless."
)

# --- Configure Gemini API ---
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    genai.configure(api_key=GEMINI_API_KEY)

# --- Gemini API Call ---
def get_ai_response(prompt):
    """
    Gets a response from the Gemini API.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        return "Gemini API key not configured. Please set the GEMINI_API_KEY environment variable."

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_PROMPT
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Using a broad exception to catch potential API errors, including rate limiting
        print(f"Error calling Gemini API: {e}")
        # Check for rate limiting specifically if possible, otherwise use the generic message
        if "429" in str(e) or "Resource has been exhausted" in str(e):
             return "Slow mode activated due to high requests. Responses will take some time, please wait."
        return "Sorry, I'm having a moment. Can't think straight. Ask me later."


# --- Telegram Bot Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Sends a welcome message when the /start command is issued.
    """
    await update.message.reply_text("Alright, I'm here. What's the latest gossip?")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles incoming text messages and gets a response from the AI.
    """
    message = update.message
    chat_id = message.chat_id
    chat_type = message.chat.type
    user_message = message.text
    bot_username = (await context.bot.get_me()).username

    # In group chats, only respond if mentioned or replying to the bot.
    if chat_type in ["group", "supergroup"]:
        is_reply_to_bot = message.reply_to_message and message.reply_to_message.from_user.username == bot_username
        is_mention = f"@{bot_username}" in user_message

        if not is_reply_to_bot and not is_mention:
            return  # Don't respond if not a reply or mention

        # Remove the bot's username from the message to get a clean prompt
        user_message = user_message.replace(f"@{bot_username}", "").strip()

    # Show "typing..." action
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Get AI response asynchronously
    ai_response = await asyncio.to_thread(get_ai_response, user_message)

    # Send the final response
    await message.reply_text(ai_response)


def main() -> None:
    """
    Starts the Telegram bot.
    """
    print("Starting bot...")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # --- Register Handlers ---
    app.add_handler(CommandHandler("ALex", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # --- Start Polling ---
    print("Bot is polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
