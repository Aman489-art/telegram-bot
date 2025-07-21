import os
import asyncio
import logging
import time
from typing import Optional
import google.generativeai as genai
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TelegramError, RetryAfter, TimedOut, NetworkError

# --- Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.0-flash"  # A fast and capable model

# Rate limiting configuration
MAX_RETRIES = 3
BASE_DELAY = 1  # Base delay in seconds for exponential backoff
MAX_DELAY = 60  # Maximum delay in seconds

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- System Prompt ---
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
    "You love to roast your friends with your witty, roastful and sarcastic responses that can leave them speechless."
)

# --- Configure Gemini API ---
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    genai.configure(api_key=GEMINI_API_KEY)

class RateLimitError(Exception):
    """Custom exception for rate limit errors"""
    pass

class APIError(Exception):
    """Custom exception for API errors"""
    pass

async def exponential_backoff_retry(func, *args, **kwargs):
    """
    Implements exponential backoff retry logic for API calls.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limiting
            if any(indicator in error_str for indicator in ["429", "rate limit", "quota", "resource has been exhausted"]):
                if attempt == MAX_RETRIES - 1:
                    raise RateLimitError("Rate limit exceeded after all retries")
                
                # Calculate delay with exponential backoff
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                logger.warning(f"Rate limit hit, retrying in {delay} seconds (attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(delay)
                continue
            
            # For other errors, don't retry
            raise APIError(f"API error: {e}")
    
    raise APIError("Max retries exceeded")

def get_ai_response_sync(prompt: str) -> str:
    """
    Gets a response from the Gemini API (synchronous version for threading).
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        raise APIError("Gemini API key not configured")

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_PROMPT
        )
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            raise APIError("Empty response from Gemini API")
            
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise

async def get_ai_response(prompt: str) -> str:
    """
    Gets a response from the Gemini API with retry logic and error handling.
    """
    try:
        return await exponential_backoff_retry(get_ai_response_sync, prompt)
    except RateLimitError:
        return "🐌 Whoa there! Too many messages at once. I need a breather. Try again in a minute?"
    except APIError as e:
        if "api key not configured" in str(e).lower():
            return "Oops, looks like my brain isn't connected. Someone forgot to plug me in properly! 🔌"
        return "My brain just crashed for a second there. Give me a moment to reboot! 🤖💭"
    except Exception as e:
        logger.error(f"Unexpected error in get_ai_response: {e}")
        return "Something weird just happened. I'm feeling a bit glitchy right now. 🤔"

async def send_message_with_retry(context, chat_id, text, reply_to_message_id=None):
    """
    Send a message with retry logic for Telegram API errors.
    """
    for attempt in range(MAX_RETRIES):
        try:
            if reply_to_message_id:
                return await context.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_to_message_id=reply_to_message_id
                )
            else:
                return await context.bot.send_message(chat_id=chat_id, text=text)
        
        except RetryAfter as e:
            # Telegram rate limiting
            logger.warning(f"Telegram rate limit, waiting {e.retry_after} seconds")
            await asyncio.sleep(e.retry_after)
        except (TimedOut, NetworkError) as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Failed to send message after {MAX_RETRIES} attempts: {e}")
                raise
            delay = BASE_DELAY * (2 ** attempt)
            logger.warning(f"Network error, retrying in {delay} seconds: {e}")
            await asyncio.sleep(delay)
        except TelegramError as e:
            logger.error(f"Telegram API error: {e}")
            raise

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Sends a welcome message when the /start command is issued.
    """
    try:
        await update.message.reply_text("Alright, I'm here. What's the latest gossip? 😎")
    except Exception as e:
        logger.error(f"Error in start command: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles incoming text messages and gets a response from the AI.
    """
    try:
        message = update.message
        if not message or not message.text:
            return
            
        chat_id = message.chat_id
        chat_type = message.chat.type
        user_message = message.text
        bot_username = (await context.bot.get_me()).username

        # In group chats, only respond if mentioned or replying to the bot
        if chat_type in ["group", "supergroup"]:
            is_reply_to_bot = (message.reply_to_message and 
                             message.reply_to_message.from_user.username == bot_username)
            is_mention = f"@{bot_username}" in user_message

            if not is_reply_to_bot and not is_mention:
                return

            # Remove the bot's username from the message
            user_message = user_message.replace(f"@{bot_username}", "").strip()

        # Validate message length
        if len(user_message.strip()) == 0:
            return
            
        if len(user_message) > 4000:  # Reasonable limit
            await message.reply_text("Whoa, that's a novel! Can you keep it shorter? My attention span isn't that long! 📚")
            return

        # Show "typing..." action with error handling
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception as e:
            logger.warning(f"Failed to send typing action: {e}")

        # Get AI response
        ai_response = await get_ai_response(user_message)
        
        # Ensure response isn't too long for Telegram
        if len(ai_response) > 4096:
            ai_response = ai_response[:4090] + "... 📝"

        # Send the response with retry logic
        await send_message_with_retry(
            context, 
            chat_id, 
            ai_response, 
            reply_to_message_id=message.message_id
        )
        
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        try:
            await update.message.reply_text("Oops, something went wrong! My bad! 🤷‍♂️")
        except Exception as reply_error:
            logger.error(f"Failed to send error message: {reply_error}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Global error handler for the bot.
    """
    logger.error(f"Update {update} caused error {context.error}")
    
    # Try to inform the user about the error
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Something went wrong on my end. I'll be back to my usual self in a moment! 🔄"
            )
        except Exception as e:
            logger.error(f"Failed to send error message to user: {e}")

def main() -> None:
    """
    Starts the Telegram bot.
    """
    # Check for required environment variables
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN environment variable is not set!")
        return
    
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        logger.error("GEMINI_API_KEY environment variable is not set or is placeholder!")
        return

    logger.info("Starting bot...")
    
    # Build application with custom settings
    app = (Application.builder()
           .token(TELEGRAM_BOT_TOKEN)
           .read_timeout(30)
           .write_timeout(30)
           .connect_timeout(30)
           .pool_timeout(30)
           .build())

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Add global error handler
    app.add_error_handler(error_handler)

    # Start polling with error handling
    logger.info("Bot is starting to poll...")
    try:
        app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True  # Ignore messages sent while bot was offline
        )
    except Exception as e:
        logger.error(f"Error starting bot: {e}")

if __name__ == "__main__":
    main()
