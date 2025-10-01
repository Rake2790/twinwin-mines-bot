import logging
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import Conflict

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Replace with your bot token (regenerate via @BotFather if needed)
BOT_TOKEN = "8408573813:AAEPEq1ntw5O4Zmuq_2tA74mouM1CrMbjHA"

# Flag to track if the bot is already running
_is_running = False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /start command."""
    await update.message.reply_text("Hello! I am Twinwin Mines Bot. Use /help for commands.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /help command."""
    await update.message.reply_text("Help is on the way! Contact @Rake27900 for support.")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /predict command with a number argument."""
    if not context.args:
        await update.message.reply_text("Please provide a number, e.g., /predict 3")
        return
    try:
        number = int(context.args[0])
        if number <= 0:
            await update.message.reply_text("Please provide a positive number.")
            return
        # Example prediction using numpy (random probability)
        prediction = np.random.random() * number
        await update.message.reply_text(f"Predicted value: {prediction:.2f} (based on {number})")
    except ValueError:
        await update.message.reply_text("Invalid input. Please provide a valid number.")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors globally."""
    logger.error(f"Update {update} caused error {context.error}")
    if isinstance(context.error, Conflict):
        logger.error("Conflict detected: Another instance is running. Shutting down.")
        raise context.error

def main() -> None:
    """Main function to run the bot."""
    global _is_running
    if _is_running:
        logger.error("Bot is already running. Aborting new instance.")
        raise RuntimeError("Only one bot instance is allowed.")

    _is_running = True
    try:
        # Initialize the Application
        application = (
            Application.builder()
            .token(BOT_TOKEN)
            .build()
        )

        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("predict", predict))
        application.add_error_handler(error_handler)

        # Start the bot with allowed updates as a list
        logger.info("Bot is starting...")
        application.run_polling(allowed_updates=["message", "callback_query"])
    except Conflict as e:
        logger.error(f"Conflict error: {e}. Ensure no other instances are running.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
    finally:
        _is_running = False
        logger.info("Bot has stopped.")

if __name__ == "__main__":
    main()