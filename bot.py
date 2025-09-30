import logging
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hmac
import binascii
import hashlib

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
GRID_SIZE = 5
TOTAL_TILES = GRID_SIZE ** 2

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    keyboard = [
        [InlineKeyboardButton("Predict Safe Tiles", callback_data='predict')],
        [InlineKeyboardButton("Verify 1Win Round", callback_data='verify')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        'Welcome to the 1Win Mines Predictor Bot.\n'
        'This tool provides probabilistic suggestions for safe tiles and verifies provably fair outcomes.\n'
        'Use /predict <mines> for simulations or /verify for seed-based grid reconstruction.\n'
        'Note: 1Win uses provably fair technology for transparency; always play responsibly.',
        reply_markup=reply_markup
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = """
Commands:
/start - Begin interaction
/predict <mines_count> - Simulate safe tiles (e.g., /predict 3)
/verify <server_seed> <client_seed> <nonce> <mines_count> - Reconstruct grid from 1Win seeds
/stats <mines> <reveals> - View simulated success probabilities

1Win Tip: Access seeds via the game history for verification. With 3 mines, first-tile safety is approximately 88%.
    """
    await update.message.reply_text(help_text)

def simulate_safe_tiles(mines_count: int, num_simulations: int = 1000) -> list:
    """Simulate safe tiles via Monte Carlo method."""
    safe_counts = np.zeros(TOTAL_TILES)
    for _ in range(num_simulations):
        mines = np.random.choice(TOTAL_TILES, mines_count, replace=False)
        grid = np.ones(TOTAL_TILES)
        grid[mines] = 0  # 0 = mine, 1 = safe
        safe_counts += grid
    avg_safe = safe_counts / num_simulations
    suggestions = np.argsort(avg_safe)[-5:].tolist()  # Top 5 suggestions
    return suggestions

def verify_mines_grid(server_seed: str, client_seed: str, nonce: int, mines_count: int) -> list:
    """Reconstruct mine positions using HMAC-SHA256 for 1Win-style provably fair verification."""
    combined = f"{client_seed}:{nonce}"
    h = hmac.new(server_seed.encode(), combined.encode(), hashes.SHA256())
    hash_hex = h.hexdigest()
    positions = []
    for i in range(mines_count):
        # Use chunks of hash for position derivation (common in Mines implementations)
        start = i * (64 // mines_count)  # Distribute across full hash
        chunk = int(hash_hex[start:start+8], 16) % TOTAL_TILES
        positions.append(chunk)
    # Ensure unique positions by sampling without replacement if duplicates occur
    unique_positions = []
    for pos in positions:
        if pos not in unique_positions and len(unique_positions) < mines_count:
            unique_positions.append(pos)
        elif len(unique_positions) < mines_count:
            # Fallback: next available
            next_pos = (pos + 1) % TOTAL_TILES
            while next_pos in unique_positions:
                next_pos = (next_pos + 1) % TOTAL_TILES
            unique_positions.append(next_pos)
    return sorted(unique_positions[:mines_count])

def grid_to_str(mines_pos: list = None, safe_suggestions: list = None) -> str:
    """Visualize 5x5 grid for 1Win Mines."""
    grid = [['?' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    if mines_pos:
        for pos in mines_pos:
            row, col = divmod(pos, GRID_SIZE)
            grid[row][col] = 'X'  # Mine
    if safe_suggestions:
        for pos in safe_suggestions:
            row, col = divmod(pos, GRID_SIZE)
            if grid[row][col] == '?':
                grid[row][col] = 'S'  # Suggested safe
    return '\n'.join([' '.join(row) for row in grid])

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /predict command."""
    if not context.args:
        await update.message.reply_text("Usage: /predict <mines_count> (1-24)")
        return
    try:
        mines = int(context.args[0])
        if not 1 <= mines <= 24:
            raise ValueError
        suggestions = simulate_safe_tiles(mines)
        grid_str = grid_to_str(safe_suggestions=suggestions)
        prob = ((TOTAL_TILES - mines) / TOTAL_TILES) * 100
        await update.message.reply_text(
            f"For {mines} mines on 1Win:\n"
            f"Initial safe probability: {prob:.1f}%\n"
            f"Suggested tiles (S): \n{grid_str}\n"
            f"These are probabilistic; verify with seeds post-round."
        )
    except ValueError:
        await update.message.reply_text("Invalid mine count. Must be an integer between 1 and 24.")

async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /verify command."""
    if len(context.args) != 4:
        await update.message.reply_text("Usage: /verify <server_seed> <client_seed> <nonce> <mines_count>")
        return
    try:
        server, client, nonce_str, mines_str = context.args
        nonce = int(nonce_str)
        mines = int(mines_str)
        mines_pos = verify_mines_grid(server, client, nonce, mines)
        grid_str = grid_to_str(mines_pos)
        await update.message.reply_text(
            f"1Win Mines verification for nonce {nonce}:\n"
            f"Mines (X) positions: {mines_pos}\n{grid_str}\n"
            f"Compare with your game history for accuracy."
        )
    except ValueError:
        await update.message.reply_text("Invalid inputs. Ensure nonce and mines are integers.")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command."""
    if len(context.args) != 2:
        await update.message.reply_text("Usage: /stats <mines> <target_reveals>")
        return
    try:
        mines = int(context.args[0])
        reveals = int(context.args[1])
        # Binomial success probability
        prob = 1.0
        remaining_safe = TOTAL_TILES - mines
        for i in range(reveals):
            prob *= remaining_safe / (TOTAL_TILES - i)
            remaining_safe -= 1
        await update.message.reply_text(
            f"For {mines} mines and {reveals} reveals on 1Win:\n"
            f"Success probability: {prob * 100:.2f}%"
        )
    except ValueError:
        await update.message.reply_text("Invalid inputs. Provide integers for mines and reveals.")

def main() -> None:
    """Start the bot."""
    # Replace with your Telegram bot token
    application = Application.builder().token("8408573813:AAFHKBel9UZ2QvaTIpNa_5UsmcKSIF2gxYo").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("predict", predict))
    application.add_handler(CommandHandler("verify", verify))
    application.add_handler(CommandHandler("stats", stats))

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()