import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import telebot
from dotenv import load_dotenv
from flask import Flask

# Initialize Flask app for Render Web Service
app = Flask(__name__)
port = int(os.getenv("PORT", 5000))  # Default to 5000 if PORT not set by Render

# Load environment variables with explicit path to .env file
current_dir = os.path.dirname(__file__)
env_path = os.path.join(current_dir, '.env')
print(f"Attempting to load .env from: {env_path}")
try:
    loaded = load_dotenv(env_path)
    print(f"load_dotenv returned: {loaded}")
except Exception as e:
    print(f"load_dotenv failed: {str(e)}")

# Retrieve BOT_TOKEN with multiple fallbacks
BOT_TOKEN = os.getenv('BOT_TOKEN')
print(f"Initial BOT_TOKEN from environment: {BOT_TOKEN}")

if not BOT_TOKEN:
    print("Attempting to load BOT_TOKEN from .env file directly...")
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('BOT_TOKEN='):
                    # DO NOT INPUT TOKEN HERE DIRECTLY. PLACE IT IN .env FILE AS: BOT_TOKEN=your_token_here
                    BOT_TOKEN = line.split('=')[1]
                    print(f"Loaded BOT_TOKEN from .env file: {BOT_TOKEN}")
                    break
            else:
                print("No BOT_TOKEN found in .env file.")
    except FileNotFoundError:
        print(f".env file not found at: {env_path}")
    except Exception as e:
        print(f"Error reading .env file: {str(e)}")

    if not BOT_TOKEN:
        print("Checking system environment for BOT_TOKEN...")
        BOT_TOKEN = os.getenv('BOT_TOKEN', None)  # Double-check system environment
        if not BOT_TOKEN:
            raise ValueError("BOT_TOKEN environment variable not set. Please check .env file, ensure correct encoding, or set it manually in the environment.")

# Initialize bot
try:
    bot = telebot.TeleBot(BOT_TOKEN)
    print("Bot initialized successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to initialize bot: {str(e)}")

# Load the mine model
try:
    mine_model = load_model(os.path.join(current_dir, 'mines_tile_model.h5'))
    print("Mine model loaded successfully.")
except Exception as e:
    raise FileNotFoundError(f"Failed to load mines_tile_model.h5: {str(e)}")

def process_image(file_path):
    """
    Process the uploaded image into a 5x5 grid of 32x32 tiles and predict mines/safe.
    """
    try:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Failed to read image file.")
        img = cv2.resize(img, (350, 350))  # 5x5 grid with 70x70 tiles
        tiles = []
        for i in range(5):
            for j in range(5):
                tile = img[i*70:(i+1)*70, j*70:(j+1)*70]
                tile = cv2.resize(tile, (32, 32)) / 255.0
                tiles.append(tile)
        tiles = np.array(tiles)

        mine_preds = mine_model.predict(tiles, verbose=0)
        mine_map = (mine_preds < 0.5).astype(int)  # 0 = safe, 1 = mine
        number_map = np.zeros(25, dtype=int)  # Placeholder until number_model.h5 is trained
        return mine_map, number_map
    except Exception as e:
        raise RuntimeError(f"Error processing image: {str(e)}")

def deduce_mines(mine_map, number_map):
    """
    Deduce the grid state based on mine predictions.
    """
    try:
        grid = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if mine_map[idx] == 0:  # Safe tile
                    grid[i, j] = number_map[idx]  # Placeholder (0) for now
                else:  # Mine tile
                    grid[i, j] = -1  # Represent mines as -1
        return grid
    except Exception as e:
        raise RuntimeError(f"Error deducing mines: {str(e)}")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    """
    Handle uploaded photo, process the 5x5 grid, and store safe/mine positions in lists.
    """
    try:
        print("Received photo, processing...")
        file_id = message.photo[-1].file_id
        file = bot.get_file(file_id)
        downloaded_file = bot.download_file(file.file_path)
        with open('image.png', 'wb') as new_file:
            new_file.write(downloaded_file)
        
        mine_map, number_map = process_image('image.png')
        grid = deduce_mines(mine_map, number_map)
        
        safe_positions = []
        mine_positions = []
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if mine_map[idx] == 0:  # Safe tile
                    safe_positions.append([i, j])
                else:  # Mine tile
                    mine_positions.append([i, j])
        
        response = f"Grid Analysis (5x5):\nSafe tiles: {len(safe_positions)}\nMines: {len(mine_positions)}\n"
        for pos in safe_positions:
            response += f"Position {pos}: Safe (Probability: 1.00)\n"
        for pos in mine_positions:
            response += f"Position {pos}: Mine (Probability: 0.00)\n"
        response += f"Deduced grid:\n{grid}"
        bot.reply_to(message, response)
        print("Photo processed and response sent.")
    except Exception as e:
        bot.reply_to(message, f"Error processing photo: {str(e)}")
        print(f"Error in handle_photo: {str(e)}")

@@bot.message_handler(commands=['start'])
def send_welcome(message):
    try:
        print(f"Received /start from user {message.from_user.username} (ID: {message.from_user.id}) at {message.date}")
        bot.reply_to(message, "Welcome! Upload a 5x5 Mines game board image to analyze.")
        print("Sent welcome message.")
    except Exception as e:
        print(f"Error in send_welcome: {str(e)}")

@bot.message_handler(commands=['predict'])
def handle_predict(message):
    try:
        args = message.text.split()[1:]
        if not args:
            bot.reply_to(message, "Usage: /predict <number> (e.g., /predict 3)")
            print("Predict command received with no argument.")
            return
        try:
            num = int(args[0])
            bot.reply_to(message, f"Received /predict {num}. This feature is under development. Please upload a 5x5 image for analysis.")
            print(f"Predict command received with argument: {num}")
        except ValueError:
            bot.reply_to(message, "Please provide a valid number after /predict.")
            print("Invalid number argument for /predict.")
    except Exception as e:
        print(f"Error in handle_predict: {str(e)}")

if __name__ == "__main__":
    print("Starting bot polling and Flask app...")
    try:
        import threading
        def polling_thread():
            try:
                print("Starting bot polling...")
                bot.polling(none_stop=True)
            except Exception as e:
                print(f"Polling error: {str(e)}")
        threading.Thread(target=polling_thread, daemon=True).start()
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"Application failed: {str(e)}")

# Minimal Flask route to satisfy Render Web Service requirements
@app.route('/')
def health_check():
    return "Bot is running", 200

# Run the Flask app with bot polling in a separate thread
if __name__ == "__main__":
    print("Starting bot polling and Flask app...")
    try:
        # Start bot polling in a background thread
        import threading
        def polling_thread():
            bot.polling(none_stop=True)
        threading.Thread(target=polling_thread, daemon=True).start()
        
        # Run Flask app on the specified port
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"Application failed: {str(e)}")