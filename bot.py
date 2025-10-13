import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import telebot
from dotenv import load_dotenv
from flask import Flask, request

app = Flask(__name__)
port = int(os.getenv("PORT", 10000))

# Load environment variables
current_dir = os.path.dirname(__file__)
env_path = os.path.join(current_dir, '.env')
print(f"Attempting to load .env from: {env_path}")
try:
    loaded = load_dotenv(env_path)
    print(f"load_dotenv returned: {loaded}")
except Exception as e:
    print(f"load_dotenv failed: {str(e)}")

BOT_TOKEN = os.getenv('BOT_TOKEN')
print(f"Initial BOT_TOKEN from environment: {BOT_TOKEN}")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not set. Configure it in Render's Environment tab.")

try:
    bot = telebot.TeleBot(BOT_TOKEN)
    print("Bot initialized successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to initialize bot: {str(e)}")

# Load the pre-game model
try:
    pregame_model = load_model(os.path.join(current_dir, 'pregame_model.h5'))
    print("Pre-game model loaded successfully.")
except Exception as e:
    raise FileNotFoundError(f"Failed to load pregame_model.h5: {str(e)}")

def process_image(file_path):
    try:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Failed to read image file.")
        img = cv2.resize(img, (350, 350))
        tiles = []
        for i in range(5):
            for j in range(5):
                tile = img[i*70:(i+1)*70, j*70:(j+1)*70]
                tile = cv2.resize(tile, (32, 32)) / 255.0
                tiles.append(tile)
        tiles = np.array(tiles)
        preds = pregame_model.predict(tiles, verbose=0)
        return preds  # Shape: (25, 3) with probabilities for [safe, mine, unrevealed]
    except Exception as e:
        raise RuntimeError(f"Error processing image: {str(e)}")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        print("Received photo, processing...")
        file_id = message.photo[-1].file_id
        file = bot.get_file(file_id)
        downloaded_file = bot.download_file(file.file_path)
        with open('image.png', 'wb') as new_file:
            new_file.write(downloaded_file)
        
        preds = process_image('image.png')
        classifications = np.argmax(preds, axis=1)
        unrevealed_positions = [[i, j] for i in range(5) for j in range(5) if classifications[i*5+j] == 2]
        mine_positions = [[i, j] for i in range(5) for j in range(5) if classifications[i*5+j] == 1]
        safe_positions = [[i, j] for i in range(5) for j in range(5) if classifications[i*5+j] == 0]

        response = f"Pre-Game Prediction (5x5):\nUnrevealed tiles: {len(unrevealed_positions)}\nMines: {len(mine_positions)}\nSafe tiles: {len(safe_positions)}\n"
        for pos in unrevealed_positions:
            prob = preds[pos[0]*5 + pos[1], 2]
            response += f"Position {pos}: Unrevealed (Probability: {prob:.2f})\n"
        for pos in mine_positions:
            prob = preds[pos[0]*5 + pos[1], 1]
            response += f"Position {pos}: Mine (Probability: {prob:.2f})\n"
        for pos in safe_positions:
            prob = preds[pos[0]*5 + pos[1], 0]
            response += f"Position {pos}: Safe (Probability: {prob:.2f})\n"
        bot.reply_to(message, response)
        print("Photo processed and response sent.")
    except Exception as e:
        bot.reply_to(message, f"Error processing photo: {str(e)}")
        print(f"Error in handle_photo: {str(e)}")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    try:
        print(f"Received /start from user {message.from_user.username} (ID: {message.from_user.id}) at {message.date}")
        bot.reply_to(message, "Welcome! Upload a 5x5 Mines game board image for pre-game prediction (unrevealed, mines, or safe tiles).")
        print("Sent welcome message.")
    except Exception as e:
        print(f"Error in send_welcome: {str(e)}")

# Webhook route to handle Telegram updates
@app.route('/webhook', methods=['POST'])
def webhook():
    update = request.get_json()
    if update:
        bot.process_new_updates([telebot.types.Update.de_json(update)])
    return '', 200

@app.route('/')
def health_check():
    return "Bot is running", 200

if __name__ == "__main__":
    print("Setting webhook and running Flask app...")
    bot.remove_webhook()
    bot.set_webhook(url="https://twinwin-mines-bot.onrender.com/webhook")
    app.run(host='0.0.0.0', port=port)