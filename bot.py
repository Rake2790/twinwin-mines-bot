import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import telebot

# Initialize bot with your token (replace 'YOUR_BOT_TOKEN' with your actual token from BotFather)
bot = telebot.TeleBot('8408573813:AAHnb6z4Az9R5TfAFNYc9kTR_YyAju9I2tE')

# Load only the mine model
mine_model = load_model(os.path.join(os.path.dirname(__file__), 'mines_tile_model.h5'))

def process_image(file):
    """
    Process the uploaded image into a 5x5 grid of 32x32 tiles and predict mines/safe.
    """
    img = cv2.imread(file)
    img = cv2.resize(img, (350, 350))  # Assume 5x5 grid with 70x70 tiles
    tiles = []
    for i in range(5):
        for j in range(5):
            tile = img[i*70:(i+1)*70, j*70:(j+1)*70]
            tile = cv2.resize(tile, (32, 32)) / 255.0
            tiles.append(tile)
    tiles = np.array(tiles)

    mine_preds = mine_model.predict(tiles)
    mine_map = (mine_preds < 0.5).astype(int)  # 0 = safe, 1 = mine
    number_map = np.zeros(25, dtype=int)  # Placeholder, all zeros if no number model
    return mine_map, number_map

def deduce_mines(mine_map, number_map):
    """
    Deduce the grid state based on mine predictions (numbers ignored for now).
    """
    grid = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if mine_map[idx] == 0:  # Safe tile
                grid[i, j] = number_map[idx]  # Will be 0 if no number model
            else:  # Mine tile
                grid[i, j] = -1  # Represent mines as -1
    return grid

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    """
    Handle uploaded photo, process the 5x5 grid, and store safe/mine positions in lists.
    """
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

@bot.message_handler(commands=['start'])
def send_welcome(message):
    """
    Handle /start command with a welcome message.
    """
    bot.reply_to(message, "Welcome! Upload a 5x5 Mines game board image to analyze.")

if __name__ == "__main__":
    bot.polling()