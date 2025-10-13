import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    classes=['safe', 'mine', 'unrevealed_tiles']
)

validation_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    classes=['safe', 'mine', 'unrevealed_tiles']
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: safe, mine, unrevealed
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

model.save('pregame_model.h5')
print("Model saved as pregame_model.h5")