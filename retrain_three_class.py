import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Data generators with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    classes=['safe', 'mine', 'unrevealed_cropped']  # Explicitly define class order
)

validation_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    classes=['safe', 'mine', 'unrevealed_cropped']
)

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # Three classes: safe, mine, unrevealed
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping]
)

# Save the model
model.save('pregame_three_class_model.h5')
print("Model saved as pregame_three_class_model.h5")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.2f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.2f}")