from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2,
    rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
    horizontal_flip=True, zoom_range=0.2
)

train_generator = datagen.flow_from_directory('dataset'
    '.', target_size=(32, 32), batch_size=32, class_mode='categorical',
    subset='training', classes=['mine', 'safe', 'unrevealed_tiles']
)
validation_generator = datagen.flow_from_directory(
    '.', target_size=(32, 32), batch_size=32, class_mode='categorical',
    subset='validation', classes=['mine', 'safe', 'unrevealed_tiles']
)

model.fit(train_generator, epochs=15, validation_data=validation_generator)
model.save('pregame_model.h5')
print("Training done")