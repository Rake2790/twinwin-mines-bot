from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    '.', target_size=(32, 32), batch_size=32, class_mode='categorical',
    subset='training', classes=['mine', 'safe', 'unrevealed_tiles']
)
validation_generator = datagen.flow_from_directory(
    '.', target_size=(32, 32), batch_size=32, class_mode='categorical',
    subset='validation', classes=['mine', 'safe', 'unrevealed_tiles']
)

model.fit(train_generator, epochs=5, validation_data=validation_generator)
model.save('pregame_model.h5')
print("Training done")