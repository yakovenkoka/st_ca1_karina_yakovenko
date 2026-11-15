from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

def leNet_model(num_classes):
    model = Sequential()
    model.add(Input(shape=(32, 32, 1)))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def modified_model(num_classes):
    model = Sequential()
    model.add(Input(shape=(32, 32, 1)))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_data_generator():
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
    return datagen

def explore_datagen(datagen, X_train, y_train):
    datagen.fit(X_train)
    batches = datagen.flow(X_train, y_train, batch_size=20)
    X_batch, y_batch = next(batches)
    fig, axs = plt.subplots(1, 15, figsize=(20, 5))
    fig.tight_layout()
    for i in range(15):
        axs[i].imshow(X_batch[i].reshape(32, 32))
        axs[i].axis('off')
    plt.show()

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, datagen):
    datagen.fit(X_train)
    history = model.fit(datagen.flow(X_train, y_train, batch_size=50), epochs=20, validation_data=(X_val, y_val), verbose=1, shuffle=1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.show()
    score = model.evaluate(X_test, y_test, verbose=1)
    print("Test score: ", score[0])
    print("Test accuracy: ", score[1])