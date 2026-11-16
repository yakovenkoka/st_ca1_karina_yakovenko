import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def apply_preprocessing(X_train, X_val, X_test):
    X_train = np.array(list(map(preprocessing, X_train)))
    X_val = np.array(list(map(preprocessing, X_val)))
    X_test = np.array(list(map(preprocessing, X_test)))
    return X_train, X_val, X_test


def examine_random_training_image_after_preprocessing(X_train):
    plt.imshow(X_train[np.random.randint(0, len(X_train) - 1)])
    plt.axis("off")
    plt.show()


def reshape_for_cnn(X_train, X_val, X_test):
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
    X_val = X_val.reshape(X_val.shape[0], 32, 32, 1)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)
    return X_train, X_val, X_test


def one_hot_encode(num_classes, y_train, y_val, y_test):
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return y_train, y_val, y_test