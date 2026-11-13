import matplotlib.pyplot as plt
import numpy as np
from data_loader import *
from preprocessing import *

X_train, y_train, X_val, y_val, X_test, y_test = load_data()
check_data(X_train, y_train, X_val, y_val, X_test, y_test)

X_train, X_val, X_test = apply_preprocessing(X_train, X_val, X_test)
X_train, X_val, X_test = reshape_for_cnn(X_train, X_val, X_test)
y_train, y_val, y_test = one_hot_encode(len(CLASS_NAMES), y_train, y_val, y_test)
examine_random_training_image_after_preprocessing(X_train)
print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, X_train.min(), X_train.max())