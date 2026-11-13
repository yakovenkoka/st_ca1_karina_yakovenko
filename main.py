import matplotlib.pyplot as plt
import numpy as np
from data_loader import *
from preprocessing import *
from data_exeminer import *

np.random.seed(0)

X_train, y_train, X_val, y_val, X_test, y_test = load_data()
check_data(X_train, y_train, X_val, y_val, X_test, y_test)

print(list(enumerate(CLASS_NAMES)))
num_of_each_class = show_training_samples(CLASS_NAMES, X_train, y_train)
plot_sample_distribution(CLASS_NAMES, num_of_each_class)
examine_typical_image(X_train, y_train, CLASS_NAMES)

X_train, X_val, X_test = apply_preprocessing(X_train, X_val, X_test)
examine_random_training_image_after_preprocessing(X_train)
X_train, X_val, X_test = reshape_for_cnn(X_train, X_val, X_test)
y_train, y_val, y_test = one_hot_encode(len(CLASS_NAMES), y_train, y_val, y_test)
print(X_train.shape, X_val.shape, X_test.shape)