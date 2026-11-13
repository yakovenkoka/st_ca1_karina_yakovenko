import matplotlib.pyplot as plt
import numpy as np
from data_loader import *

X_train, y_train, X_val, y_val, X_test, y_test = load_data()
check_data(X_train, y_train, X_val, y_val, X_test, y_test)