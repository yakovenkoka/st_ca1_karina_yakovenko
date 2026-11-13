import matplotlib.pyplot as plt
import numpy as np
from data_loader import *

# Temporary test: both datasets load and the images look right
c10_names = load_cifar10_label_names()
fine_names, coarse_names = load_cifar100_label_names()
X10, y10 = load_cifar10('train')
X10_test, y10_test = load_cifar10('test')
X100, y100_fine, y100_coarse = load_cifar100('train')
X100_test, _, _ = load_cifar100('test')
c10_names = load_cifar10_label_names()
fine_names, coarse_names = load_cifar100_label_names()

print("CIFAR-10  train", X10.shape, "test", X10_test.shape, "classes", len(c10_names))
print("CIFAR-100 train", X100.shape, "test", X100_test.shape,
      "fine classes", len(fine_names), "coarse classes", len(coarse_names))

X10, y10 = filter_cifar10(X10, y10, c10_names)
X100, y100 = filter_cifar100(X100, y100_fine, y100_coarse, fine_names, coarse_names)
print("CIFAR-10 kept:", X10.shape, "CIFAR-100 kept:", X100.shape)
labels, counts = np.unique(np.concatenate((y10, y100)), return_counts=True)
for label, count in zip(labels, counts):
    print(label, CLASS_NAMES[label], count)
assert len(labels) == 24

trees = X100[y100 == CLASS_NAMES.index('trees')]
plt.imshow(trees[0])
plt.title('trees')
plt.show()