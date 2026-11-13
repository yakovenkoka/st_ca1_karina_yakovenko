import matplotlib.pyplot as plt
from data_loader import *

# Temporary test: both datasets load and the images look right
X10, y10 = load_cifar10('train')
X10_test, y10_test = load_cifar10('test')
X100, y100_fine, y100_coarse = load_cifar100('train')
X100_test, _, _ = load_cifar100('test')
c10_names = load_cifar10_label_names()
fine_names, coarse_names = load_cifar100_label_names()

print("CIFAR-10  train", X10.shape, "test", X10_test.shape, "classes", len(c10_names))
print("CIFAR-100 train", X100.shape, "test", X100_test.shape,
      "fine classes", len(fine_names), "coarse classes", len(coarse_names))

fig, axs = plt.subplots(1, 2)
axs[0].imshow(X10[0])
axs[0].set_title(c10_names[y10[0]])
axs[1].imshow(X100[0])
axs[1].set_title(fine_names[y100_fine[0]] + " / " + coarse_names[y100_coarse[0]])
plt.show()