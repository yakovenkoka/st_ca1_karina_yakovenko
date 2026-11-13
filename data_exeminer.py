import numpy as np
import matplotlib.pyplot as plt

def examine_typical_image(X_train, y_train, class_names):
    pre_img = X_train[500]
    plt.imshow(pre_img)
    plt.title(class_names[y_train[500]])
    plt.axis("off")
    plt.show()


def plot_sample_distribution(class_names, num_of_each_class):
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, len(class_names)), num_of_each_class)
    plt.xticks(range(0, len(class_names)), class_names, rotation=90)
    plt.title("Distribution of the training set")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.show()


def show_training_samples(class_names, X_train, y_train):
    num_samples = []
    cols = 5
    num_classes = len(class_names)

    fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 25))
    fig.tight_layout()
    for i in range(cols):
        for j, name in enumerate(class_names):
            x_selected = X_train[y_train==j]
            axs[j][i].imshow(x_selected[np.random.randint(0, len(x_selected)-1), :, :])
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(str(j) + " - " + name, fontsize=8)
                num_samples.append(len(x_selected))
    plt.show()
    return num_samples
