import pickle
import numpy as np

CIFAR10_DIR = 'data/cifar-10-batches-py'
CIFAR100_DIR = 'data/cifar-100-python'

CIFAR10_CLASSES = ['automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'truck']
CIFAR100_FINE_CLASSES = ['cattle', 'fox', 'baby', 'boy', 'girl', 'man', 'woman',
                         'rabbit', 'squirrel', 'bicycle', 'bus', 'motorcycle',
                         'pickup_truck', 'train', 'lawn_mower', 'tractor']
CIFAR100_COARSE_CLASSES = ['trees']
CLASS_NAMES = CIFAR10_CLASSES + CIFAR100_FINE_CLASSES + CIFAR100_COARSE_CLASSES

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def to_images(raw_data):
    images = raw_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(images)


def decode_names(names):
    return [name.decode('utf-8') for name in names]


def load_cifar10(split):
    if split == 'train':
        files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    else:
        files = ['test_batch']
    X_parts, y_parts = [], []
    for file in files:
        batch = unpickle(CIFAR10_DIR + '/' + file)
        X_parts.append(batch[b'data'])
        y_parts.append(batch[b'labels'])
    X = to_images(np.concatenate(X_parts))
    y = np.concatenate(y_parts)
    return X, y


def load_cifar10_label_names():
    meta = unpickle(CIFAR10_DIR + '/batches.meta')
    return decode_names(meta[b'label_names'])


def load_cifar100(split):
    data = unpickle(CIFAR100_DIR + '/' + split)
    X = to_images(data[b'data'])
    y_fine = np.array(data[b'fine_labels'])
    y_coarse = np.array(data[b'coarse_labels'])
    return X, y_fine, y_coarse


def load_cifar100_label_names():
    meta = unpickle(CIFAR100_DIR + '/meta')
    return decode_names(meta[b'fine_label_names']), decode_names(meta[b'coarse_label_names'])

def filter_cifar10(X, y, label_names):
    keep, new_y = [], []
    for i in range(len(y)):
        name = label_names[y[i]]
        if name in CIFAR10_CLASSES:
            keep.append(i)
            new_y.append(CLASS_NAMES.index(name))
    return X[keep], np.array(new_y)


def filter_cifar100(X, y_fine, y_coarse, fine_names, coarse_names):
    keep, new_y = [], []
    for i in range(len(y_fine)):
        fine_name = fine_names[y_fine[i]]
        coarse_name = coarse_names[y_coarse[i]]
        if fine_name in CIFAR100_FINE_CLASSES:
            keep.append(i)
            new_y.append(CLASS_NAMES.index(fine_name))
        elif coarse_name in CIFAR100_COARSE_CLASSES:
            keep.append(i)
            new_y.append(CLASS_NAMES.index(coarse_name))
    return X[keep], np.array(new_y)