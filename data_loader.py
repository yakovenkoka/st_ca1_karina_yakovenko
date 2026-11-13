import pickle
import numpy as np
from sklearn.model_selection import train_test_split

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


def combine_datasets(X_a, y_a, X_b, y_b):
    X = np.concatenate((X_a, X_b))
    y = np.concatenate((y_a, y_b))
    return X, y


def split_validation(X, y, val_size=0.2):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=0, stratify=y)
    return X_train, y_train, X_val, y_val


def load_split(split, c10_names, fine_names, coarse_names):
    X10, y10 = load_cifar10(split)
    X100, y100_fine, y100_coarse = load_cifar100(split)
    print(split, "raw: CIFAR-10", X10.shape, "CIFAR-100", X100.shape)
    X10, y10 = filter_cifar10(X10, y10, c10_names)
    X100, y100 = filter_cifar100(X100, y100_fine, y100_coarse, fine_names, coarse_names)
    print(split, "filtered: CIFAR-10", X10.shape, "CIFAR-100", X100.shape)
    return combine_datasets(X10, y10, X100, y100)


def load_data():
    c10_names = load_cifar10_label_names()
    fine_names, coarse_names = load_cifar100_label_names()
    X_train, y_train = load_split('train', c10_names, fine_names, coarse_names)
    X_test, y_test = load_split('test', c10_names, fine_names, coarse_names)
    X_train, y_train, X_val, y_val = split_validation(X_train, y_train)
    return X_train, y_train, X_val, y_val, X_test, y_test


def check_data(X_train, y_train, X_val, y_val, X_test, y_test):
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    assert(X_train.shape[0] == y_train.shape[0]), "The number of training images does not match the number of labels"
    assert(X_val.shape[0] == y_val.shape[0]), "The number of validation images does not match the number of labels"
    assert(X_test.shape[0] == y_test.shape[0]), "The number of test images does not match the number of labels"
    assert(X_train.shape[1:] == (32, 32, 3)), "The dimensions of the training images are not 32 x 32 x 3"
    assert(X_val.shape[1:] == (32, 32, 3)), "The dimensions of the validation images are not 32 x 32 x 3"
    assert(X_test.shape[1:] == (32, 32, 3)), "The dimensions of the test images are not 32 x 32 x 3"