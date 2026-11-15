import matplotlib.pyplot as plt
import numpy as np
from data_loader import *
from preprocessing import *
from data_exeminer import *
from model import *
from tester import *

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

model = leNet_model(len(CLASS_NAMES))
print(model.summary())
datagen = create_data_generator()
explore_datagen(datagen, X_train, y_train)
evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, datagen)

# CIFAR-10 
url_horse = "https://c7.alamy.com/comp/AM9N2P/horse-standing-alone-in-front-of-a-red-wall-AM9N2P.jpg"
url_dog = "https://c7.alamy.com/comp/GDA2T1/yellow-labrador-retriever-portrait-of-adult-GDA2T1.jpg"
url_truck = "https://c7.alamy.com/comp/PG5AHJ/close-up-of-speeding-lorry-cargo-truck-with-trailer-on-the-zagreb-motor-way-bypass-in-zagreb-croatia-PG5AHJ.jpg"
test_model_with_image(model, url_horse)
test_model_with_image(model, url_dog)
test_model_with_image(model, url_truck)

# CIFAR-100
url_fox = "https://c7.alamy.com/comp/2BW05MG/red-fox-looking-back-over-shoulder-on-a-meadow-in-autumn-nature-2BW05MG.jpg"
url_tractor = "https://c7.alamy.com/comp/CWD4M3/tractor-during-hay-making-in-oxfordshire-field-CWD4M3.jpg"
url_squirrel = "https://c7.alamy.com/comp/2NMX340/a-close-up-profile-portrait-of-a-red-squirrel-with-a-hazelnut-resting-on-an-old-log-taken-in-winter-still-showing-the-ear-tufts-2NMX340.jpg"
test_model_with_image(model, url_fox)
test_model_with_image(model, url_tractor)
test_model_with_image(model, url_squirrel)

# Combined
url_bird = "https://c7.alamy.com/comp/2PP6W5X/robin-erithacus-rubecula-close-up-of-a-single-specimen-of-the-common-garden-and-woodland-bird-perched-on-the-branch-of-a-tree-2PP6W5X.jpg"
url_deer = "https://c7.alamy.com/comp/EECHXE/red-deer-stag-EECHXE.jpg"
url_cat = "https://c7.alamy.com/comp/2YFRG73/young-red-house-cat-sitting-up-facing-front-looking-straight-to-camera-isolated-on-a-white-background-2YFRG73.jpg"
url_baby = "https://c7.alamy.com/comp/3E5EHG0/sleeping-newborn-posed-in-woven-basket-with-soft-blanket-and-wooden-blocks-neutral-rustic-styling-ideal-for-professional-baby-photography-portfolios-3E5EHG0.jpg"
url_tree = "https://c7.alamy.com/comp/2JR4FTB/lone-oak-tree-in-empty-harvested-field-under-blue-sky-2JR4FTB.jpg"
url_motorcycle = "https://c7.alamy.com/comp/2EFHXBJ/side-profile-of-a-person-riding-a-motorcycle-2EFHXBJ.jpg"
test_model_with_image(model, url_bird)
test_model_with_image(model, url_deer)
test_model_with_image(model, url_cat)
test_model_with_image(model, url_baby)
test_model_with_image(model, url_tree)
test_model_with_image(model, url_motorcycle)
