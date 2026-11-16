import numpy as np
import matplotlib.pyplot as plt
import requests
import cv2
from PIL import Image
from preprocessing import preprocessing
from data_loader import CLASS_NAMES


def test_model_with_image(model, image_to_test):
    r = requests.get(image_to_test, stream=True, headers={'User-Agent': 'st_ca1_image_tester/1.0 (student project)'})
    r.raise_for_status()
    img = Image.open(r.raw).convert('RGB')
    # Preprocess before passing to model
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.show()
    img = img.reshape(1, 32, 32, 1)
    print("Predicted class: " + CLASS_NAMES[np.argmax(model.predict(img), axis=1)[0]])
