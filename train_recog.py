from tensorflow.keras.preprocessing.image import ImageDataGenerator # Generate batches of tensor image data with real-time data augmentation.
from tensorflow.keras.applications import MobileNetV2 # CNN architecture that works well w/mobile devices as well
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam # stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # Preprocesses a tensor or Numpy array encoding a batch of images.
from tensorflow.keras.preprocessing.image import img_to_array, load_img 
from tensorflow.keras.utils import to_categorical # converts a class vector (integers) to a binary class matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = 'lightgrey'

# init the initial learning rate & number of epochs to train for
init_lr = 1e-4
epochs = 20
bs = 32

DIRECTORY = r"data"
CATEGORIES = ["with_mask","without_mask","mask_weared_incorrect"]

print("[INFO] loading images...")

# get the list of images in our dataset directory, 
# then init the list of data and class images
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    # looping over image paths and prepocessing the images
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        image = load_img(img_path,target_size=(224,224)) # resize all images, target size
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)
        
# one-hot encoding on the labels (1 and 0) binary format
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# convert data to arrays b/c hidden layers only accept this format
data = np.array(data,dtype="float32")
labels = np.array(labels)

# split the data
X_train, y_train, X_test, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)