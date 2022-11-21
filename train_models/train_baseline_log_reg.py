from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.preprocessing.image import load_img

import numpy as np
import glob
import random


# Preprocessing the image dataset -----
train_set = []
test_set = []

y_train = []
y_test = []

# normal : 0
# pneumonia_bacteria : 1
# pneumonia_virus : 2

for image in glob.glob("../image_data/chest_xray/train/NORMAL/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    train_set.append((image, 0))


for image in glob.glob("../image_data/chest_xray/train/PNEUMONIA_bacteria/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    train_set.append((image, 1))

for image in glob.glob("../image_data/chest_xray/train/PNEUMONIA_virus/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    train_set.append((image, 2))

# Test dataset -----
for image in glob.glob("../image_data/chest_xray/test/NORMAL/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    test_set.append((image, 0))

for image in glob.glob("../image_data/chest_xray/test/PNEUMONIA_bacteria/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    test_set.append((image, 1))

for image in glob.glob("../image_data/chest_xray/test/PNEUMONIA_virus/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    test_set.append((image, 2))

random.seed(1)
random.shuffle(train_set)
random.shuffle(test_set)

train_data = list(zip(*train_set))[0]
train_labels = list(zip(*train_set))[1]
test_data = list(zip(*test_set))[0]
test_labels = list(zip(*test_set))[1]

X_train = np.array(train_data).astype('float32') / 255
X_test = np.array(test_data).astype('float32') / 255

X_train = X_train.reshape(5232, 224 * 224 * 3)
X_test = X_test.reshape(624, 224 * 224 * 3)

y_train = train_labels
y_test = test_labels

model = LogisticRegression().fit(X_train, y_train)
predictions = model.predict(X_test)

precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
