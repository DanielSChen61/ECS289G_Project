from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.preprocessing.image import load_img

import numpy as np
import glob

# Preprocessing the image dataset -----
train_data = []
val_data = []
test_data = []

y_train = []
y_val = []
y_test = []

# normal : 1
# pneumonia_bacteria : 2
# pneumonia_virus : 3

# Training dataset -----
for image in glob.glob("image_data/chest_xray/train/NORMAL/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    train_data.append(image)
    y_train.append(1)

for image in glob.glob("image_data/chest_xray/train/PNEUMONIA_bacteria/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    train_data.append(image)
    y_train.append(2)

for image in glob.glob("image_data/chest_xray/train/PNEUMONIA_virus/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    train_data.append(image)
    y_train.append(3)

# Validation dataset -----
for image in glob.glob("image_data/chest_xray/val/NORMAL/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    val_data.append(image)
    y_val.append(1)

for image in glob.glob("image_data/chest_xray/val/PNEUMONIA_bacteria/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    val_data.append(image)
    y_val.append(2)

for image in glob.glob("image_data/chest_xray/val/PNEUMONIA_virus/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    val_data.append(image)
    y_val.append(3)

# Test dataset -----
for image in glob.glob("image_data/chest_xray/test/NORMAL/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    test_data.append(image)
    y_test.append(1)

for image in glob.glob("image_data/chest_xray/test/PNEUMONIA_bacteria/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    test_data.append(image)
    y_test.append(2)

for image in glob.glob("image_data/chest_xray/test/PNEUMONIA_virus/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    test_data.append(image)
    y_test.append(3)


X_train = np.array(train_data).astype('float32') / 255
X_val = np.array(val_data).astype('float32') / 255
X_test = np.array(test_data).astype('float32') / 255

X_train = X_train.reshape(5208, 224 * 224 * 3)
X_val = X_val.reshape(24, 224 * 224 * 3)
X_test = X_test.reshape(624, 224 * 224 * 3)


model = LogisticRegression().fit(X_train, y_train)
predictions = model.predict(X_test)

precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
