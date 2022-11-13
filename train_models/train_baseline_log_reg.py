from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import glob
import random


# Preprocessing the image dataset -----
train_set = []
test_set = []

y_train = []
y_test = []

# normal : 0
# pneumonia : 1

for file in glob.glob('image_data/stage_2_train_normal_full/*.*'):
    image_bytes = tf.io.read_file(file)
    image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
    resized_image = tf.squeeze(tf.image.resize(image, size=(224, 224)), [0])
    resized_image = tf.image.grayscale_to_rgb(resized_image)
    resized_image = np.array(resized_image)
    train_set.append((resized_image, 0))

for file in glob.glob('image_data/stage_2_train_pneumonia_full/*.*'):
    image_bytes = tf.io.read_file(file)
    image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
    resized_image = tf.squeeze(tf.image.resize(image, size=(224, 224)), [0])
    resized_image = tf.image.grayscale_to_rgb(resized_image)
    resized_image = np.array(resized_image)
    train_set.append((resized_image, 1))

random.shuffle(train_set)

train_data = list(zip(*train_set))[0]
train_labels = list(zip(*train_set))[1]

train_data = np.array(train_data).astype('float32') / 255

X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=1)
train_len = len(X_train)
test_len = len(X_test)

X_train = X_train.reshape(train_len, 224 * 224 * 3)
X_test = X_test.reshape(test_len, 224 * 224 * 3)

model = LogisticRegression().fit(X_train, y_train)
predictions = model.predict(X_test)

precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')

print(f"Precision: {precision}")
print(f"Recall: {recall}")



