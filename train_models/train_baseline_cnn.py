from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.metrics import Precision, Recall
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import glob
import random
import os


# Preprocessing the image dataset -----
train_set = []
test_set = []

for file in glob.glob('image_data/stage_2_train_normal_full/*.*'):
    image_bytes = tf.io.read_file(file)
    image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
    resized_image = tf.squeeze(tf.image.resize(image, size=(224, 224)), [0])
    resized_image = tf.image.grayscale_to_rgb(resized_image)
    resized_image = np.array(resized_image)
    train_set.append((resized_image, 'normal'))

for file in glob.glob('image_data/stage_2_train_pneumonia_full/*.*'):
    image_bytes = tf.io.read_file(file)
    image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
    resized_image = tf.squeeze(tf.image.resize(image, size=(224, 224)), [0])
    resized_image = tf.image.grayscale_to_rgb(resized_image)
    resized_image = np.array(resized_image)
    train_set.append((resized_image, 'pneumonia'))

random.shuffle(train_set)

train_data = list(zip(*train_set))[0]
train_labels = list(zip(*train_set))[1]

X_train = np.array(train_data).astype('float32') / 255

lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(train_labels))


# Build the model -----
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the first four convolution blocks in vgg16 and only make last block trainable
for layer in vgg.layers[:15]:
    layer.trainable = False

x = vgg.output
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(2, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=x)

# More hyperparameters
learning_rate = 7.5e-7
batch_size = 32
epochs = 40
optimizer = optimizers.Adam(learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[Precision(), Recall()])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
print("Training done...\n")

