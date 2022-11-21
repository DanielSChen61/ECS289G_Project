from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.metrics import Precision, Recall
from sklearn.preprocessing import LabelEncoder

import numpy as np
import glob
import random


# Preprocessing the image dataset -----
train_set = []
test_set = []

# Training dataset -----
for image in glob.glob("../image_data/chest_xray/train/NORMAL/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    train_set.append((image, 'normal'))


for image in glob.glob("../image_data/chest_xray/train/PNEUMONIA_bacteria/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    train_set.append((image, 'pneumonia_bacteria'))

for image in glob.glob("../image_data/chest_xray/train/PNEUMONIA_virus/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    train_set.append((image, 'pneumonia_virus'))

# Test dataset -----
for image in glob.glob("../image_data/chest_xray/test/NORMAL/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    test_set.append((image, 'normal'))

for image in glob.glob("../image_data/chest_xray/test/PNEUMONIA_bacteria/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    test_set.append((image, 'pneumonia_bacteria'))

for image in glob.glob("../image_data/chest_xray/test/PNEUMONIA_virus/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    test_set.append((image, 'pneumonia_virus'))

random.seed(1)
random.shuffle(train_set)
random.shuffle(test_set)

train_data = list(zip(*train_set))[0]
train_labels = list(zip(*train_set))[1]
test_data = list(zip(*test_set))[0]
test_labels = list(zip(*test_set))[1]

X_train = np.array(train_data).astype('float32') / 255
X_test = np.array(test_data).astype('float32') / 255

lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(train_labels))
y_test = to_categorical(lb.fit_transform(test_labels))


# Build the model -----
resnet = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in resnet.layers:
    layer.trainable = False

x = resnet.output
x = layers.Flatten()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(3, activation='softmax')(x)
model = Model(inputs=resnet.input, outputs=x)

# More hyperparameters
learning_rate = 7.5e-7
batch_size = 32
epochs = 35
optimizer = optimizers.Adam(learning_rate)
validation_split = 0.2

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[Precision(), Recall()])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
print("Training done...\n")

print("Results on Test Dataset")
results = model.evaluate(X_test, y_test, batch_size=batch_size)

