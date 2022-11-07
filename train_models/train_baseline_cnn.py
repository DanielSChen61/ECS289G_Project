from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall
from sklearn.preprocessing import LabelEncoder

import numpy as np
import glob


# Preprocessing the image dataset -----

train_data = []
val_data = []
test_data = []

train_labels = []
val_labels = []
test_labels = []

# Training dataset -----
for image in glob.glob("image_data/chest_xray/train/NORMAL/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    train_data.append(image)
    train_labels.append('normal')

for image in glob.glob("image_data/chest_xray/train/PNEUMONIA_bacteria/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    train_data.append(image)
    train_labels.append('pneumonia_bacteria')

for image in glob.glob("image_data/chest_xray/train/PNEUMONIA_virus/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    train_data.append(image)
    train_labels.append('pneumonia_virus')

# Validation dataset -----
for image in glob.glob("image_data/chest_xray/val/NORMAL/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    val_data.append(image)
    val_labels.append('normal')

for image in glob.glob("image_data/chest_xray/val/PNEUMONIA_bacteria/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    val_data.append(image)
    val_labels.append('pneumonia_bacteria')

for image in glob.glob("image_data/chest_xray/val/PNEUMONIA_virus/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    val_data.append(image)
    val_labels.append('pneumonia_virus')

# Test dataset -----
for image in glob.glob("image_data/chest_xray/test/NORMAL/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    test_data.append(image)
    test_labels.append('normal')

for image in glob.glob("image_data/chest_xray/test/PNEUMONIA_bacteria/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    test_data.append(image)
    test_labels.append('pneumonia_bacteria')

for image in glob.glob("image_data/chest_xray/test/PNEUMONIA_virus/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(224, 224))
    image = np.array(image)
    test_data.append(image)
    test_labels.append('pneumonia_virus')


X_train = np.array(train_data).astype('float32') / 255
X_val = np.array(val_data).astype('float32') / 255
X_test = np.array(test_data).astype('float32') / 255

lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(train_labels))
y_val = to_categorical(lb.fit_transform(val_labels))
y_test = to_categorical(lb.fit_transform(test_labels))



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
x = layers.Dense(3, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=x)

# Hyperparameters
learning_rate = 7.5e-7
batch_size = 32
epochs = 1 #20
optimizer = optimizers.Adam(learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[Precision(), Recall()])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
print("Training done...")

print("Results on Test Dataset")
results = model.evaluate(X_test, y_test, batch_size=batch_size)

