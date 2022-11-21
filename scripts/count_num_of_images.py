from tensorflow.keras.preprocessing.image import load_img

import pandas as pd
import numpy as np
import glob


train_set = []
test_set = []

train_normal_count = 0
train_bacteria_count = 0
train_virus_count = 0

test_normal_count = 0
test_bacteria_count = 0
test_virus_count = 0


# Training dataset -----
for image in glob.glob("image_data/chest_xray/train/NORMAL/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(229, 229))
    image = np.array(image)
    train_set.append((image, 'normal'))
    train_normal_count += 1


for image in glob.glob("image_data/chest_xray/train/PNEUMONIA_bacteria/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(229, 229))
    image = np.array(image)
    train_set.append((image, 'pneumonia_bacteria'))
    train_bacteria_count += 1

for image in glob.glob("image_data/chest_xray/train/PNEUMONIA_virus/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(229, 229))
    image = np.array(image)
    train_set.append((image, 'pneumonia_virus'))
    train_virus_count += 1

# Test dataset -----
for image in glob.glob("image_data/chest_xray/test/NORMAL/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(229, 229))
    image = np.array(image)
    test_set.append((image, 'normal'))
    test_normal_count += 1

for image in glob.glob("image_data/chest_xray/test/PNEUMONIA_bacteria/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(229, 229))
    image = np.array(image)
    test_set.append((image, 'pneumonia_bacteria'))
    test_bacteria_count += 1

for image in glob.glob("image_data/chest_xray/test/PNEUMONIA_virus/*.*"):
    image = load_img(image, color_mode='rgb', target_size=(229, 229))
    image = np.array(image)
    test_set.append((image, 'pneumonia_virus'))
    test_virus_count += 1

train_total_count = train_normal_count + train_bacteria_count + train_virus_count
test_total_count = test_normal_count + test_bacteria_count + test_virus_count

print(f"Train (Normal) Size: {train_normal_count}")
print(f"Train (Bacteria) Size: {train_bacteria_count}")
print(f"Train (Virus) Size: {train_virus_count}")
print(f"Train Total Size: {train_total_count}")
print("------------------------")

print(f"Test (Normal) Size: {test_normal_count}")
print(f"Test (Bacteria) Size: {test_bacteria_count}")
print(f"Test (Virus) Size: {test_virus_count}")
print(f"Test Total Size: {test_total_count}")
