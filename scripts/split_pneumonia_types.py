import os
import glob
import shutil

# Separate different types of Pneumonia into separate folders for training data -----
os.mkdir('./image_data/chest_xray/train/PNEUMONIA_virus')
os.mkdir('./image_data/chest_xray/train/PNEUMONIA_bacteria')

train_virus_list = glob.glob('./image_data/chest_xray/train/PNEUMONIA/*virus*')
train_bacteria_list = glob.glob('./image_data/chest_xray/train/PNEUMONIA/*bacteria*')

for file in train_virus_list:
    cleaned_filename = os.path.basename(file)
    src = os.path.join('./image_data/chest_xray/train/PNEUMONIA', cleaned_filename)
    target = os.path.join('./image_data/chest_xray/train/PNEUMONIA_virus', cleaned_filename)
    shutil.copyfile(src, target)

for file in train_bacteria_list:
    cleaned_filename = os.path.basename(file)
    src = os.path.join('./image_data/chest_xray/train/PNEUMONIA', cleaned_filename)
    target = os.path.join('./image_data/chest_xray/train/PNEUMONIA_bacteria', cleaned_filename)
    shutil.copyfile(src, target)


# Separate different types of Pneumonia into separate folders for test data -----
os.mkdir('./image_data/chest_xray/test/PNEUMONIA_virus')
os.mkdir('./image_data/chest_xray/test/PNEUMONIA_bacteria')

test_virus_list = glob.glob('./image_data/chest_xray/test/PNEUMONIA/*virus*')
test_bacteria_list = glob.glob('./image_data/chest_xray/test/PNEUMONIA/*bacteria*')

for file in test_virus_list:
    cleaned_filename = os.path.basename(file)
    src = os.path.join('./image_data/chest_xray/test/PNEUMONIA', cleaned_filename)
    target = os.path.join('./image_data/chest_xray/test/PNEUMONIA_virus', cleaned_filename)
    shutil.copyfile(src, target)

for file in test_bacteria_list:
    cleaned_filename = os.path.basename(file)
    src = os.path.join('./image_data/chest_xray/test/PNEUMONIA', cleaned_filename)
    target = os.path.join('./image_data/chest_xray/test/PNEUMONIA_bacteria', cleaned_filename)
    shutil.copyfile(src, target)

