from PIL import Image

import glob


train_smallest_width = 99999999999
train_smallest_height = 99999999999

train_largest_width = 0
train_largest_height = 0

test_smallest_width = 99999999999
test_smallest_height = 99999999999

test_largest_width = 0
test_largest_height = 0

train_image_paths = glob.glob("../image_data/chest_xray/train/NORMAL/*.*")
train_image_paths += glob.glob("../image_data/chest_xray/train/PNEUMONIA_bacteria/*.*")
train_image_paths += glob.glob("../image_data/chest_xray/train/PNEUMONIA_virus/*.*")

test_image_paths = glob.glob("../image_data/chest_xray/test/NORMAL/*.*")
test_image_paths += glob.glob("../image_data/chest_xray/test/PNEUMONIA_bacteria/*.*")
test_image_paths += glob.glob("../image_data/chest_xray/test/PNEUMONIA_virus/*.*")

for image in train_image_paths:
    opened = Image.open(image)
    width, height = opened.size

    if width < train_smallest_width:
        train_smallest_width = width
    
    if height < train_smallest_height:
        train_smallest_height = height
    

    if width > train_largest_width:
        train_largest_width = width

    if height > train_largest_height:
        train_largest_height = height


for image in test_image_paths:
    opened = Image.open(image)
    width, height = opened.size

    if width < test_smallest_width:
        test_smallest_width = width
    
    if height < test_smallest_height:
        test_smallest_height = height
    
    if width > test_largest_width:
        test_largest_width = width

    if height > test_largest_height:
        test_largest_height = height


print(f'(Train) Smallest width: {train_smallest_width}')
print(f'(Train) Smallest height: {train_smallest_height}')
print('----------------------')
print(f'(Test) Smallest width: {test_smallest_width}')
print(f'(Test) Smallest height: {test_smallest_height}')
print('----------------------')
print(f'(Train) Largest width: {train_largest_width}')
print(f'(Train) Largest height: {train_largest_height}')
print('----------------------')
print(f'(Test) Largest width: {test_largest_width}')
print(f'(Test) Largest height: {test_largest_height}')