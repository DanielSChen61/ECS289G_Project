import pandas as pd
import numpy as np
import pickle
import random

df = pd.read_csv("image_data/stage_2_train_labels.csv")
pneumonia_ids = open("image_data/pneumonia_ids.txt").read().splitlines()

extracted_tuples = []
box_dict = {}

for id in pneumonia_ids:
    box_dict[id] = []

for index, row in df.iterrows():
    if df.loc[index].at['patientId'] in pneumonia_ids:
        # box_dict[df.loc[index].at['patientId']].append((df.loc[index].at['x'], df.loc[index].at['y'], df.loc[index].at['width'], df.loc[index].at['height']))

        x = df.loc[index].at['x']
        y = df.loc[index].at['y']
        w = df.loc[index].at['width']
        h = df.loc[index].at['height']

        box_dict[df.loc[index].at['patientId']].append((x, y, w/2 + random.randrange(-5, 5), h/2 + random.randrange(-5, 5)))  # Top left box
        box_dict[df.loc[index].at['patientId']].append((x+w/2, y, w/2 + random.randrange(-5, 5), h/2 + random.randrange(-5, 5)))  # Top right box
        box_dict[df.loc[index].at['patientId']].append((x+w/2, y+h/2, w/2 + random.randrange(-5, 5), h/2 + random.randrange(-5, 5)))  # Bottom right box
        box_dict[df.loc[index].at['patientId']].append((x, y+h/2, w/2 + random.randrange(-5, 5), h/2 + random.randrange(-5, 5)))  # Bottom left box

with open('image_data/box_dict_pneumonia.pickle', 'wb') as handle:
    pickle.dump(box_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# last four before noise: [(723.75, 386.5), (831.25, 386.5), (831.25, 505.5), (723.75, 505.5)]