import pandas as pd
import numpy as np
import pickle
import random

df = pd.read_csv("image_data/stage_2_train_labels.csv")
normal_ids = open("image_data/normal_ids.txt").read().splitlines()

extracted_tuples = []
box_dict = {}

for id in normal_ids:
    box_dict[id] = []

for index, row in df.iterrows():
    if df.loc[index].at['patientId'] in normal_ids:
        # box_dict[df.loc[index].at['patientId']].append((df.loc[index].at['x'], df.loc[index].at['y'], df.loc[index].at['width'], df.loc[index].at['height']))

        box_dict[df.loc[index].at['patientId']].append((0, 0, 0, 0))

with open('image_data/box_dict_normal.pickle', 'wb') as handle:
    pickle.dump(box_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
