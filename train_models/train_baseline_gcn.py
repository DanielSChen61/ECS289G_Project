
import numpy as np
import pickle
import math



def calc_dist(c1, c2):
    gamma = 200

    value = ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) / (-2*(gamma)**2)
    value = math.exp(value)
    return value


# 3000 patients, 5138 boxes total
with open('image_data/box_dict_pneumonia.pickle', 'rb') as handle:
    box_dict_pneumonia = pickle.load(handle)

with open('image_data/box_dict_normal.pickle', 'rb') as handle:
    box_dict_normal = pickle.load(handle)


center_dict_pneumonia = {}
center_dict_normal = {}

for key in box_dict_pneumonia:
    center_dict_pneumonia[key] = []

    for value in box_dict_pneumonia[key]:
        # value[0] = x, value[1] = y, value[2] = width, value[3] = height
        center = (value[0] + value[2] / 2, value[1] + value[3] / 2)
        center_dict_pneumonia[key].append(center)

for key in box_dict_normal:
    center_dict_normal[key] = [(0, 0)]

center_dict_merged = center_dict_pneumonia | center_dict_normal

adj_matrix_dict = {}
for key in center_dict_merged:
    size = len(center_dict_merged[key])
    matrix = np.zeros((size, size))

    for i in range(size):
        matrix[i][i] = 0

    for i in range(size):
        for j in range(i+1, size):
            matrix[i][j] = calc_dist(center_dict_merged[key][i], center_dict_merged[key][j])
            matrix[j][i] = matrix[i][j]

    adj_matrix_dict[key] = matrix


binary_label_matrix = np.zeros((5000, 2))

for i in range(5000):
    if i < 3000:
        binary_label_matrix[i] = [1, 0]
    else:
        binary_label_matrix[i] = [0, 1]

print(binary_label_matrix)