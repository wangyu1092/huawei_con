import os
import pandas as pd
from tqdm import tqdm

dataload = "train_set"


def file_name(data_dir):
    L = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            img_name = os.path.split(file)[1]
            L.append(img_name)

    return L


csv_names = file_name(dataload)

for i in tqdm(range(len(csv_names))):
    path = os.path.join(dataload, csv_names[i])
