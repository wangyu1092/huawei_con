import os
import pandas as pd
from tqdm import tqdm

dataload = "D:\\FDU\\19shumo\\train_set"
outputfile = "D:\\FDU\\19shumo\\train.csv"


def file_name(data_dir):
    L = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            img_name = os.path.split(file)[1]
            L.append(img_name)

    return L


csv_names = file_name(dataload)
x = pd.read_csv(os.path.join(dataload, csv_names[0]))
x[0:0].to_csv(outputfile, mode='a', index=False)
for i in tqdm(range(len(csv_names))):
    # print(csv_names[i])
    path = os.path.join(dataload, csv_names[i])
    data = pd.read_csv(path)
    data.to_csv(outputfile, mode='a', index=False, header=False)

