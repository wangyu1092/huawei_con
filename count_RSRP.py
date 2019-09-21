import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
dataload = "D:\\FDU\\19shumo\\train_set2"

train_data = pd.read_csv(os.path.join(dataload, "train.csv"))

# x = list(train_data["Altitude"])
drop_names = ["Cell Index", "Cell X", "Cell Y", "Height", "Azimuth", "Electrical Downtilt",
              "Mechanical Downtilt", "Frequency Band", "RS Power", "Cell Altitude",
              "Cell Building Height", "Cell Clutter Index", "X", "Y", "Altitude", "Building Height"]
train_data = train_data.drop(columns=drop_names)
x = {}
print("train_data.shape = ", train_data.shape)

for i in tqdm(range(len(train_data["RSRP"]))):
    if train_data["Clutter Index"][i] not in x:
        x[train_data["Clutter Index"][i]] = {}
    if  train_data["RSRP"][i] not in x[train_data["Clutter Index"][i]]:
        x[train_data["Clutter Index"][i]][train_data["RSRP"][i]] = 1
    else:
        x[train_data["Clutter Index"][i]][train_data["RSRP"][i]] += 1

result = {}

for k, v in tqdm(x.items()):
    # list1 = sorted(v.items(), key=lambda x: x[1], reverse=True)

    # print(k)
    # print("list1: ", list1)
    # list2 = list1[:10]
    list2 = list(v.items())
    # print(list2)
    # print(list2)
    list3 = []
    for i in range(len(list2)):
        tupe_RSRP = list2[i]
        RSRP = tupe_RSRP[0]
        list3.append(RSRP)

    # print(len(list3))
    result[k] = sum(list3) / len(list3)

for k, v in result.items():
    print(k, v)




# print("len(x) = ", len(x))
# y = set(x)
# print("len(y) = ", len(y))
# d = []
# for i in tqdm(y):
#     d.append(x.count(i))
#
# # 设置中文字体和负号正常显示
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
#
# x = range(len(d))
#
# rects1 = plt.bar(x=x, height=d, width=0.4, alpha=0.8, color='red')
# plt.ylabel("数量")
#
# plt.xticks([index for index in x], y, rotation=90)
#
# plt.tick_params(labelsize=8)
# plt.xlabel("种类")
# plt.title("频数分布-海拔高度")
# plt.legend()     # 设置题注
# # 编辑文本
# for rect in rects1:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(int(height/10000))+'w', ha="center", va="bottom")
# plt.savefig(os.path.join(dataload, "Altitude.png"))
# plt.show()
