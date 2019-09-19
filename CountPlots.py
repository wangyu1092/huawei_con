import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
dataload = "D:\\FDU\\19shumo"

train_data = pd.read_csv(os.path.join(dataload, "train.csv"))
x = list(train_data["Frequency Band"])
y = set(x)
d = []
for i in tqdm(y):
    d.append(x.count(i))

# 设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

x = range(len(d))

rects1 = plt.bar(left=x, height=d, width=0.4, alpha=0.8, color='red')
plt.ylabel("数量")

plt.xticks([index for index in x], y, rotation=90)

plt.tick_params(labelsize=8)
plt.xlabel("种类")
plt.title("频数分布-发射机中心频率")
plt.legend()     # 设置题注
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(int(height/10000))+'w', ha="center", va="bottom")
plt.savefig(os.path.join(dataload, "Frequency-Band.png"))
plt.show()

