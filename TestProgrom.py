import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf


dataload = "D:\\FDU\\19shumo\\train_set"
file_name = "train_108401.csv"
checkpoint = "D:\\FDU\\19shumo\\checkpoint"

train_data = pd.read_csv(os.path.join(dataload, file_name), usecols=[1,2,3,4,5,6,7,12,13,15,16,17])
# print(train_data.info())
train_data.drop_duplicates()
# print(train_data.info())
train_data.fillna(train_data.mean())

h1 = train_data["Height"].astype(int)
h2 = train_data["Building Height"].astype(int)
train_data["HeightA-B"] = h1 - h2

t1 = train_data["Electrical Downtilt"].astype(int)
t2 = train_data["Mechanical Downtilt"].astype(int)
train_data["thetaA+B"] = t1 + t2

# print(train_data.info())
train_data_1 = train_data.copy()

x1 = train_data["Cell X"].astype(float)
y1 = train_data["Cell Y"].astype(float)
x2 = train_data["X"].astype(float)
y2 = train_data["Y"].astype(float)

dis = []

for i in range(len(x1)):
    di = sqrt((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2)
    dis.append(di)

train_data["Distance"] = dis
train_data_2 = train_data.copy()
# print(train_data.info())

y_label = train_data["RSRP"]
y_data = pd.DataFrame({"RSRP": y_label})
x_data = train_data.drop(columns=["Height", "Building Height", "Cell X", "Cell Y", "Electrical Downtilt",
                                  "Mechanical Downtilt", "X", "Y", "RSRP"])


print(x_data.info())
print(y_data.info())
print(x_data.shape[0])

def normalize(x):

    mean = x.mean(axis=0)

    std = x.std(axis=0)
    R= (x - mean) / std
    return R
def normalize2(x):

    ma = x.max()
    mi = x.min()
    R= (x - mi) / ma
    return R

def addbias(x, y):
    m, n = x.shape
    x = np.reshape(np.c_[np.ones(m), x], [m, n+1])
    y = np.reshape(np.power(2, y), [m, 1])
    return x, y

x_data, y_data = addbias(x_data, y_data)
y_data = np.array(y_data)
y_data.astype(np.float32)
val_rate = 0.2
train_num = int(x_data.shape[0] * (1.0 - val_rate))
x_train = x_data[:train_num]
y_train = y_data[:train_num]
x_val = x_data[train_num:]
y_val = y_data[train_num:]
a, b = x_train.shape
print(a, b)

# os.makedirs(checkpoint, exist_ok=True)
# 定义变量函数(权重和偏差)，stdev参数表示方差
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return (weight)


def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return (bias)


def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(tf.cast(input_layer, tf.float32), weights), biases)
    return (tf.nn.relu(layer))


weight_1 = init_weight(shape=[7, 25], st_dev=10.0)
print(weight_1.shape)
bias_1 = init_bias(shape=[25], st_dev=10.0)
print(bias_1.shape)
layer_1 = fully_connected(x_train, weight_1, bias_1)
# --------Create second layer (10 hidden nodes)--------
weight_2 = init_weight(shape=[25, 10], st_dev=10.0)
bias_2 = init_bias(shape=[10], st_dev=10.0)
layer_2 = fully_connected(layer_1, weight_2, bias_2)

# --------Create third layer (3 hidden nodes)--------
weight_3 = init_weight(shape=[10, 3], st_dev=10.0)
bias_3 = init_bias(shape=[3], st_dev=10.0)
layer_3 = fully_connected(layer_2, weight_3, bias_3)

# --------Create output layer (1 output value)--------
weight_4 = init_weight(shape=[3, 1], st_dev=10.0)
bias_4 = init_bias(shape=[1], st_dev=10.0)
final_output = fully_connected(layer_3, weight_4, bias_4)

X = tf.placeholder(tf.float32, name='X', shape=[a, b])
Y = tf.placeholder(tf.float32, name='Y')
# batch_size = 100
# n_batchs = int(np.ceil(a / batch_size))

loss = tf.reduce_mean(tf.abs(Y - final_output, name="loss"))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

init_op = tf.global_variables_initializer()
total = []
with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('graphs', sess.graph)
    for i in range(100):
        _, l = sess.run([optimizer, loss], feed_dict={X: x_train, Y: y_train})
        total.append(l)
        print("Epoch {0}: Loss {1}".format(i, l))
    writer.close()
    tf.saved_model.simple_save(sess, checkpoint, inputs={"myInput": X}, outputs={"myOutput": final_output})

plt.plot(total)
plt.show()




'''fig,ax = plt.subplots()
ax.scatter(train_data["X"], train_data["Y"], alpha=0.4)
ax.grid(True)
fig.tight_layout()
plt.show()'''

'''
combine_data = x_data.copy()
combine_data["RSRP"] = y_label
corr_matrix = combine_data.corr()
print(corr_matrix["RSRP"].sort_values(ascending=False))

combine_data_1 = combine_data.copy()
new_combine_data = (combine_data-combine_data.mean())/combine_data.std()

print(new_combine_data.info())'''