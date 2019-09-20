import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf


dataload = "D:\\FDU\\19shumo\\train_set"
# file_name = "train.csv"
# checkpoint = "D:\\FDU\\19shumo\\checkpoint"

def file_name(data_dir):
    L = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            img_name = os.path.split(file)[1]
            L.append(img_name)

    return L

def pro_data():

    preprocessed_data = {}
    names = ["Cell Index", "Cell X", "Cell Y", "Height", "Azimuth", "Electrical Downtilt",
             "Mechanical Downtilt", "Frequency Band", "RS Power", "Cell Altitude",
             "Cell Building Height", "Cell Clutter Index", "X", "Y",
             "Altitude", "Building Height", "Clutter Index"]
    index_names = ["oceans", "lakes", "wetlands", "suburban open", "urban open areas",
                   "road open areas", "vegetation", "shrub", "forest", "super-high buildings",
                   "high buildings", "mid buildings", "density buildings", "buildings",
                   "sparse industria", "density", "suburban", "developed suburban", "rural", "CBD"]

    df_data = pd.DataFrame(columns=names)
    data_names = file_name(dataload)

    print("---------------------------")
    print("Step 1: Reading data ......")
    for i in tqdm(range(1000)):
        pb_data = pd.read_csv(os.path.join(dataload, data_names[i]))
        df_data = pd.concat([df_data, pb_data], ignore_index=True)
        #input_data = np.array(pb_data.get_values()[:, 0:17], dtype=np.float32)
        #print(file_content, input_data.shape)
        # filesDatas.extend(input_data)

    # print(filesDatas[0].shape)

    station_X = df_data["Cell X"].astype(float)
    station_Y = df_data["Cell Y"].astype(float)
    mobile_X = df_data["X"].astype(float)
    mobile_Y = df_data["Y"].astype(float)

    station_AH = df_data["Cell Altitude"].astype(int)
    station_BH = df_data["Height"].astype(int)
    mobile_AH = df_data["Altitude"].astype(int)
    mobile_BH = df_data["Building Height"].astype(int)

    theta_A = df_data["Electrical Downtilt"]
    theta_B = df_data["Mechanical Downtilt"]

    station_Index = df_data["Cell Clutter Index"].astype(int)
    mobile_Iindex = df_data["Clutter Index"].astype(int)

    print("---------------------------")
    print("Step 2: Getting dhs, distances and indexes ......")
    dh = []
    dis = []
    n = len(index_names)
    index_label = np.zeros((len(station_Index), n))
    for i in tqdm(range(len(station_X))):
        di = sqrt((station_X[i] - mobile_X[i]) ** 2 + (station_Y[i] - mobile_Y[i]) ** 2)
        he = abs(station_AH[i] + station_BH[i] - mobile_AH[i] - 0.5 * mobile_BH[i])
        ht = di * np.tan(theta_A[i] + theta_B[i])
        d_h = np.log10(he - ht)
        dh.append(d_h)
        dis.append(np.log10(sqrt(di ** 2 + he ** 2)))
        idx1 = station_Index[i]
        idx2 = mobile_Iindex[i]
        index_label[i][idx1 - 1] += 1
        index_label[i][idx2 - 1] += 1

    df_data["DealtaHeight"] = dh
    df_data["Distance"] = dis
    df_data["Frequency Band"] = np.log10(df_data["Frequency Band"])

    for i in tqdm(range(n)):
        # print(index_names[i])
        infom = index_label[:, i].reshape(-1, 1)
        # print(infom.shape)
        df_data[index_names[i]] = infom

    # print(df_data.info())
    label_data = df_data["RSRP"]

    print("---------------------------")
    print("Step 3: Dropping unuse datas ......")
    df_data = df_data.drop(columns=["Cell Index", "Cell X", "Cell Y", "Height", "Electrical Downtilt",
                                    "Mechanical Downtilt", "Cell Altitude", "Cell Clutter Index",
                                    "X", "Y", "Altitude", "Building Height", "Clutter Index", "RSRP"])
    InputDatas = np.array(df_data, dtype=np.float32).reshape(-1, 26)
    OutptDatas = np.array(label_data, dtype=np.float32).reshape(-1, 1)

    label_data = pd.DataFrame({"RSRP": label_data})
    print(df_data.info())
    print(label_data.info())
    preprocessed_data['myInput'] = InputDatas
    preprocessed_data['myLabel'] = OutptDatas

    print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)
    print("preprocessed_data[\'myLabel\'].shape = ", preprocessed_data['myLabel'].shape)
    # print(filesDatas[0])

    return preprocessed_data

def fetch_batch_data(x_da, y_da, batchsize, idx):
    x_fda = x_da[batchsize*idx:batchsize*(idx+1)]
    y_fda = y_da[batchsize * idx:batchsize * (idx + 1)]

    return x_fda, y_fda

def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return (weight)

def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return (bias)

def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(tf.cast(input_layer, tf.float32), weights), biases)
    return (tf.nn.relu(layer))


def main():

    datapre = pro_data()
    x_data = datapre["myInput"]
    y_data = datapre["myLabel"]

    val_rate = 0.1
    train_num = int(x_data.shape[0] * (1.0 - val_rate))
    x_train = x_data[:train_num]
    y_train = y_data[:train_num]
    x_val = x_data[train_num:]
    y_val = y_data[train_num:]
    a, b = x_train.shape
    c, d = x_val.shape
    print(a, b)

    batch_size = 10000
    tn_batchs = int(np.ceil(a / batch_size))
    vn_batchs = int(np.ceil(c / batch_size))

    X = tf.placeholder(tf.float32, name='X', shape=[None, b])
    Y = tf.placeholder(tf.float32, name='Y', shape=[None, 1])

    # 定义变量函数(权重和偏差)，stdev参数表示方差
    # --------Create second layer (50 hidden nodes)--------
    weight_0 = init_weight(shape=[26, 50], st_dev=10.0)
    bias_0 = init_bias(shape=[50], st_dev=10.0)
    layer_0 = fully_connected(X, weight_0, bias_0)

    # --------Create second layer (25 hidden nodes)--------
    weight_1 = init_weight(shape=[50, 25], st_dev=10.0)
    bias_1 = init_bias(shape=[25], st_dev=10.0)
    layer_1 = fully_connected(layer_0, weight_1, bias_1)

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

    # print(final_output.shape)


    loss = tf.reduce_mean(tf.square(Y - final_output, name="loss"))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    init_op = tf.global_variables_initializer()
    train_total = []
    val_total = []
    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter('graphs', sess.graph)
        for i in range(100):
            print("Epoch {0}: ------".format(i))
            for j in range(tn_batchs):
                x, y = fetch_batch_data(x_train, y_train, batch_size, j)
                _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
                train_total.append(l)
                if j % 10 == 0:
                    print("{0} / {1} : TrainLoss {2}".format(j, tn_batchs, l))

            for j in range(vn_batchs):
                x, y = fetch_batch_data(x_val, y_val, batch_size, j)
                _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
                val_total.append(l)
                if j % 10 == 0:
                    print("{0} / {1} : ValLoss {2}".format(j, vn_batchs, l))
        writer.close()
        # tf.saved_model.simple_save(sess, checkpoint, inputs={"myInput": X}, outputs={"myOutput": final_output})

    plt.plot(train_total)
    plt.show()


if __name__ == '__main__':
    main()




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