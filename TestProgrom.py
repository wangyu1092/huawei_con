import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from math import sqrt
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf
from random import shuffle
from tensorflow.python.saved_model import signature_constants

from get_road_point import get_road_point, get_kv_point2prop

dataload = "D:\\FDU\\19shumo\\train_set"
dataload_2 = "D:\\FDU\\19shumo"
# file_name = "train.csv"
checkpoint = "D:\\FDU\\19shumo\\checkpoint"

def file_name(data_dir):
    L = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            img_name = os.path.split(file)[1]
            L.append(img_name)

    return L

def normalize(x):

    mi = x.min(axis=0)
    ma = x.max(axis=0)
    R= (x - mi) / (ma - mi)
    return R

def normalize2(x):

    me = x.mean(axis=0)
    st = x.std(axis=0)
    R= (x - me) / st
    return R

def addbias(x, y):
    m, n = x.shape
    x = np.reshape(np.c_[np.ones(m), x], [m, n+1])
    y = np.reshape(y, [m, 1])
    return x, y

def pro_data():

    preprocessed_data = {}
    names = ["Cell Index", "Cell X", "Cell Y", "Height", "Azimuth", "Electrical Downtilt",
             "Mechanical Downtilt", "Frequency Band", "RS Power", "Cell Altitude",
             "Cell Building Height", "Cell Clutter Index", "X", "Y",
             "Altitude", "Building Height", "Clutter Index", "RSRP"]
    index_names = ["oceans", "lakes", "wetlands", "suburban open areas", "urban open areas",
                   "road open areas", "vegetation", "shrub", "forest", "super-high buildings",
                   "high buildings", "mid buildings", "density buildings", "buildings",
                   "sparse industrial", "density", "suburban", "developed suburban", "rural", "CBD"]

    index_weight = [0.0, 0.0, 93.5, 0.0, 0.0, 94.5, 93.5, 95.5, 96.5, 0.0,
                    94.5, 96.5, 95.5, 95.5, 95.5, 95.5, 99.7, 94.3, 92.85, 0.0]

    df_data = pd.DataFrame(columns=names)
    data_names = file_name(dataload)
    shuffle(data_names)
    print("---------------------------")
    print("Step 1: Reading data ......")

    # df_data = pd.read_csv(os.path.join(dataload_2, "train.csv"))
    for i in tqdm(range(50)):
        pb_data = pd.read_csv(os.path.join(dataload, data_names[i]))
        df_data = pd.concat([df_data, pb_data], ignore_index=True)
        #input_data = np.array(pb_data.get_values()[:, 0:17], dtype=np.float32)
        #print(file_content, input_data.shape)
        # filesDatas.extend(input_data)

    # print(filesDatas[0].shape)

    station_X = np.array(df_data["Cell X"], dtype=np.float64)
    station_Y = np.array(df_data["Cell Y"], dtype=np.float64)
    mobile_X = np.array(df_data["X"], dtype=np.float64)
    mobile_Y = np.array(df_data["Y"], dtype=np.float64)
    theta_AZ = np.array(df_data["Azimuth"], dtype=np.float64)

    station_AH = np.array(df_data["Cell Altitude"], dtype=np.float32)
    station_BH = np.array(df_data["Height"], dtype=np.float32)
    mobile_AH = np.array(df_data["Altitude"], dtype=np.float32)
    mobile_BH = np.array(df_data["Building Height"], dtype=np.float32)

    theta_A = df_data["Electrical Downtilt"].astype(float)
    theta_B = df_data["Mechanical Downtilt"].astype(float)

    station_Index = df_data["Cell Clutter Index"].astype(int)
    mobile_Iindex = df_data["Clutter Index"].astype(int)

    print("---------------------------")
    print("Step 2: Getting dhs, distances and indexes ......")

    delat_X = mobile_X - station_X

    theta = np.arctan((mobile_Y - station_Y) / (mobile_X - station_Y)) * 180 / np.pi
    theta *= -1
    theta[delat_X > 0] += 90
    theta[delat_X < 0] += 270

    di = np.sqrt(np.multiply((station_X - mobile_X), (station_X - mobile_X)) + np.multiply((station_Y - mobile_Y), (station_Y - mobile_Y)))
    he = np.abs(station_AH + station_BH - mobile_AH - 0.5*mobile_BH)
    ht = np.multiply(di, np.tan((theta_A + theta_B)*np.pi / 180))
    dis = np.sqrt(np.multiply(di, di) + np.multiply(he, he))
    dh = abs(ht - he)

    '''
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
        index_label[i][idx2 - 1] += 1'''

    n = len(index_names)
    index_label = np.zeros((len(station_Index), n))

    point_dic = get_kv_point2prop(mobile_X, mobile_Y, mobile_AH, mobile_BH, mobile_Iindex)

    count = {}
    for i in tqdm(range(len(station_X))):
        # print(station_X[i], station_Y[i], mobile_X[i], mobile_Y[i])
        # print(i)
        load_line = get_road_point([0, 0], [int(mobile_X[i] - station_X[i]), int(mobile_Y[i] - station_Y[i])], 10)
        if len(load_line) not in count:
            count[len(load_line)] = 1
        else:
            count[len(load_line)] += 1
        for j in load_line:
            # print(j)
            point = (station_X[i] + j[0], station_Y[i] + j[1])
            if point in point_dic:
                # print(point)
                index_label[i][point_dic[point][2]] += 1

    df_data["Index Weight"] = np.dot(index_label, np.array(index_weight).reshape(-1, 1))
    df_data["DealtaHeight"] = dh
    df_data["Distance"] = dis
    df_data["Frequency Band"] = np.log2(df_data["Frequency Band"])
    litheta = abs(theta - theta_AZ)
    litheta[litheta > 180] *= -1
    litheta[litheta < -180] += 360

    df_data["LineTheta"] = litheta * np.pi / 180

    for i in tqdm(range(n)):
        # print(index_names[i])
        infom = index_label[:, i].reshape(-1, 1)
        # print(infom.shape)
        df_data[index_names[i]] = infom

    # print(df_data.info())
    label_data = df_data["RSRP"]

    print("---------------------------")
    print("Step 3: Dropping unuse datas ......")
    df_data = df_data.drop(columns=["Cell Index", "Azimuth", "Cell X", "Cell Y", "Height", "Electrical Downtilt",
                                    "Mechanical Downtilt", "Cell Altitude", "Cell Clutter Index",
                                    "X", "Y", "Altitude", "Building Height", "Clutter Index", "RSRP",
                                    "oceans", "wetlands", "suburban open areas", "forest", "rural", "CBD"])

    print(df_data.info())
    df_data_write = df_data.copy()
    df_data_write["RSRP"] = label_data
    # df_data_write.to_csv(os.path.join(dataload_2, "feature2.csv"), index=False)
    InputDatas = np.array(df_data, dtype=np.float32).reshape(-1, 21)
    OutptDatas = np.array(label_data, dtype=np.float32).reshape(-1, 1)

    # InputDatas = normalize(InputDatas) * 100
    # OutptDatas = normalize(OutptDatas)

    label_data = pd.DataFrame({"RSRP": label_data})
    print(df_data.info())
    print(label_data.info())
    preprocessed_data['myInput'] = InputDatas
    preprocessed_data['myLabel'] = OutptDatas

    print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)
    print("preprocessed_data[\'myLabel\'].shape = ", preprocessed_data['myLabel'].shape)

    print(count)
    print(preprocessed_data["myInput"][100])
    print("------------------------------")
    print(preprocessed_data["myInput"][1000])
    print("------------------------------")
    print(preprocessed_data["myInput"][10000])
    print("------------------------------")

    return preprocessed_data

def fetch_batch_data(x_da, y_da, batchsize, idx):
    x_fda = x_da[batchsize*idx:batchsize*(idx+1)]
    y_fda = y_da[batchsize * idx:batchsize * (idx + 1)]

    return x_fda, y_fda

def init_weight(shape, mea, st_dev):
    weight = tf.Variable(tf.random_normal(shape, mean=mea, stddev=st_dev))
    return (weight)

def init_bias(shape, mea, st_dev):
    bias = tf.Variable(tf.random_normal(shape, mean=mea, stddev=st_dev))
    return (bias)

def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(tf.cast(input_layer, tf.float32), weights), biases)
    return (tf.nn.relu(layer))

def CaculatePcrr(y_true,y_pred):
    t = -103

    # print(yt, yp)
    # y_true = np.multiply(y_true, -1)
    # y_pred = np.multiply(y_pred, -1)
    tp = 0
    fp = 0
    fn = 0

    for i, j in zip(y_true, y_pred):
        if i < t and j < t:
            tp += 1
        elif i >= t and j < t:
            fp += 1
        elif i < t and j >= t:
            fn +=1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    pcrr = 2 * (precision * recall) / (precision + recall)
    return pcrr


def add_layer(inputs, input_size, output_size, sdv, activation_function=None):
    with tf.variable_scope("Weights"):
        Weights = tf.Variable(tf.random_normal(shape=[input_size,output_size], stddev=sdv), name="weights")
    with tf.variable_scope("biases"):
        biases = tf.Variable(tf.zeros(shape=[1,output_size]) + 0.1, name="biases")
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b = tf.matmul(inputs,Weights) + biases
    with tf.name_scope("dropout"):
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob=1)
    if activation_function is None:
        return Wx_plus_b
    else:
        with tf.name_scope("activation_function"):
            return activation_function(Wx_plus_b)

def train_net_model(x_in, sdv):

    # 定义变量函数(权重和偏差)，stdev参数表示方差

    layer_0 = add_layer(x_in, 21, 50, sdv, activation_function=tf.nn.relu)
    layer_1 = add_layer(layer_0, 50, 30, sdv, activation_function=tf.nn.relu)
    layer_2 = add_layer(layer_1, 30, 10, sdv, activation_function=tf.nn.relu)
    layer_3 = add_layer(layer_2, 10, 3, sdv, activation_function=tf.nn.relu)
    pred = add_layer(layer_3, 3, 1, sdv)
    return pred

    # # --------Create second layer (25 hidden nodes)--------
    # weight_1 = init_weight(shape=[50, 80], mea=me, st_dev=sdv)
    # bias_1 = init_bias(shape=[80], mea=me, st_dev=sdv)
    # layer_1 = fully_connected(layer_0, weight_1, bias_1)
    #
    # # --------Create second layer (25 hidden nodes)--------
    # weight_2 = init_weight(shape=[80, 50], mea=me, st_dev=sdv)
    # bias_2 = init_bias(shape=[50], mea=me, st_dev=sdv)
    # layer_2 = fully_connected(layer_1, weight_2, bias_2)
    # # --------Create second layer (10 hidden nodes)--------
    # weight_3 = init_weight(shape=[50, 30], mea=me, st_dev=sdv)
    # bias_3 = init_bias(shape=[30], mea=me, st_dev=sdv)
    # layer_3 = fully_connected(layer_2, weight_3, bias_3)
    #
    # # --------Create third layer (3 hidden nodes)--------
    # weight_4 = init_weight(shape=[30, 10], mea=me, st_dev=sdv)
    # bias_4 = init_bias(shape=[10], mea=me, st_dev=sdv)
    # layer_4 = fully_connected(layer_3, weight_4, bias_4)
    #
    # # --------Create output layer (1 output value)--------
    # weight_5 = init_weight(shape=[10, 1], mea=me, st_dev=sdv)
    # bias_5 = init_bias(shape=[1], mea=me, st_dev=sdv)
    # pred = fully_connected(layer_4, weight_5, bias_5)

    # print(pred.get_shape().as_list())


def train_line_model(x, stv):

    w = tf.Variable(tf.random_normal([21, 1], stddev=stv), name="W")
    b = tf.Variable(1.0, name="b")
    pred = tf.matmul(x, w) + b

    return pred



def main():

    datapre = pro_data()
    x_data = datapre["myInput"]
    y_data = datapre["myLabel"]

    val_rate = 0.2
    train_num = int(x_data.shape[0] * (1.0 - val_rate))
    x_train = x_data[:train_num]
    y_train = y_data[:train_num]
    x_val = x_data[train_num:]
    y_val = y_data[train_num:]
    a, b = x_train.shape
    c, d = x_val.shape
    print(a, b)

    batch_size = 1000
    tn_batchs = int(np.ceil(a / batch_size))
    vn_batchs = int(np.ceil(c / batch_size))

    X = tf.placeholder(tf.float32, name='X', shape=[None, b])
    Y = tf.placeholder(tf.float32, name='Y', shape=[None, 1])

    pred = train_net_model(X, 0.01)
    # pred = train_line_model(X, 1)

    # bias = tf.Variable(tf.fill(pred.get_shape().as_list(), -1), name="bias")
    # preds = tf.matmul(pred, tf.cast(bias, tf.float32))
    # pcrr = CaculatePcrr(Y, pred)
    # print(final_output.shape)

    loss = tf.reduce_mean(tf.square(Y - pred, name="loss"))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    init_op = tf.global_variables_initializer()
    train_total = []
    val_total = []

    # with tf.Session(graph=tf.Graph()) as sess:
    #     meta_graph_def = tf.saved_model.loader.load(sess, ["serve"], checkpoint)
    #     signature = meta_graph_def.signature_def
    #
    #     in_tensor_name = signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['myInput'].name
    #     out_tensor_name = signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['myOutput'].name
    #     x = sess.graph.get_tensor_by_name(in_tensor_name)
    #     y = sess.graph.get_tensor_by_name(out_tensor_name)
    #
    #     for i in range(vn_batchs):
    #
    #         batch_xs, batch_ys = fetch_batch_data(x_val, y_val,batch_size, i)
    #         pre = sess.run(y,feed_dict={x: batch_xs})
    #         print("predict: {0}, y_true: {1}".format(pre[200], batch_ys[200]))


    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter('graphs', sess.graph)
        pre = list()
        for i in range(1000):
            print("Epoch {0}: ------".format(i))
            ev_loss = []
            pre_per = []
            for j in range(tn_batchs):
                x, y = fetch_batch_data(x_train, y_train, batch_size, j)
                _, l, fl = sess.run([optimizer, loss, pred], feed_dict={X: x, Y: y})
                ev_loss.append(l)
                pre_per.extend(fl)
                if j % 100 == 0:
                    print("y-pre[10] and y-true[10]:", fl[100], y[100])
                    # print("{0} / {1} : TrainLoss {2}".format(j, tn_batchs, l))
            # print("y-pre[10] and y-true[10]:", fl[100], y[100])
            t_loss = sum(ev_loss) / tn_batchs
            train_total.append(t_loss)
            print("Total TrainLoss {0}".format(t_loss))
            if i >= 900:
               pc = CaculatePcrr(y_val, pre_per)
               print("train_pcrr:", pc)

            # pre = pre_per
            # pre = []
            # for j in range(vn_batchs):
            #     x, y = fetch_batch_data(x_val, y_val, batch_size, j)
            #     _, l, fl = sess.run([optimizer, loss, pred], feed_dict={X: x, Y: y})
            #     val_total.append(l)
            #     pre.extend(fl)
            #     print("fl[10]:", fl[10], y[10])
            #     if j % 10 == 0:
            #         print("{0} / {1} : ValLoss {2}".format(j, vn_batchs, l))
            # if i >= 90:
            #     pc = CaculatePcrr(pre, y_val)
            #     print("val_pcrr:", pc)

        writer.close()
        # bias = tf.Variable(tf.fill(pred.get_shape().as_list(), -1), name="bias")
        # preds = tf.matmul(pred, tf.cast(bias, tf.float32))
        tf.saved_model.simple_save(sess, checkpoint, inputs={"myInput": X}, outputs={"myOutput": pred})

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