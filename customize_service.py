import numpy as np
# from model_service.tfserving_model_service import TfServingBaseService
import pandas as pd
import os
from math import sqrt


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)


def file_name(data_dir):
    L = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            img_name = os.path.split(file)[1]
            L.append(img_name)

    return L


dataload = "D:\\FDU\\19shumo\\train_set1"

def _preprocess(data):

    # filesDatas = []
    preprocessed_data = {}
    names = ["Cell Index", "Cell X", "Cell Y", "Height", "Azimuth", "Electrical Downtilt",
             "Mechanical Downtilt", "Frequency Band", "RS Power", "Cell Altitude",
             "Cell Building Height", "Cell Clutter Index", "X", "Y",
             "Altitude", "Building Height", "Clutter Index"]
    index_names = ["oceans", "lakes", "wetlands", "suburban open", "urban open areas",
                   "road open areas", "vegetation", "shrub", "forest", "super-high buildings",
                   "high buildings", "mid buildings", "density buildings", "buildings",
                   "sparse industrial", "density","suburban", "developed suburban", "rural", "CBD"]
    df_data = pd.DataFrame(columns=names)
    data_names = file_name(dataload)

    for file_content in data_names:
        pb_data = pd.read_csv(os.path.join(dataload, file_content))
        pb_data = pb_data.drop(columns=["RSRP"])
        df_data = pd.concat([df_data, pb_data], ignore_index=True)
        input_data = np.array(pb_data.get_values()[:, 0:17], dtype=np.float32)

        print(file_content, input_data.shape)
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


    dh = []
    dis = []
    n = len(index_names)
    index_label = np.zeros((len(station_Index), n))

    for i in range(len(station_X)):
        di = sqrt((station_X[i] - mobile_X[i]) ** 2 + (station_Y[i] - mobile_Y[i]) ** 2)
        he = abs(station_AH[i] + station_BH[i] - mobile_AH[i] - 0.5 * mobile_BH[i])
        ht = di * np.tan((theta_A[i] + theta_B[i])/ 180 * np.pi)
        d_h = he - ht
        dh.append(d_h)
        dis.append(np.log10(sqrt(di**2 + he**2)))
        idx1 = station_Index[i]
        idx2 = mobile_Iindex[i]
        index_label[i][idx1-1] += 1
        index_label[i][idx2-1] += 1
    print(station_AH[0], station_BH[0], mobile_AH[0], mobile_BH[0])
    print(theta_A[0], theta_B[0])
    print(np.tan((theta_A[i] + theta_B[i])/ 180 * np.pi))
    df_data["DeltaHeight"] = dh
    df_data["Distance"] = dis
    df_data["Frequency Band"] = np.log10(df_data["Frequency Band"])

    for i in range(n):
        # print(index_names[i])
        infom = index_label[:,i].reshape(-1, 1)
        # print(infom.shape)
        df_data[index_names[i]] = infom

    # print(df_data.info())
    drop_names = ["Cell Index", "Cell X", "Cell Y", "Height", "Electrical Downtilt",
                 "Mechanical Downtilt", "Cell Altitude", "Cell Clutter Index",
                 "X", "Y", "Altitude", "Building Height", "Clutter Index"]
    df_data = df_data.drop(columns=drop_names)
    filesDatas = np.array(df_data, dtype=np.float32).reshape(-1, len(names) + len(index_names) - len(drop_names) + 2)

    # print(df_data.info())
    preprocessed_data['myInput'] = filesDatas
    print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)
    # print(filesDatas[0])
    print(df_data.head(10))
    # print(filesDatas[:5, :])
    return preprocessed_data


if __name__ == '__main__':
    pre_data = _preprocess(dataload)








'''class mnist_service(TfServingBaseService):

   

    def _postprocess(self, data):
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            infer_output["RSRP"] = results
        return infer_output'''