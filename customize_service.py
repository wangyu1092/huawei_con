import numpy as np
# from model_service.tfserving_model_service import TfServingBaseService
import pandas as pd
import os
from math import sqrt

def file_name(data_dir):
    L = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            img_name = os.path.split(file)[1]
            L.append(img_name)

    return L


dataload = "D:\\FDU\\19shumo\\test_set"

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
                   "sparse industria", "density","suburban", "developed suburban", "rural", "CBD"]
    df_data = pd.DataFrame(columns=names)
    data_names = file_name(dataload)

    for file_content in data_names:
        pb_data = pd.read_csv(os.path.join(dataload, file_content))
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
        ht = di * np.tan(theta_A[i] + theta_B[i])
        d_h = np.log10(he - ht)
        dh.append(d_h)
        dis.append(np.log10(sqrt(di**2 + he**2)))
        idx1 = station_Index[i]
        idx2 = mobile_Iindex[i]
        index_label[i][idx1-1] += 1
        index_label[i][idx2-1] += 1

    df_data["DealtaHeight"] = dh
    df_data["Distance"] = dis
    df_data["Frequency Band"] = np.log10(df_data["Frequency Band"])

    for i in range(n):
        print(index_names[i])
        infom = index_label[:,i].reshape(-1, 1)
        print(infom.shape)
        df_data[index_names[i]] = infom

    print(df_data.info())
    df_data = df_data.drop(columns=["Cell Index","Cell X", "Cell Y", "Height","Electrical Downtilt",
                                    "Mechanical Downtilt", "Cell Altitude", "Cell Clutter Index",
                                    "X", "Y", "Altitude", "Building Height", "Clutter Index"])
    filesDatas = np.array(df_data, dtype=np.float32).reshape(-1, 26)

    print(df_data.info())
    preprocessed_data['myInput'] = filesDatas
    print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)
    print(filesDatas[0])

    return preprocessed_data


pre_data = _preprocess(dataload)








'''class mnist_service(TfServingBaseService):

   

    def _postprocess(self, data):
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            infer_output["RSRP"] = results
        return infer_output'''