import pandas as pd
import numpy as np
from tqdm import tqdm
from get_road_point import get_road_point, get_kv_point2prop
import time


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)


def pro_data(dataload):

    index_names = ["oceans", "lakes", "wetlands", "suburban open", "urban open areas",
                   "road open areas", "vegetation", "shrub", "forest", "super-high buildings",
                   "high buildings", "mid buildings", "density buildings", "buildings",
                   "sparse industrial", "density", "suburban", "developed suburban", "rural", "CBD"]

    # df_data = pd.DataFrame(columns=names)

    print("---------------------------")
    print("Step 1: Reading data ......")

    df_data = pd.read_csv(dataload + '/train.csv')

    station_X = np.array(df_data["Cell X"], dtype=np.float64)
    station_Y = np.array(df_data["Cell Y"], dtype=np.float64)
    mobile_X = np.array(df_data["X"], dtype=np.float64)
    mobile_Y = np.array(df_data["Y"], dtype=np.float64)

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
    di = np.sqrt(np.multiply((station_X - mobile_X), (station_X - mobile_X)) + np.multiply((station_Y - mobile_Y), (station_Y - mobile_Y)))
    he = np.abs(station_AH + station_BH - mobile_AH - 0.5*mobile_BH)
    ht = np.multiply(di, np.tan((theta_A + theta_B)*np.pi / 180))
    dis = (np.sqrt(np.multiply(di, di) + np.multiply(he, he)))
    dh = he - ht

    n = len(index_names)
    index_label = np.zeros((len(station_Index), n))

    point_dic = get_kv_point2prop(mobile_X, mobile_Y, mobile_AH, mobile_BH, mobile_Iindex)

    count = {}
    for i in tqdm(range(len(station_X))):
        # print(station_X[i], station_Y[i], mobile_X[i], mobile_Y[i])
        # print(i)
        load_line = get_road_point([0, 0], [int(mobile_X[i] - station_X[i]), int(mobile_Y[i] - station_Y[i])], 5)
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

    df_data["DealtaHeight"] = dh
    df_data["Distance"] = dis

    for i in tqdm(range(n)):
        # print(index_names[i])
        infom = index_label[:, i].reshape(-1, 1)
        # print(infom.shape)
        df_data[index_names[i]] = infom

    # print(df_data.info())
    label_data = df_data["RSRP"]

    print("---------------------------")
    print("Step 3: Dropping unuse datas ......")
    df_data['index'] = df_data["Cell Index"]
    df_data = df_data.drop(columns=["Cell Index", "Cell X", "Cell Y", "Electrical Downtilt",
                                    "Mechanical Downtilt", "Cell Altitude", "Cell Clutter Index",
                                    "X", "Y", "Altitude", "Building Height", "Clutter Index",
                                    "oceans", "wetlands", "suburban open", "forest", "rural", "CBD"])
    cell_index_list = list(df_data['index'])

    df_data.columns = ['Height', 'Azimuth', 'Frequency_Band', 'RS_Power',
                       'Cell_Building_Height', 'RSRP', 'DealtaHeight', 'Distance', 'lakes',
                       'urban_open_areas', 'road_open_areas', 'vegetation', 'shrub',
                       'super_high_buildings', 'high_buildings', 'mid_buildings',
                       'density_buildings', 'buildings', 'sparse_industrial', 'density',
                       'suburban', 'developed_suburban', 'index']
    print(df_data.head(10))
    print(len(cell_index_list))
    cell_index_set = set(cell_index_list)
    print(len(cell_index_set))
    print(df_data.info())
    for i in tqdm(cell_index_set):
        # i is Cell Index
        print(i)
        num = i
        print(type(np.int64(num)))
        print(type(df_data['index'][10]))
        save_data = df_data[df_data['index'] == np.int64(num)]
        save_data.to_csv(dataload + '/data/train_' + str(i) + ".csv", index=False)



if __name__ == "__main__":
    dataload = "D:/FDU/19shumo/train_set2"
    start_data = time.time()
    pro_data(dataload)
    end_data = time.time()
    print("data process time = ", end_data - start_data)