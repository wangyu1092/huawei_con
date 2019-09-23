from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from get_road_point import get_road_point, get_kv_point2prop



def train_xgboost(dataload):

    df_data = pd.read_csv("D:/FDU/19shumo/train_set2/train.csv")
    train_X = df_data.drop(columns=["RSRP"])
    train_Y = df_data['RSRP']
    # 参数调优
    X_train, X_test, y_train, y_true = train_test_split(train_X, train_Y, test_size=0.98)
    # list1 = y_true.tolist()
    true_list = y_true.tolist()
    # true_list = []
    # for i in range(len(list1)):
    #     true_list.append(list1[i][0])

    params = [5]
    for param in params:
        model = XGBRegressor(learning_rate=0.01,
                             n_estimators=1000,
                             max_depth=8,
                             gamma=0.05,
                             min_child_weight=1,
                             seed=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             reg_alpha=0,
                             reg_lambda=1)
        model.fit(X_train, y_train)

        model.get_booster().save_model('D:/model_18_100.py')
        # y_pre = model.predict(X_test)
        # y_pre_list = y_pre.tolist()
        # # 计算均方误差
        # s = 0
        # for i in range(len(true_list)):
        #     # print("true = ", true_list[i])
        #     # print("pre = ", y_pre_list[i])
        #     s += ((y_pre_list[i] - true_list[i])**2)
        # mse = s / len(true_list)
        # rmse = math.sqrt(mse)
        # print(param)
        # print("rsme = ", rmse)

"""
# 预测代码
model = XGBRegressor(learning_rate=0.01,
                     n_estimators=700,
                     max_depth=8,
                     gamma=0.05,
                     min_child_weight=1,
                     seed=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     reg_alpha=0,
                     reg_lambda=1)
# train_X = train_X.drop('time', axis=1)
model.fit(train_X, train_Y)
X = test.drop('id', axis=1)
# X = X.drop('time', axis=1)
Y = model.predict(X)

ans = pd.DataFrame(Y, columns=['price'])
test_id = pd.read_csv("dataset/test2.csv")
pd.concat([test_id['id'], ans], axis=1).to_csv('ans/xgb_ans.csv', index=False)
"""


if __name__ == "__main__":
    dataload = "D:/FDU/19shumo/train_set2"
    start_xgb = time.time()
    train_xgboost(dataload)
    end_xgb = time.time()
    print("xgb time = ", end_xgb - start_xgb)


    # tar = xgb.Booster(model_file='xgb.model')
    # x_test = xgb.DMatrix(x_test)
    # pre = tar.predict(x_test)
    # act = y_test
    # print(mean_squared_error(act, pre))
