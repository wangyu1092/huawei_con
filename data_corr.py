import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

dataload = "D:/FDU/19shumo"
df_data = pd.read_csv(dataload + '/feature_7.csv')

# print(df_data.head(10))
corr_M = df_data.corr()

print(corr_M["RSRP"].sort_values(ascending=False))
