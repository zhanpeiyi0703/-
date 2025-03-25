from sklearn.ensemble import RandomForestRegressor
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_score
from wordcloud import WordCloud
import re

#
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'
"""用于获取数据，字符转换"""


def preprocess_all_price(data):
    # 去除末尾的“万”字并将缺省值替换为0
    processed_data = []
    for item in data:
        if item.endswith('万'):
            item = item[:-1]  # 去除末尾的“万”字
        if not item:  # 如果为空字符串（缺失值），将其替换为0
            item = '0'
        processed_data.append(float(item))
    return processed_data

def encode_direct(data):
    # 创建一个空字典用于保存已存在的字符串及其对应的数值
    mapping = {}
    # 初始化数值
    count = 1
    # 编码数据
    encoded_data = []
    for item in data:
        # 如果字符串在字典中不存在，则将其添加到字典中，并为其分配一个新的数值
        if item not in mapping:
            mapping[item] = count
            count += 1
        # 将字符串转换为对应的数值
        encoded_data.append(mapping[item])
    return encoded_data

def preprocess_area(data):
    # 去除末尾的“㎡”符号并将剩余部分转换为浮点数，同时处理缺失值
    processed_data = []
    for item in data:
        item = item.rstrip("㎡")  # 去除末尾的“㎡”符号
        if not item:  # 如果为空字符串（缺失值），将其替换为0
            item = '0'
        processed_data.append(float(item))
    return processed_data

def encode_data_with_default(data, default_value=0):
    # 创建一个空字典用于保存已存在的字符串及其对应的数值
    mapping = {}
    # 初始化数值
    count = 1
    # 编码数据
    encoded_data = []
    for item in data:
        # 如果字符串在字典中不存在，则将其添加到字典中，并为其分配一个新的数值
        if item not in mapping:
            mapping[item] = count
            count += 1
        # 将字符串转换为对应的数值
        encoded_data.append(mapping.get(item, default_value))
    return encoded_data


def extract_room_info(data):
    # 初始化三个数组用于存储A室、B厅、C卫的信息
    room_info_A = []
    room_info_B = []
    room_info_C = []

    # 提取每行数据的A室、B厅、C卫信息
    for item in data:
        parts = item.split("室")  # 将字符串分割成A室和B厅C卫的两部分
        A_part = int(parts[0])  # 提取A室的信息
        B_C_part = parts[1].split("厅")  # 将B厅C卫部分再分割成B厅和C卫的两部分
        B_part = int(B_C_part[0])  # 提取B厅的信息
        C_part = int(B_C_part[1].rstrip("卫"))  # 提取C卫的信息，并去除末尾的"卫"

        # 将提取的信息添加到对应的数组中
        room_info_A.append(A_part)
        room_info_B.append(B_part)
        room_info_C.append(C_part)

    return room_info_A, room_info_B, room_info_C

def remove_unit(data):
    # 去除每一行中的"元/㎡"符号
    processed_data = []
    for item in data:
        item = item.rstrip("元/㎡")  # 去除末尾的"元/㎡"符号
        processed_data.append(item)
    return processed_data



"""这些"""
data = pd.read_csv("安居客.csv")
x = data[['总价','朝向','面积','优势1','优势2','优势3','房型']]
y = data['单价']
all_price = np.array(preprocess_all_price(x['总价']))
direct = np.array(encode_direct(x['朝向']))
area = np.array(preprocess_area(x['面积']))
advan1 = np.array(encode_data_with_default(x['优势1'], default_value=0))
advan2 = np.array(encode_data_with_default(x['优势2'], default_value=0))
advan3 = np.array(encode_data_with_default(x['优势3'], default_value=0))
shi, ting, wei = extract_room_info(x['房型'])
shi = np.array(shi)
ting = np.array(ting)
wei = np.array(wei)
unit_price= np.array(remove_unit(y))
all_x = np.column_stack((all_price.astype(float), direct.astype(float), area.astype(float), advan1.astype(float), advan2.astype(float), advan3.astype(float), shi.astype(float), ting.astype(float), wei.astype(float)))
all_y = np.array(unit_price.astype(float))





X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=42)

# 创建随机森林回归模型
regressor = RandomForestRegressor()



# 训练模型
regressor.fit(X_train, y_train)

# 进行预测
y_pred = regressor.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# 打印回归参数

# 计算 R² 系数
r2 = r2_score(y_test, y_pred)
print('R² Coefficient:', abs(r2))


temp = [[102.0, 5.0, 120.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0]]
temp = np.array(temp)
result = regressor.predict(temp)
real = 8931
print((real-result)/real*100)
