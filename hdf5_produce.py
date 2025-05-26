

import os
import pandas as pd
import numpy as np

# 定义文件夹路径
data_folder_train = r"/data_fil/train"  # 替换为你的文件夹路径
data_folder_test = r"/data_fil/test"  # 替换为你的文件夹路径

# 定义 get_amp_phase 函数
def get_amp_phase(data):
    X_cmplx = data[0, :] + 1j * data[1, :]
    X_amp = np.abs(X_cmplx)
    X_ang = np.arctan2(data[1, :], data[0, :]) / np.pi

    X_amp_min = np.min(X_amp)
    X_amp_diff = np.max(X_amp) - np.min(X_amp)
    X_amp = ((X_amp - X_amp_min) / X_amp_diff)

    X = np.stack((X_amp, X_ang), axis=0)
    return X

# 处理单个 .csv 文件
def process_csv(file_path, data_list):
    # 读取 .csv 文件
    df = pd.read_csv(file_path, header=None)  # 假设 .csv 文件没有表头

    # 提取 IQ 信号（第一列和第二列），转换为 2*L 的 numpy 数组
    iq_signal = df.iloc[:, :2].values.T  # 转置为 2*L 形状
    ap_signal = get_amp_phase(iq_signal)

    # 提取码序列（第三列），去除空缺值并转换为 numpy 数组
    code_sequence = df.iloc[:, 2].dropna().values  # 去除空缺值并转换为数组

    # 提取调制方式编号（第四列）和码元宽度（第五列），只有第一行有值
    modulation_type = df.iloc[0, 3] if not pd.isna(df.iloc[0, 3]) else None
    symbol_width = df.iloc[0, 4] if not pd.isna(df.iloc[0, 4]) else None

    # 将数据组织为一个字典
    data_dict = {
        "ap_signal": ap_signal,
        "code_sequence": code_sequence,
        "modulation_type": modulation_type,
        "symbol_width": symbol_width,
    }

    # 将字典添加到数据列表中
    data_list.append(data_dict)

# 递归遍历文件夹及其子文件夹
def process_folder(folder_path):
    data_list = []
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isdir(full_path):
            # 如果是子文件夹，递归处理并合并数据
            sub_data_list = process_folder(full_path)
            data_list.extend(sub_data_list)  # 使用 extend 而不是 append
        elif entry.endswith(".csv"):
            # 如果是 .csv 文件，读取数据
            process_csv(full_path, data_list)
    return data_list

# 处理文件夹及其子文件夹中的所有 .csv 文件
data_list_train = process_folder(data_folder_train)
data_list_test = process_folder(data_folder_test)  # 修复：使用 data_folder_test

# 将数据列表转换为 DataFrame，并指定列名
df_train = pd.DataFrame(data_list_train, columns=["ap_signal", "code_sequence", "modulation_type", "symbol_width"])
df_test = pd.DataFrame(data_list_test, columns=["ap_signal", "code_sequence", "modulation_type", "symbol_width"])

# 查看读取的数据
print("Train Data:")
print(df_train.head())
print("\nTest Data:")
print(df_test.head())

# 将两个数据集保存到同一个 HDF5 文件中，并赋予不同的 key
hdf5_path = r"/data_fil/data_fil_AP.h5"  # 文件保存路径
with pd.HDFStore(hdf5_path, mode="w") as store:
    store.put("train", df_train)  # 保存第一个数据集，key 为 "train"
    store.put("test", df_test)  # 保存第二个数据集，key 为 "test"

print(f"数据已保存到 {hdf5_path}")