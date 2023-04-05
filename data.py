#!usr/bin/env python
#encoding:utf-8
from __future__ import division
 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd    
# 导入工具库
import numpy as np
import pywt


T = input_seq_len = 10
output_seq_len = 1
dataNum = 7
output_dataNum = 1
n = n_in_features = dataNum
n_out_features = output_dataNum
rec_a = []
rec_d = []
length = 0

# ------------------小波变换 全部
mode = pywt.Modes.smooth


def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    global rec_a,rec_d
    lenx = data.shape[0]
    w = pywt.Wavelet(w)#选取小波函数
    a = data
    ca = []#近似分量
    cd = []#细节分量
    for i in range(2):
        (a, d) = pywt.dwt(a, w, mode)#进行5阶离散小波变换
        ca.append(a)
        cd.append(d) 

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))#重构

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i 
        rec_d.append(pywt.waverec(coeff_list, w) )
 
    for i, y in enumerate(rec_a):
        rec_a[i] = y[:lenx] 

    for i, y in enumerate(rec_d):
        rec_d[i] = y[:lenx]
         

def xiaobo(data_file,fc='db9' ):
    global rec_a,rec_d
    rec_a = []
    rec_d = []
    length = 0
    data = pd.read_csv(data_file)
    cols = list(data.columns[:])
    x_data1 = data
    da = data['pollution_pm2.5'].fillna(method='ffill').fillna(method='bfill') 
    plot_signal_decomp(da,fc, "DWT: Ecg sample - Symmlets5")
    print('""""""""""""""""""""""""""')
    print(fc)
    print('""""""""""""""""""""""""""')
    rec_a = np.array(rec_a).transpose()
    rec_d = np.array(rec_d).transpose()
    # 只使用一阶的高频和二阶的低频
    x = np.hstack([rec_a[:,1].reshape(rec_a.shape[0],1),rec_d[:,0].reshape(rec_a.shape[0],1)])
    x_data1 = np.hstack([x_data1,x])  
    data= pd.DataFrame(x_data1)
    cols = list(data.columns[:])
    print(cols)
    print(cols[0])
    print(data.head())
    data[6] = data[6].fillna(method='ffill').fillna(method='bfill')
    X = np.zeros((len(data), input_seq_len, (dataNum+2)))
    y = np.zeros((len(data),output_seq_len,n_out_features))
    for i, name in enumerate(cols): 
        for j in range(input_seq_len):
            X[:, j, i] = data[name].shift(input_seq_len - j - 1).fillna(method='bfill')
    for j in range(output_seq_len):
        y[:, j, 0] = data[6].shift(- 1).fillna(method="bfill") 
    train_bound = int(0.85*(len(data)))
    X = X[input_seq_len:-(input_seq_len+1)]
    y = y[input_seq_len:-(input_seq_len+1)] 
    X_train = X[input_seq_len:train_bound]
    X_test = X[train_bound:]
    y_train = y[input_seq_len:train_bound]
    y_test = y[train_bound:]
    X_train_min, X_train_max = X_train.min(axis=0), X_train.max(axis=0)
    y_train_min, y_train_max = y_train.min(axis=0), y_train.max(axis=0)
    input_seq = (X_train - X_train_min)/(X_train_max - X_train_min + 1e-9)
    inputSeqTest = (X_test - X_train_min)/(X_train_max - X_train_min + 1e-9)
    output_seq = (y_train - y_train_min)/(y_train_max - y_train_min +1e-9)
    outputSeqTest = (y_test - y_train_min)/(y_train_max - y_train_min + 1e-9)

    print(len(data))
    print("input_seq:",input_seq.shape)
    print("output_seq:",output_seq.shape)
    print("inputSeqTest:",inputSeqTest.shape)
    print("outputSeqTest:",outputSeqTest.shape)
    print(inputSeqTest[0])
    print(outputSeqTest[0])
    return input_seq,output_seq,inputSeqTest,outputSeqTest,X_train_max,X_train_min,y_train_max,y_train_min

 
def label_data(data_file):
    data = pd.read_csv(data_file)
    cols = list(data.columns[:])
    data['pollution_pm2.5'] = data['pollution_pm2.5'].fillna(method='ffill').fillna(method='bfill')
    input_X = np.zeros((len(data), input_seq_len, len(cols)-1))
    input_Y = np.zeros((len(data),input_seq_len-1,1))
    label_Y = np.zeros((len(data),1,1))

    for i, name in enumerate(cols):
        if i==6: break
        for j in range(input_seq_len):
            input_X[:, j, i] = data[name].shift(input_seq_len - j - 1).fillna(method='bfill')
    for j in range(input_seq_len-1):
            input_Y[:, j, 0] = data['pollution_pm2.5'].shift(input_seq_len - j - 1).fillna(method='bfill')
    label_Y = data['pollution_pm2.5'].shift(0).fillna(method='ffill').values

    input_X = input_X[input_seq_len:-input_seq_len-1]
    input_Y = input_Y[input_seq_len:-input_seq_len-1]
    label_Y = label_Y[input_seq_len:-input_seq_len-1]
    input_X = np.array(input_X).reshape(len(input_X),input_seq_len,len(cols)-1)
    input_Y = np.array(input_Y).reshape(len(input_X),input_seq_len-1,1)
    label_Y = np.array(label_Y).reshape(len(input_X),1)  

    train_bound = int(0.85*(len(data)))+1
    X_train = input_X[:train_bound]
    X_test = input_X[train_bound:]
    y_train = input_Y[:train_bound]
    y_test = input_Y[train_bound:]
    label_train = label_Y[:train_bound]
    label_test = label_Y[train_bound:]
    X_train_min, X_train_max = X_train.min(axis=0), X_train.max(axis=0)
    y_train_min, y_train_max = y_train.min(axis=0), y_train.max(axis=0)
    label_train_min, label_train_max = label_train.min(axis=0), label_train.max(axis=0)

    input_X_train = (X_train - X_train_min)/(X_train_max - X_train_min + 1e-9)
    input_X_test = (X_test - X_train_min)/(X_train_max - X_train_min + 1e-9)

    input_Y_train = (y_train - y_train_min)/(y_train_max - y_train_min +1e-9)
    input_Y_test = (y_test - y_train_min)/(y_train_max - y_train_min + 1e-9)

    label_Y_train = (label_train - label_train_min)/(label_train_max - label_train_min +1e-9)
    label_Y_test = (label_test - label_train_min)/(label_train_max - label_train_min + 1e-9) 
    return input_X_train,input_X_test,input_Y_train,input_Y_test,label_Y_train,label_Y_test,X_train_max,X_train_min,y_train_max,y_train_min,label_train_max,label_train_min


def origin_data(data_file):
    data = pd.read_csv(data_file)
    cols = list(data.columns[:])
    data['pollution_pm2.5'] = data['pollution_pm2.5'].fillna(method='ffill').fillna(method='bfill')
    X = np.zeros((len(data), input_seq_len, len(cols)))
    y = np.zeros((len(data),output_seq_len,n_out_features))
    for i, name in enumerate(cols):
        for j in range(input_seq_len):
            X[:, j, i] = data[name].shift(input_seq_len - j - 1).fillna(method='bfill')
    for j in range(output_seq_len):
        y[:, j, 0] = data["pollution_pm2.5"].shift(-1).fillna(method="bfill")
    X = X[input_seq_len:-(input_seq_len+1)]
    y = y[input_seq_len:-(input_seq_len+1)] 
    train_bound = int(0.85*(len(data)))
    X_train = X[:train_bound]
    X_test = X[train_bound:]
    y_train = y[:train_bound]
    y_test = y[train_bound:]
    X_train_min, X_train_max = X_train.min(axis=0), X_train.max(axis=0)
    y_train_min, y_train_max = y_train.min(axis=0), y_train.max(axis=0)
    input_seq = (X_train - X_train_min)/(X_train_max - X_train_min + 1e-9)
    inputSeqTest = (X_test - X_train_min)/(X_train_max - X_train_min + 1e-9)
    output_seq = (y_train - y_train_min)/(y_train_max - y_train_min +1e-9)
    outputSeqTest = (y_test - y_train_min)/(y_train_max - y_train_min + 1e-9)
    return input_seq,inputSeqTest,output_seq,outputSeqTest,X_train_max,X_train_min,y_train_max,y_train_min
