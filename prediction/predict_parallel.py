# -*- coding: utf-8 -*-

from tensorflow.keras import optimizers
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from tensorflow.keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import random
import argparse
import math
import scipy.io as scio 
import scipy 
import scipy.stats
import copy
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')



def CNN_model(promoter_length):
    model = Sequential()
    model.add(
            Conv2D(100, (5, 1),
            padding='same',
            input_shape=(promoter_length, 1, 4))
            )
    model.add(Activation('relu'))
    model.add(Conv2D(200, (5, 1),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(200, (5, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(200, (5, 1),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    return model


def read_file(file_locate = './'):
    file_list = os.listdir(file_locate)
    flag = 0
    for item in file_list:
        if '.csv' in item and flag == 0 and 'prediction' not in item:
            result = pd.read_csv(item)
            flag = 1
        elif '.csv' in item and 'prediction' not in item:
            sub_file = pd.read_csv(item)
            result = pd.concat([result,sub_file])
    predict_seq = []
    for item in list(result['seq']):
        predict_seq.append(item[-64:-14])     
    
    return result, predict_seq

# Find the maximum length of promoter sequence
def max_len(seq):
    maxLength = max(len(x) for x in seq )
    return maxLength

def seq2onehot(seq):
    ref = {'T':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'A':[0,0,0,1],'M':[0,0,0,0]}
    maxlen = max_len(seq)
    onehot = []
    for item in seq:
        item = item + 'M' * (maxlen - len(item))
        tmp = []
        for letter in item:
            tmp.append(ref[letter])
        onehot.append(tmp)
        # 不足部分补0占位
    return np.array(onehot)


def train_test_split(sample, label, shuffle = True, split_th = 0.7):
    if shuffle == True:
        seed = list(range(len(sample)))
        random.shuffle(seed)
        sample = sample[seed]
        label = label[seed]
    # train_test_split
    train_size = int(len(sample)*split_th)
    train_sample = sample[0:train_size]
    train_label = label[0:train_size]
    
    test_sample = sample[train_size::]
    test_label = label[train_size::]

    return  train_sample, train_label, test_sample, test_label


def get_data():
    record_dict = {}
    record_average = []
    sequence = []
    LB_exp = []
    M9_exp = []
    control = []
    plt.figure(0)
    for i in range(15):
        dict_name = 'file_' + str(i)
        file = pd.read_excel(open('deal_data_.xlsx', 'rb'),
                  sheet_name = str(i+1) + '-1-94')  
        file = file.dropna(subset=['Tube Name:'])
        record_dict[dict_name] = file
        exp = list(file['LB-ACI(诱导14h)'])
        record_average.append(np.mean(exp))
        plt.hist(exp, bins=5, label=dict_name, alpha = 0.5)
        plt.legend()
        
        # sequence and exp
        sequence = sequence + list(file['序列'])[0:-2]
        LB_exp = LB_exp + list(file['LB-ACI(诱导14h)'])[0:-2]
        M9_exp = M9_exp + list(file['M9-ACI(诱导3h)'])[0:-2]
        control = control + list(file['LB-AC（不诱导）'])[0:-2]
    return sequence,LB_exp,M9_exp,control,record_dict,record_average

sequence,LB_exp,M9_exp,control,record_dict,record_average = get_data()
# 去除诱导前表达量
LB_exp = np.array(LB_exp) - np.array(control)
M9_exp = np.array(M9_exp) - np.array(control)

# 将数值变为正
LB_exp = LB_exp - min(LB_exp) + 1
M9_exp = M9_exp - min(M9_exp) + 1

# 取log值,预防梯度爆炸
LB_exp = np.log2(LB_exp)
M9_exp = np.log2(M9_exp)
sequence_onehot = seq2onehot(sequence)
sequence_onehot = sequence_onehot.reshape(len(sequence_onehot),len(sequence_onehot[0]),1,4)

record_cof = []
record_trainnum = []
record_spearman = []
record_cof_std = []
record_spearman_std = []
for i in range(55):
    if i < 40:
        iter_num = 3
    else:
        iter_num = 6
    tmp_record_pearson = []
    tmp_record_spearman = []
    for j in range(iter_num):
        train_feature, train_label, test_feature, test_label = train_test_split(sequence_onehot, LB_exp, split_th = (i+25)/100)
        model = CNN_model(max_len(sequence))
        sgd = optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model.fit(train_feature,train_label, epochs = 1000, batch_size = 8, validation_split=0.1, \
                  callbacks=[EarlyStopping(patience=10),ModelCheckpoint(filepath='weight_CNN_LB.h5',save_best_only=True)],shuffle=True) 
        model.load_weights('weight_CNN_LB.h5')
        loss_and_metrics = model.evaluate(test_feature, test_label, batch_size=1)
        result = model.predict(test_feature, verbose=0)[:,0]
        cof = scipy.corrcoef(result,test_label)
        spearman = scipy.stats.spearmanr(result,test_label)
        print('pearson_cor:' + str(cof[0,1]))
        print('spearman_cor:' + str(spearman))
        tmp_record_pearson.append(cof[0,1])
        tmp_record_spearman.append(spearman[0])
    record_cof.append(np.mean(tmp_record_pearson))
    record_cof_std.append(np.std(tmp_record_pearson))
    record_spearman.append(np.mean(tmp_record_spearman))
    record_spearman_std.append(np.std(tmp_record_spearman))
    record_trainnum.append(len(train_feature))
    
plt.figure(2)
plt.errorbar(record_trainnum,record_cof,yerr=record_cof_std,fmt='o',ecolor='black',color='b',elinewidth=0.5,capsize=1,marker='.')
plt.errorbar(record_trainnum,record_spearman,yerr=record_spearman_std,fmt='o',ecolor='black',color='r',elinewidth=0.5,capsize=1,marker='.')
plt.plot(record_trainnum,record_cof,label = 'pearson_cor',color='b')
plt.plot(record_trainnum,record_spearman,label = 'spearman_cor',color='r')
plt.xlabel('train_num')
plt.ylabel('correlation')
plt.legend()
plt.title('relationship LB_exp_average 15_experiments')
plt.savefig('relationship_record_LB_exp_average.jpg')

plt.figure(4)
plt.scatter(result,test_label,s=2,color="#2b83ba")
plt.xlabel('test_exp')
plt.ylabel('predict_exp')
y_lim = plt.xlim()
x_lim = plt.xlim()
plt.plot(x_lim, y_lim, 'k-', color = 'r')
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.title('scatter plot for the last iteration LB')
plt.savefig('scatter plot_LB_exp.jpg')

sequence,LB_exp,M9_exp,control,record_dict,record_average = get_data()
LB_exp = np.log2(LB_exp)
M9_exp = np.log2(M9_exp)
sequence_onehot = seq2onehot(sequence)
sequence_onehot = sequence_onehot.reshape(len(sequence_onehot),len(sequence_onehot[0]),1,4)

record_cof = []
record_trainnum = []
record_spearman = []
record_cof_std = []
record_spearman_std = []
for i in range(55):
    if i < 40:
        iter_num = 3
    else:
        iter_num = 6
    tmp_record_pearson = []
    tmp_record_spearman = []
    for j in range(iter_num):
        train_feature, train_label, test_feature, test_label = train_test_split(sequence_onehot, M9_exp, split_th = (i+25)/100)
        model = CNN_model(max_len(sequence))
        sgd = optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model.fit(train_feature,train_label, epochs = 1000, batch_size = 8, validation_split=0.1, \
                  callbacks=[EarlyStopping(patience=10),ModelCheckpoint(filepath='weight_CNN_M9.h5',save_best_only=True)],shuffle=True) 
        model.load_weights('weight_CNN_M9.h5')
        loss_and_metrics = model.evaluate(test_feature, test_label, batch_size=1)
        result = model.predict(test_feature, verbose=0)[:,0]
        cof = scipy.corrcoef(result,test_label)
        spearman = scipy.stats.spearmanr(result,test_label)
        print('pearson_cor:' + str(cof[0,1]))
        print('spearman_cor:' + str(spearman))
        tmp_record_pearson.append(cof[0,1])
        tmp_record_spearman.append(spearman[0])
    record_cof.append(np.mean(tmp_record_pearson))
    record_cof_std.append(np.std(tmp_record_pearson))
    record_spearman.append(np.mean(tmp_record_spearman))
    record_spearman_std.append(np.std(tmp_record_spearman))
    record_trainnum.append(len(train_feature))

plt.figure(3)
plt.errorbar(record_trainnum,record_cof,yerr=record_cof_std,fmt='o',ecolor='black',color='b',elinewidth=0.5,capsize=1,marker='.')
plt.errorbar(record_trainnum,record_spearman,yerr=record_spearman_std,fmt='o',ecolor='black',color='r',elinewidth=0.5,capsize=1,marker='.')
plt.plot(record_trainnum,record_cof,label = 'pearson_cor',color='b')
plt.plot(record_trainnum,record_spearman,label = 'spearman_cor',color='r')
plt.xlabel('train_num')
plt.ylabel('correlation')
plt.legend()
plt.title('relationship M9_exp 15_experiments')
plt.savefig('relationship_record_M9_exp_average.jpg')

plt.figure(5)
plt.scatter(result,test_label,s=2,color="#2b83ba")
plt.xlabel('test_exp')
plt.ylabel('predict_exp')
y_lim = plt.xlim()
x_lim = plt.xlim()
plt.plot(x_lim, y_lim, 'k-', color = 'r')
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.title('scatter plot for the last iteration M9')
plt.savefig('scatter plot_M9_exp.jpg')
    








