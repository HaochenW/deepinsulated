# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
import numpy as np
from Bio import motifs
import random
import os 
import pandas as pd
from keras import backend as K
import copy
from Bio import motifs
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import random
import math

random_seed = 1000

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def CNN_model_small(promoter_length):
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
    model.add(Conv2D(200, (5, 1),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(5, (5, 1),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(1))
    return model


# Generate certain number of random sequences
def generate_random(seq_num, seq_len):
    nu_dict = ['T','C','G','A']
    sequence = []
    for i in range(seq_num):
        tmp_seq = ''
        for j in range(seq_len):
            # random.seed(i*1005 + j + random_seed)
            tmp_seq = tmp_seq + random.choice(nu_dict)
        sequence.append(tmp_seq)
    return sequence


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
    onehot = np.array(onehot)
    onehot = onehot.reshape(len(onehot),len(onehot[0]),1,4)
    return onehot

def single_onehot2seq(sub_onehot):
    sub_seq = ''
    ref = ['T','C','G','A']
    for item in sub_onehot:
        max_num = np.argmax(item) 
        sub_seq = sub_seq + ref[max_num]
    return sub_seq
        
def onehot2seq(onehot):
    if len(onehot) == 1:
        return single_onehot2seq(onehot)
    else:
        seq_list = []
        for item in onehot:
            seq_list.append(single_onehot2seq(item))
        return seq_list
      
def get_data():
    record_dict = {}
    record_average = []
    sequence = []
    LB_exp = []
    M9_exp = []
    control = []
    for i in range(15):
        dict_name = 'file_' + str(i)
        file = pd.read_excel(open('./data/deal_data_.xlsx', 'rb'),
                  sheet_name = str(i+1) + '-1-94')  
        file = file.dropna(subset=['Tube Name:'])
        record_dict[dict_name] = file
        exp = list(file['LB-ACI(诱导14h)'])
        record_average.append(np.mean(exp))
        
        # sequence and exp
        sequence = sequence + list(file['序列'])[0:-2]
        LB_exp = LB_exp + list(file['LB-ACI(诱导14h)'])[0:-2]
        M9_exp = M9_exp + list(file['M9-ACI(诱导3h)'])[0:-2]
        control = control + list(file['LB-AC（不诱导）'])[0:-2]
    return sequence,LB_exp,M9_exp,control,record_dict,record_average

def read_experiment1():
    test_file = 'Sequence with fixed expression new(1).csv'
    info = pd.read_table(test_file,sep=',')
    start_seq = info['most_similar_sequence']
    end_seq = info['sequence']
    exp = info[' exp_LB']
    return start_seq, end_seq, exp

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


# Calculate the smallest edit distance between two groups
def get_edit_distance(str_A,str_B_groups):
    record_distance = []
    for j in range(0,len(str_B_groups)):
        tmp_distance = levenshteinDistance(str_A, str_B_groups[j])
        record_distance.append(tmp_distance)
    
    # Get the smallest edit distance(except itself)
    arg_sort = np.array(record_distance).argsort()
    record_distance.sort()
    return record_distance, arg_sort


if __name__ == '__main__':
    weight_path = 'weight_CNN_LB_small2.h5'
    model = CNN_model_small(promoter_length = 31)
    model.load_weights(weight_path)
    
    output = model.layers[-1].output
    loss = K.mean(output[:,0])
    grads = K.gradients(loss, model.input)[0]
    func = K.function([model.input], [loss, grads])
    
    
    input_img = generate_random(100, 31) # define an initial random image
    input_img = seq2onehot(input_img)
    input_img = np.float64(input_img)

    lr = 1.  # learning rate used for gradient updates
    max_iter = 1000  # number of gradient updates iterations
    for i in range(max_iter):
        loss_val, grads_val = func([input_img])
        input_img += grads_val * lr  # update the image based on gradients
        if i % 10 == 0:
            print(i)
    
    ## Select high expression random sequence
    th = 0
    seq_list = onehot2seq(input_img)
    seq_onehot = seq2onehot(seq_list)
    result = model.predict(seq_onehot)
    result = np.power(2,result)
    # seq_list = np.array(seq_list).reshape(len(seq_list),1)
    # seq_list = seq_list[result>th]
    # result = result[result>th]
    # seq_list = seq_list.tolist()

    
    sequence,LB_exp,M9_exp,control,record_dict,record_average = get_data()
    start_seq, end_seq, end_seq_exp = read_experiment1()
    end_seq = list(end_seq)
    end_seq = np.array(end_seq).reshape(len(end_seq),1)
    end_seq = end_seq[end_seq_exp > th]
    end_seq_exp = end_seq_exp[end_seq_exp > th]
    
    # preserve the identical sequences
    preserve_end_seq = []
    preserve_evolve_seq = []
    for i,item in enumerate(result):
        for j in range(len(end_seq_exp)):
            if item > end_seq_exp[j] * 0.95 and item < end_seq_exp[j] * 1.05:
                if j not in preserve_end_seq:
                    preserve_end_seq.append(j)
                    preserve_evolve_seq.append(seq_list[i])
    
    end_seq = np.array(end_seq)[preserve_end_seq][:,0]
    end_seq = end_seq.tolist()
    seq_list = preserve_evolve_seq
    sequence_group = sequence + end_seq + seq_list
                
        
    

    record_distance = []
    for i,item in enumerate(end_seq):
        distance, arg_sort = get_edit_distance(item, sequence_group)
        record_distance = record_distance + distance[1::]
        if i %10 == 0:
            print(i)
            
    record_distance_experiment = []
    sequence_perm = np.random.permutation(sequence)
    for i,item in enumerate(sequence_perm[0:200]):
        distance, arg_sort = get_edit_distance(item, sequence_group)
        record_distance_experiment = record_distance_experiment + distance[1::]
        if i %10 == 0:
            print(i)
            
    record_distance_random = []
    for i,item in enumerate(seq_list):
        distance, arg_sort = get_edit_distance(item, sequence_group)
        record_distance_random = record_distance_random + distance[1::]
        if i % 10 == 0:
            print(i)
    
    
    plt.hist(record_distance, bins=np.arange(0,18)+0.5, facecolor="blue", edgecolor="black", alpha=0.7, density=True, label = 'optimized_' + str(len(end_seq)))
    plt.hist(record_distance_experiment, bins=np.arange(0,18)+0.5, facecolor="red", edgecolor="black", alpha=0.7, density=True, label = 'origin_' + str(len(sequence)))
    plt.hist(record_distance_random, bins=np.arange(0,18)+0.5, facecolor="orange", edgecolor="black", alpha=0.7, density=True, label = 'random_optimized_' + str(len(seq_list)))
    plt.locator_params(axis='x', integer=True)
    plt.xlabel("Distance Range")
    plt.ylabel("Frequency Distribution")
    plt.title("Frequency Distribution Picture")
    plt.legend()
    plt.savefig('Frequency Distribution Picture.png')
    plt.show()
    
    
    
        
        
    
    
    
    































