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

def get_data():
    record_dict = {}
    record_average = []
    sequence = []
    LB_exp = []
    M9_exp = []
    control = []
    for i in range(15):
        dict_name = 'file_' + str(i)
        file = pd.read_excel(open('deal_data_.xlsx', 'rb'),
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

# Generate certain number of random sequences
def generate_random(seq_num, seq_len):
    nu_dict = ['T','C','G','A']
    sequence = []
    for i in range(seq_num):
        tmp_seq = ''
        for j in range(seq_len):
            random.seed(i*1005 + j + random_seed)
            tmp_seq = tmp_seq + random.choice(nu_dict)
        sequence.append(tmp_seq)
    return sequence

def read_mutant_seq(data_loc = './mutant_seq'):
    file_name = os.listdir(data_loc)
    sequence = []
    for item in file_name:
        tmp_info = pd.read_table(os.path.join(data_loc,item),sep=',')
        tmp_seq = list(tmp_info['seq'])
        for sub_item in tmp_seq:
            sequence.append(sub_item[-45:-14])
    return sequence

class Feature_extracter:
    def __init__(self, model_name):
        self.model = model_name
    
    def extract_featuremap(self, info_input, layer_num, model_weight_path):
        self.model.load_weights(model_weight_path)
        outputs = [layer.output for layer in self.model.layers]
        active_func = K.function([self.model.input], [outputs])
        activations = active_func(info_input)[0]
        feature_map = activations[0][layer_num]
        return feature_map

def read_experiment3():
    file = pd.read_excel(open('2022.6.28__all prediction.xlsx', 'rb'),
                         sheet_name = "去除诱导前预测")  
    file = file.dropna(subset=['sequence'])
    original_predictexp = file[' old_exp_LB']
    control = file['LB-AC']
    experiment_exp = file['LB-平均']
    sequence = file['sequence']
    experiment_increased = file['差值']
    return sequence, original_predictexp, experiment_exp, control, experiment_increased

    

if __name__ == '__main__': 
    # Load the model
    weight_path = 'weight_CNN_control.h5'
    model = CNN_model_small(promoter_length = 31)
    model.load_weights(weight_path)
    

    # Load the datasets
    sequence_mutant = read_mutant_seq()
    # sequence_random = generate_random(seq_num = 100000, seq_len =31)
    onehot_mutant = seq2onehot(sequence_mutant)
    # onehot_random = seq2onehot(sequence_random)
    LB_exp_mutant = model.predict(onehot_mutant)[:,0].tolist()
    # LB_exp_random = model.predict(onehot_random)[:,0].tolist()
    sequence,LB_exp_experiment,M9_exp,control,record_dict,record_average = get_data()
    LB_exp_experiment = np.log2(LB_exp_experiment).tolist()
    onehot_experiment = seq2onehot(sequence)
    # onehot = np.concatenate((onehot_mutant,onehot_random,onehot_experiment))
    onehot = onehot_mutant
    LB_exp = LB_exp_mutant
    
    ## Just for test
    LB_exp = LB_exp_experiment
    onehot = onehot_experiment
    LB_exp = np.power(2, LB_exp)
    
    
    feature_extractor = Feature_extracter(model)
    feature_map4 = feature_extractor.extract_featuremap(onehot, 7, weight_path)
    feature_map4 = feature_map4.reshape(len(onehot),15,5)
    m = feature_map4.max(axis=1)
    m_pos = feature_map4.argmax(axis=1)
    # onehot = onehot.reshape(len(onehot),len(onehot[0]),4)
    pad = np.ones((14, 4)) * np.array([0, 0, 0, 0])
    
    
    exp = model.predict(onehot)

    for i in range(4):
        for j in range(4):
            if  i < j:
                dim1 = np.max(feature_map4[:,:,i], axis = 1)
                dim2 = np.max(feature_map4[:,:,j], axis = 1)
                ngridx = 30
                ngridy = 30
                # Create grid values first.
                # xi = np.linspace(min(dim1) - max(dim1) * 0.1, max(dim1) + max(dim1) * 0.1, ngridx)
                # yi = np.linspace(min(dim2) - max(dim1) * 0.1, max(dim2) + max(dim1) * 0.1, ngridy)
                xi = np.linspace(min(dim1), max(dim1), ngridx)
                yi = np.linspace(min(dim2), max(dim2), ngridy)
                
                # Perform linear interpolation of the data (x,y)
                # on a grid defined by (xi,yi)
                triang = tri.Triangulation(dim1, dim2)
                interpolator = tri.LinearTriInterpolator(triang, exp)
                Xi, Yi = np.meshgrid(xi, yi)
                zi = interpolator(Xi, Yi)
                np.savez("mat_Kernel {} and Kernel {}.npz".format(str(i+1), str(j+1)), name1=Xi, name2=Yi, name3 = zi)
                
                np.savez("experimental_mat_Kernel {} and Kernel {}.npz".format(str(i+1), str(j+1)),name1=dim1,name2=dim2,name3=LB_exp)
                
    # 100余个实验数据点
    sequence, original_predictexp, experiment_exp, control, experiment_increased= read_experiment3()
    sequence = list(sequence)
    onehot_exper = seq2onehot(sequence)
    feature_extractor = Feature_extracter(model)
    feature_map4 = feature_extractor.extract_featuremap(onehot_exper, 7, weight_path)
    feature_map4 = feature_map4.reshape(len(onehot_exper,),15,5)
    m = feature_map4.max(axis=1)
    m_pos = feature_map4.argmax(axis=1)
    onehot = onehot.reshape(len(onehot),len(onehot[0]),4)
    pad = np.ones((14, 4)) * np.array([0, 0, 0, 0])

    for i in range(4):
        for j in range(4):
            if  i < j:
                dim1 = np.max(feature_map4[:,:,i], axis = 1)
                dim2 = np.max(feature_map4[:,:,j], axis = 1)
                np.savez("original_mat_Kernel {} and Kernel {}.npz".format(str(i+1), \
                    str(j+1)),name1=dim1,name2=dim2,name3=experiment_exp)
 
    
    
    
    
    
# Translate to .mat file
from scipy.io import savemat
import numpy as np
import glob
import os
npzFiles = glob.glob("*.npz")
for f in npzFiles:
    fm = os.path.splitext(f)[0]+'.mat'
    d = np.load(f, allow_pickle=True)
    savemat(fm, d)
    print('generated ', fm, 'from', f)
    
    
    
    
    