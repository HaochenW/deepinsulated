# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:01:22 2021

@author: hcwan
"""

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
import Levenshtein
matplotlib.use('Agg')
import copy

random_seed = 3
seq_num = 1000000
seq_len = 31
# For all exp generation
# select_loc = [3/4, 1/2, 1/3,1/4, 1/6, 1/8, 1/12, 1/16, 1/24, 1/32, 1/48, 1/64, 1/96, 1/128]
# mutation_num = 8

# For divergent seuqences in certain exp generation
select_loc = [1/4]
mutation_num = 20

def CNN_model(promoter_length):
    model = Sequential()
    model.add(
            Conv2D(100, (5, 1),
            padding='same',
            input_shape=(promoter_length, 1, 4))
            )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(200, (5, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    return model



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
        file = pd.read_excel(open('../data/deal_data_.xlsx', 'rb'),
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


# Input: List of sequences, expected mutant nucleotides number;
# Output: List of sequences after mutation
def random_mutation(sequence, mutant_num = 10, generated_num = 10000):
    nu_dict = ['T','C','G','A']
    real_mutantnum = int(mutant_num * 4 / 3) + 1
    mutant_seq = []
    mould_sequence = []
    for i in range(generated_num):
        random.seed(i*10 + random_seed)
        current_sequence = random.choice(sequence)
        current_mould_seq = copy.copy(current_sequence)
        for j in range(real_mutantnum):
            random.seed(i*10 + j +  random_seed)
            random_num = random.randint(0,len(current_sequence)-1)
            random.seed(i*1001 + j +  random_seed)
            current_sequence = current_sequence[0:random_num] + random.choice(nu_dict) + current_sequence[random_num+1::]
        mutant_seq.append(current_sequence)
        mould_sequence.append(current_mould_seq)
    return mutant_seq, mould_sequence
            
            
## Get the select range of each promoter strength
## Output:select_range: 0. low_range of LB exp; 1. high_range of LB exp 2. low_range of M9 exp 3. high_range of M9 exp
def get_select_range():        
    LB_highest = 66504.8
    M9_highest = 10651.1
    select_range = []
    for item in select_loc:
        low_range_LB = LB_highest * item * 0.75
        high_range_LB = LB_highest * item * 1.25
        low_range_M9 = M9_highest * item * 0.75
        high_range_M9 = M9_highest * item * 1.25
        select_range.append([low_range_LB,high_range_LB,low_range_M9,high_range_M9])
    return select_range

# Prediciton
def predict(sequence_onehot):
    model = CNN_model(seq_len) 
    model.load_weights('weight_CNN_M9.h5')
    result_M9 = model.predict(sequence_onehot, verbose=0)[:,0]
    result_M9 = [2 ** x for x in result_M9]
    
    model.load_weights('weight_CNN_LB_small2.h5')
    result_LB = model.predict(sequence_onehot, verbose=0)[:,0]
    result_LB = [2 ** x for x in result_LB]
    return result_M9, result_LB

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
    return record_distance[1], arg_sort[1]



def remove_simarity_sequence(str_A, str_B_groups, sim_th = 15):
    flag = 0
    for j in range(0,len(str_B_groups)):    
        result = Levenshtein.editops(str_A, str_B_groups[j]) # catch the editting steps of two sequences
            
        record = []  # Get the different region
        for item in result:
            if item [0] == 'delete' or item[0] == 'replace' or (item[0] == 'insert' and item[2] == seq_len):
                record.append(item[1])
                record = list(set(record)) #remove repeating location
                
        tmp_dissim_region = []  # Get the dissimilar region in the str_A
        for k in range(0,seq_len):
            if k in record:
                tmp_dissim_region.append(k)
        
        for k in range(len(tmp_dissim_region) - 1):
            if tmp_dissim_region[k+1] - tmp_dissim_region[k] > sim_th:
                flag = True
        try:
            if tmp_dissim_region[0] >  sim_th or tmp_dissim_region[-1] < seq_len - sim_th + 1:
                flag = True
        except:
            pass
    return flag

if __name__ == '__main__':
    

    # Read the intial data
    sequence_origin,LB_exp,M9_exp,control,record_dict,record_average = get_data()
    select_range = get_select_range()
    
    
    # Generate the random sequence
    sequence, mould_sequence = random_mutation(sequence_origin, generated_num = seq_num, mutant_num = mutation_num)
    # sequence  = generate_random(seq_num, seq_len)
    sequence_onehot = seq2onehot(sequence)
    sequence_onehot = sequence_onehot.reshape(len(sequence_onehot),len(sequence_onehot[0]),1,4)
    
    
    
    result_M9, result_LB = predict(sequence_onehot)
    
    # sort_result_M9 = result_M9.sort(reverse=True)
    # sort_result_LB = result_LB.sort(reverse=True)
    
    ## Parallel test: Test if M9/LB prediciton results are in the same range
    plt.figure(10000)
    plt.scatter(result_M9,result_LB,s=2,color="#2b83ba")
    plt.xlabel('result_M9')
    plt.ylabel('result_LB')
    plt.title('The prediction results of LB/M9 model')
    plt.savefig('scatter plot_M9_LB_predict.jpg')
    
    # The cof of M9 and LB prediction results
    cof = scipy.corrcoef(result_LB,result_M9)
    spearman = scipy.stats.spearmanr(result_LB,result_M9)
    print('pearson_cor:' + str(cof[0,1]))
    print('spearman_cor:' + str(spearman))
    
    # intialize the selected sequence list
    final_sequence = []
    for i in range(len(select_range)):
        final_sequence.append([])
    
    # Select sequence in certain exp range
    record_sequence = []
    for i in range(seq_num):
        for j in range(len(select_range)):
            if result_LB[i] > select_range[j][0] and result_LB[i] < select_range[j][1] and \
                result_M9[i] > select_range[j][2] and result_M9[i] < select_range[j][3]:
                    final_sequence[j].append([sequence[i],result_LB[i],result_M9[i],mould_sequence[i]])
                    record_sequence.append(sequence[i])
    
    # all_sequence_to_be_compared
    all_compared_sequence = record_sequence + sequence_origin
        
    # Get the edit distance and select edit distance > 5 sequences
    for i in range(len(final_sequence)):
        not_permit_sequence = []
        for j in range(len(final_sequence[i])):
            distance,argdistance = get_edit_distance(final_sequence[i][j][0],all_compared_sequence)
            if  distance <= 5:
                not_permit_sequence.append(final_sequence[i][j])
            # elif remove_simarity_sequence(final_sequence[i][j][0],all_compared_sequence) == True:
            #     not_permit_sequence.append(final_sequence[i][j])  # Too strong, abandon
            else:
                final_sequence[i][j].append(distance)
                final_sequence[i][j].append(all_compared_sequence[argdistance])
        for item in not_permit_sequence:
            final_sequence[i].remove(item)
        
        if i % 2 == 0:
            print(i)
    
    
    # Output the results
    with open('selected_sequence.csv','w') as f:
        f.write('Sequence, Exp_LB, Exp_M9, Original_sequence, The smallest edit distance, The smallest edit distance sequence, Exp percentage + \n')
        for i in range(len(final_sequence)):
            j = 0
            for item in final_sequence[i]:
                if j < 10086:
                    f.write(item[0] + ',' + str(item[1]) + ',' + str(item[2]) + ',' + str(item[3]) + ',' + str(item[4]) +\
                            ',' + str(item[5]) + ',' + str(select_loc[i]) + '\n')
                    j = j + 1

























