
# -*- coding: utf-8 -*-

# ./jaspar2meme -pfm ./K1.5_kernel1_difference_permed/ > kernel_K1_5_diff_origin.meme
# ./tomtom -oc kernel_K1_5_diff_origin --norc -thresh 1 kernel_K1_5_diff_origin.meme kernel_K1_5_diff_origin.meme


import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers import Flatten
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
import seaborn as sns
import shutil
from sklearn import linear_model 
from matplotlib.backends.backend_pdf import PdfPages
import logomaker
from scipy import stats



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

## Read the data from T7 promoter sequence and expression files
def get_data_T7():
    #1. Read data
    record_dict = {}
    record_average = []
    sequence = []
    LB_exp = []
    M9_exp = []
    control = []
    for i in range(15):
        dict_name = 'file_' + str(i)
        file = pd.read_excel(open('2022.5.27-T7 Library_all data.xlsx', 'rb'),
                  sheet_name = str(i+1))  
        file = file.dropna(subset=['Tube Name:'])
        file = file.dropna(subset=['序列'])
        file = file[file['序列'].str.len() == 21] #Only sequence length == 21 would be involved
        file = file[~file['序列'].str.contains('-')] #Only sequence length == 21 would be involved
        file['备注'] = file['备注'].fillna('0')
        file = file[~file['备注'].str.contains('杂峰')] #
        record_dict[dict_name] = file
        exp = list(file['LB-ACI(诱导14h)'])
        record_average.append(np.mean(exp))
        
        # sequence and exp
        sequence = sequence + list(file['序列'])[0:-2]
        LB_exp = LB_exp + list(file['LB-ACI(诱导14h) 平均'])[0:-2]
        M9_exp = M9_exp + list(file['M9-ACI(诱导3h) 平均'])[0:-2]
        control = control + list(file['LB-AC（不诱导）平均'])[0:-2]
        sequence = [item.upper() for item in sequence]
    return sequence,LB_exp,M9_exp,control,record_dict,record_average,file


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

class Feature_extracter:
    def __init__(self, model_name):
        self.model = model_name
    
    def extract_featuremap(self, info_input, layer_num, model_weight_path):
        self.model.load_weights(model_weight_path)

        # Get all layer outputs
        outputs = [layer.output for layer in self.model.layers]

        # Create backend function
        active_func = K.function([self.model.input], outputs)

        # Wrap info_input in a list!
        activations = active_func([info_input])

        # Get the feature map of the desired layer
        feature_map = activations[layer_num]

        return feature_map

    
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

def draw_logo(n_kernel, outpath, pwm_folder, file_name, filter_size = 5):
    import logomaker as lm
    from math import sqrt, floor, ceil
    row = floor(sqrt(n_kernel))
    col = ceil(n_kernel / row)
    fig = plt.figure(figsize=(col * 4, row * 4))
    gs = plt.GridSpec(row * 2, col, figure=fig, height_ratios=[10, 1] * row)
    for i in range(n_kernel):
        ax = fig.add_subplot(gs[i // col * 2, i % col])
        pwm = np.zeros((4,filter_size))
        kernel = motifs.read(open(pwm_folder + file_name[i]),"pfm")
        try: 
            nt_counts_all = {'A': kernel.pwm['A'], 'T': kernel.pwm['T'], 
                                 'C': kernel.pwm['C'], 'G': kernel.pwm['G']}
        
            pwm = pd.DataFrame(nt_counts_all)
            lm.Logo(pwm, ax=ax, color_scheme="classic")
            ax.set_ylim([0, 2])
            ax.set_xticks(list(range(filter_size)))
            ax.text(
                filter_size // 2, 1.96, 
                file_name[i],
                ha='center', va='top', fontsize=12
                )
            ax.set_yticks([])
        except:
            pass
    plt.savefig(outpath, transparent=False)

def list_file(path, file_type = ".pfm"):
    record = []
    for file in os.listdir(path):
         # check the files which are end with specific extension
        if file.endswith(file_type):
            # print path name of selected files
            record.append(file)
    return record

def find_location(list1):
    from collections import defaultdict
    set1 = set(list1)
    res_dict = defaultdict(list)
    
    for x in set1:
        for i, y in enumerate(list1):
            if x == y:
                res_dict[x].append(i)
    return res_dict


def extract_pwm(feature_map, onehot_sequences, kernel_size):
    num_sequences, sequence_length, num_filters = feature_map.shape
    pwm_list = []
    onehot_sequences = onehot_sequences.reshape(len(onehot),31,4)
    for i in range(num_filters):
        # Initialize a PWM matrix for the current filter
        pwm = np.zeros((kernel_size, onehot_sequences.shape[2]))

        for j in range(num_sequences):
            # Find the position of the maximum activation for the current filter in the current sequence
            max_activation_position = np.argmax(feature_map[j, :, i])
            # Get the activation score at this position
            activation_score = feature_map[j, max_activation_position, i]
            # Add the weighted one-hot enct_sne_featureoded sequence to the PWM
            pwm += activation_score * onehot_sequences[j, max_activation_position]

        pwm_list.append(pwm)

    return pwm_list

def generate_sequence_logos(pwm_matrices, mode_pos, output_pdf, t_sne_feature, logos_per_page=4):
    """
    Generate sequence logos from a list of PWM matrices and save them to a PDF file.

    Parameters:
    pwm_matrices (list of np.array): A list of PWM matrices, each as a numpy array.
    output_pdf (str): The file path for the output PDF containing the sequence logos.
    logos_per_page (int): Number of logos to display per page.
    """
    with PdfPages(output_pdf) as pdf:
        num_pages = int(np.ceil(len(pwm_matrices) / logos_per_page))
        
        record_flag = 0
        for page in range(num_pages):
            # Calculate the number of rows and columns for the grid
            num_logos = min(logos_per_page, len(pwm_matrices) - page * logos_per_page)
            num_cols = int(np.ceil(np.sqrt(num_logos)))
            num_rows = int(np.ceil(num_logos / num_cols))
            
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8.5, 6))
            axes = axes.flatten() if num_logos > 1 else [axes]
            
            for i in range(num_logos):
                index = page * logos_per_page + i
                if index >= len(pwm_matrices):
                    break
                
                pwm_matrix = pwm_matrices[index]
                pwm_df = pd.DataFrame(pwm_matrix.T, columns=['T', 'C', 'G', 'A'])

                # Normalize the PWM to get probabilities
                prob_df = pwm_df.div(pwm_df.sum(axis=1), axis=0)

                # Calculate the information content
                entropy = -np.nansum(prob_df * np.log2(prob_df), axis=1)
                information_content = np.log2(4) - entropy

                # Multiply the probabilities by the information content
                ic_df = prob_df.multiply(information_content, axis=0)

                # Create a sequence logo
                logo = logomaker.Logo(ic_df, ax=axes[i])

                # Customize the logo
                logo.style_spines(visible=False)
                logo.style_spines(spines=['left', 'bottom'], visible=True)
                logo.ax.set_ylabel('Bits')
                logo.ax.set_title(
    f'Logo {index + 1} pos {mode_pos[record_flag]} '
    f'[{t_sne_feature[record_flag][0]:.2f}, {t_sne_feature[record_flag][1]:.2f}]'
)
              
                record_flag = record_flag + 1

            # Remove any unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            # Adjust layout and save the current figure to the PDF
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
            

if __name__ == '__main__': 
    # weight_path = 'weight_CNN_LB_small_origin.h5'
    weight_path = 'weight_CNN_LB_small_diff.h5'
    model = CNN_model_small(promoter_length = 31)
    model.load_weights(weight_path)
    

    sequence,LB_exp_experiment,M9_exp,control,record_dict,record_average = get_data()
    onehot_experiment = seq2onehot(sequence)
    
    onehot = onehot_experiment
    
    feature_extractor = Feature_extracter(model)
    feature_map1 = feature_extractor.extract_featuremap(onehot, 1, weight_path)
    feature_map1 = feature_map1.reshape(len(onehot),31,100)
    
    m = feature_map1.max(axis=1)
    m_pos = feature_map1.argmax(axis=1)
    onehot = onehot.reshape(len(onehot),len(onehot[0]),4)
    
    kernel_size = 5
    pad = np.ones((2, 4)) * np.array([0, 0, 0, 0])
    pfm_record_diff = []
    for i in range(len(m_pos[1])):
        print('.', end='')
        kernel_pos = m_pos[:,i]
        kernel_strength = m[:,i]
        pfm = np.zeros((kernel_size, 4))
        for j in range(len(m_pos)):
            original_seq = onehot[j]
            padded_seq = np.concatenate([pad, original_seq, pad], axis=0) 
            if kernel_strength[j] > 0:
                pfm += kernel_strength[j] * padded_seq[kernel_pos[j]:kernel_pos[j] + kernel_size]
        if np.sum(pfm[0]) == 0:
            continue
        ppm = pfm / np.sum(pfm[0])
        # seqlogo.seqlogo(seqlogo.CompletePm(ppm=ppm, background=[.31, .19, .19, .31]), format='png',
        #                 size='medium', resolution=300,
        #                 filename='{:02}.png'.format(i))
        # Change the location
        
        # pfm = np.trunc(pfm)
        pfm = pfm.T
        pfm = pfm + 0.001
        pfm_record_diff.append(pfm)
        
    mode_pos = stats.mode(m_pos,axis=0)[0]
    motif_order = np.argsort(mode_pos)[0,:]
    mode_pos = mode_pos.T[motif_order] + 1
    
        
    weight_path = 'weight_CNN_LB_small_origin.h5'
    # weight_path = 'weight_CNN_LB_small_diff.h5'
    model = CNN_model_small(promoter_length = 31)
    model.load_weights(weight_path)
    

    sequence,LB_exp_experiment,M9_exp,control,record_dict,record_average = get_data()
    onehot_experiment = seq2onehot(sequence)
    
    onehot = onehot_experiment
    
    feature_extractor = Feature_extracter(model)
    feature_map1 = feature_extractor.extract_featuremap(onehot, 1, weight_path)
    feature_map1 = feature_map1.reshape(len(onehot),31,100)
    
    m = feature_map1.max(axis=1)
    m_pos = feature_map1.argmax(axis=1)
    onehot = onehot.reshape(len(onehot),len(onehot[0]),4)
    
    kernel_size = 5
    pad = np.ones((2, 4)) * np.array([0, 0, 0, 0])
    pfm_record_origin = []
    for i in range(len(m_pos[1])):
        print('.', end='')
        kernel_pos = m_pos[:,i]
        kernel_strength = m[:,i]
        pfm = np.zeros((kernel_size, 4))
        for j in range(len(m_pos)):
            original_seq = onehot[j]
            padded_seq = np.concatenate([pad, original_seq, pad], axis=0) 
            if kernel_strength[j] > 0:
                pfm += kernel_strength[j] * padded_seq[kernel_pos[j]:kernel_pos[j] + kernel_size]
        if np.sum(pfm[0]) == 0:
            continue
        ppm = pfm / np.sum(pfm[0])
        # seqlogo.seqlogo(seqlogo.CompletePm(ppm=ppm, background=[.31, .19, .19, .31]), format='png',
        #                 size='medium', resolution=300,
        #                 filename='{:02}.png'.format(i))
        # Change the location
        
        # pfm = np.trunc(pfm)
        pfm = pfm.T
        pfm = pfm + 0.001
        pfm_record_origin.append(pfm)

    mode_pos_origin = stats.mode(m_pos,axis=0)[0]
    motif_order_origin = np.argsort(mode_pos_origin)[0,:]
    mode_pos_origin = mode_pos_origin.T[motif_order_origin] + 1
        

    pfm_record_diff = np.array(pfm_record_diff).reshape(len(pfm_record_diff), -1)
    pfm_record_origin = np.array(pfm_record_origin).reshape(len(pfm_record_diff), -1)
    

    
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # Assuming pfm_record_diff and pfm_record_origin are your datasets
    # with shapes (100, 20)
    # pfm_record_diff = np.array(pfm_record_diff).reshape(len(pfm_record_diff), -1)
    # pfm_record_origin = np.array(pfm_record_origin).reshape(len(pfm_record_diff), -1)
    
    # Combine the datasets for t-SNE
    combined_data = np.vstack((pfm_record_diff, pfm_record_origin))
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=40)
    # tsne = TSNE(n_components=2, perplexity=30, learning_rate=50, n_iter=1000, random_state=42)
    reduced_data = tsne.fit_transform(combined_data)
    
    
    # Perform PCA
    # pca = PCA(n_components=2)
    # reduced_data = pca.fit_transform(combined_data)
    
    # Split the reduced data back into the two original datasets
    reduced_diff = reduced_data[:len(pfm_record_diff)]
    reduced_origin = reduced_data[len(pfm_record_diff):]

    motif_order = np.argsort(motif_order)
    motif_order_origin = np.argsort(motif_order_origin)
    # Annotate each point with its motif_order
    for i, (x, y) in enumerate(reduced_diff):
        plt.annotate(str(motif_order[i]), (x, y), fontsize=8, alpha=0.6)

    for i, (x, y) in enumerate(reduced_origin):
        plt.annotate(str(motif_order[i]), (x, y), fontsize=8, alpha=0.6)
        
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_diff[:, 0], reduced_diff[:, 1], label='pfm_record_diff', alpha=0.7)
    plt.scatter(reduced_origin[:, 0], reduced_origin[:, 1], label='pfm_record_origin', alpha=0.7)
    # Annotate each point with its motif_order
    for i, (x, y) in enumerate(reduced_diff):
        plt.annotate(str(motif_order[i]), (x, y), fontsize=8, alpha=0.6)

    for i, (x, y) in enumerate(reduced_origin):
        plt.annotate(str(motif_order_origin[i]), (x, y), fontsize=8, alpha=0.6)
  
    plt.title('t-SNE Visualization of pfm_record_diff and pfm_record_origin')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.savefig('t-sne_layer1.pdf', dpi = 600)
    plt.show()
    
    
    # # Kernel Visualization - purified
    # pfm_record = np.array(pfm_record_diff)[motif_order]
    # t_sne_feature = reduced_diff[motif_order]
    # pfm_record  = np.array(pfm_record).reshape(len(pfm_record), 4,5)
    # generate_sequence_logos(pfm_record, mode_pos, 't_sne_layer1_diff.pdf', t_sne_feature, logos_per_page=4)
    
    # # Kernel Visualization - non-purified
    # pfm_record_origin = np.array(pfm_record_origin)[motif_order_origin]
    # t_sne_feature = reduced_origin[motif_order_origin]
    # pfm_record_origin  = np.array(pfm_record_origin).reshape(len(pfm_record_origin), 4,5)
    # generate_sequence_logos(pfm_record_origin, mode_pos_origin, 't_sne_layer1_origin.pdf', t_sne_feature, logos_per_page=4)
    
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Use combined_data for clustering
    best_k = 0
    best_score = -1
    max_k = min(30, len(combined_data) - 1)
    
    for k in range(5, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(combined_data)  
        score = silhouette_score(combined_data, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_centers = kmeans.cluster_centers_
    
    # Visualization in t-SNE space (reduced_data)
    plt.figure(figsize=(8, 6))
    for i in range(best_k):
        idx = np.where(best_labels == i)[0]
        cluster_points = reduced_data[idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}', alpha=0.6)
    
    # t-SNE projection of cluster centers is not meaningful, so skip or optionally use PCA to project them if desired
    
    plt.title(f't-SNE + K-means Clustering (on combined_data, k={best_k})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('t-sne_kmeans_clusters_legend_outside.pdf', dpi=600, bbox_inches='tight')
    
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.show()
    
    # Label types: first 100 are 'diff', next 100 are 'origin'
    labels_diff_origin = np.array(['diff'] * 100 + ['origin'] * 100)
    
    # Identify pure clusters
    pure_cluster_indices = []
    for i in range(best_k):
        idx = np.where(best_labels == i)[0]
        types = labels_diff_origin[idx]
        if np.all(types == 'diff') or np.all(types == 'origin'):
            pure_cluster_indices.append(i)
    
    # Plot pure clusters
    plt.figure(figsize=(8, 6))
    for i in pure_cluster_indices:
        idx = np.where(best_labels == i)[0]
        cluster_points = reduced_data[idx]
        label_type = labels_diff_origin[idx[0]]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    label=f'Pure Cluster {i} ({label_type})', alpha=0.8)
    
    plt.title('Pure Clusters (Only diff or only origin)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig('t-sne_pure_clusters.pdf', dpi=600, bbox_inches='tight')
    plt.show()



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        



