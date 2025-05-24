# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 20:39:10 2025

@author: hcwan
"""

# -*- coding: utf-8 -*-

import argparse
import math
import os
import random
import copy

import numpy as np
import pandas as pd
import scipy
import scipy.io as scio
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Keras/TensorFlow imports
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
    UpSampling2D
)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import LSTM, Bidirectional, Attention
import tensorflow as tf
from tensorflow.keras.layers import Reshape



import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

MEDIUM_SIZE = 8
SMALLER_SIZE = 6
BIGGER_SIZE = 25
plt.rc('font', family='Arial', size=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)	# fontsize of the axes title
plt.rc('xtick', labelsize=SMALLER_SIZE)	# fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)	# fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
FIG_HEIGHT = 2
FIG_WIDTH = 2



def read_experiment_random_ecoli(control_out = False):
    file = pd.read_excel(open('./2022.6.28__all prediction_add mammalian_ 预测k1.5 Promoter__with randomized initiation_nonrandom initiation-v3.xlsx', 'rb'),
                         sheet_name = "20220118---random_optimiz结果 ")  
    file = file.dropna(subset=['sequence'])
    file = file.dropna(subset=['LB-std'])
    file = file.dropna(subset=['LB-average'])
    file = file.dropna(subset=['new_predict'])
    file = file.dropna(subset=['old_predict'])
    sequence = file['sequence']
    predictexp = file['new_predict']
    experiment_exp = file['LB-average']
    experiment_std = file['LB-std']
    predictexp_origin = file['old_predict']
    control = file['LB-AC(uninduced)']
    if control_out == False:
        return sequence, predictexp, predictexp_origin, experiment_exp, experiment_std
    else:
        return sequence, predictexp, predictexp_origin, experiment_exp, experiment_std, control
    
    

# Set matplotlib backend
matplotlib.use('Agg')

def cnn_lstm_model(promoter_length):
    """Build and return a CNN-LSTM model for promoter sequence analysis.
    
    Args:
        promoter_length (int): Length of the promoter sequences.
    
    Returns:
        Sequential: Compiled CNN-LSTM model.
    """
    model = Sequential()
    
    # CNN part
    model.add(Conv2D(100, (5, 1), padding='same', 
                     input_shape=(promoter_length, 1, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    
    # LSTM part
    model.add(Reshape((-1, 100)))
    model.add(LSTM(200, return_sequences=True))
    model.add(LSTM(100, return_sequences=False))
    
    # Fully connected layers
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    
    model.add(Flatten())
    

    return model


def attention_model(promoter_length):
    """Build and return an Attention-based model for promoter sequence analysis.
    
    Args:
        promoter_length (int): Length of the promoter sequences.
    
    Returns:
        Sequential: Compiled Attention-based model.
    """
    # Input Tensor
    input_tensor = tf.keras.layers.Input(shape=(promoter_length, 1, 4))
    
    # CNN part
    conv = Conv2D(100, (5, 1), padding='same')(input_tensor)
    conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size=(2, 1))(conv)
    
    # Adjust dimensions for attention
    conv = tf.keras.layers.Reshape((-1, 100))(conv)  # Shape: [batch_size, new_seq_length, embedding_dim]

    # Attention part
    query = Dense(64)(conv)  # Query vector
    value = Dense(64)(conv)  # Value vector
    attention_layer, attention_scores = Attention()(
        inputs=[query, value],
        return_attention_scores=True
    )

    # Flatten the attention layer's output
    attention_layer = tf.keras.layers.Flatten()(attention_layer)
    
    # Fully connected layers
    dense = Dense(1024)(attention_layer)
    dense = Activation('tanh')(dense)
    output = Dense(1)(dense)

    output = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(output) 
    
    # Build the final model
    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    
    return model




def cnn_model(promoter_length):
    """Build and return a CNN model for promoter sequence analysis.
    
    Args:
        promoter_length (int): Length of the promoter sequences
    
    Returns:
        Sequential: Compiled CNN model
    """
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(100, (5, 1), padding='same', 
                     input_shape=(promoter_length, 1, 4)))
    model.add(Activation('relu'))
    
    # Second convolutional block
    model.add(Conv2D(200, (5, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    
    # Third convolutional block
    model.add(Conv2D(200, (5, 1)))
    model.add(Activation('relu'))
    
    # Fourth convolutional block
    model.add(Conv2D(5, (5, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(1))
    
    return model


def read_data_files(file_path='./'):
    """Read and concatenate CSV files from specified directory.
    
    Args:
        file_path (str): Path to directory containing CSV files
    
    Returns:
        tuple: (Combined DataFrame, List of prediction sequences)
    """
    file_list = os.listdir(file_path)
    result_df = None
    
    for file in file_list:
        if file.endswith('.csv') and 'prediction' not in file:
            if result_df is None:
                result_df = pd.read_csv(os.path.join(file_path, file))
            else:
                temp_df = pd.read_csv(os.path.join(file_path, file))
                result_df = pd.concat([result_df, temp_df])
    
    # Extract sequences from the last 64-14 positions
    predict_sequences = [seq[-64:-14] for seq in result_df['seq'].tolist()]
    
    return result_df, predict_sequences


def calculate_max_length(sequences):
    """Calculate maximum length from a list of sequences."""
    return max(len(seq) for seq in sequences)


def sequence_to_onehot(sequences):
    """Convert DNA sequences to one-hot encoded format.
    
    Args:
        sequences (list): List of DNA sequences
    
    Returns:
        numpy.ndarray: One-hot encoded sequence array
    """
    encoding_map = {
        'T': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'A': [0, 0, 0, 1],
        'M': [0, 0, 0, 0]  # Padding character
    }
    
    max_length = calculate_max_length(sequences)
    encoded_seqs = []
    
    for seq in sequences:
        # Pad sequence with 'M' to max length
        padded_seq = seq.ljust(max_length, 'M')
        # Convert each character to one-hot encoding
        encoded_seq = [encoding_map[nt] for nt in padded_seq]
        encoded_seqs.append(encoded_seq)
        
    return np.array(encoded_seqs)


def split_train_test(data, labels, test_size=0.3, shuffle=True):
    """Split dataset into training and testing sets.
    
    Args:
        data (numpy.ndarray): Input features
        labels (numpy.ndarray): Target labels
        test_size (float): Proportion of data to use for testing
        shuffle (bool): Whether to shuffle the data before splitting
    
    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """
    if shuffle:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
    
    split_idx = int(len(data) * (1 - test_size))
    return (data[:split_idx], labels[:split_idx],
            data[split_idx:], labels[split_idx:])


def load_experiment_data():
    """Load and process experimental data from Excel files.
    
    Returns:
        tuple: Contains various experimental measurements and metadata
    """
    experiment_data = {}
    average_values = []
    all_sequences = []
    lb_measurements = []
    m9_measurements = []
    control_measurements = []
    
    # Initialize figure for data visualization
    plt.figure(0)
    
    for exp_num in range(15):
        sheet_name = f'{exp_num+1}-1-94'
        df = pd.read_excel('deal_data_.xlsx', sheet_name=sheet_name)
        df = df.dropna(subset=['Tube Name:'])
        
        # Store experiment data
        exp_key = f'experiment_{exp_num}'
        experiment_data[exp_key] = df
        
        # Process LB measurements
        lb_values = df['LB-ACI(诱导14h)'].tolist()
        average_values.append(np.mean(lb_values))
        
        # Visualize data distribution
        plt.hist(lb_values, bins=5, label=exp_key, alpha=0.5)
        plt.legend()
        
        # Collect sequence and measurement data
        all_sequences.extend(df['序列'].tolist()[:-2])
        lb_measurements.extend(lb_values[:-2])
        m9_measurements.extend(df['M9-ACI(诱导3h)'].tolist()[:-2])
        control_measurements.extend(df['LB-AC（不诱导）'].tolist()[:-2])
    
    return (all_sequences, lb_measurements, m9_measurements,
            control_measurements, experiment_data, average_values)



def train_one_time(data, labels, model_name):
    """Train and evaluate a single model once for debugging.
    
    Args:
        data (numpy.ndarray): Input features.
        labels (numpy.ndarray): Target labels.
        model_name (str): Name of the model to use ("CNN", "CNN-LSTM", "Attention").
    
    Returns:
        dict: Contains evaluation metrics for debugging (pearson and spearman coefficients).
    """
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras import optimizers
    import numpy as np
    import scipy.stats
    
    # Split the data with a smaller training size for faster debugging
    train_size = 0.8
    (X_train, y_train, 
     X_test, y_test) = split_train_test(data, labels, test_size=1-train_size)

    # Create the corresponding model
    if model_name == "CNN":
        model = cnn_model(calculate_max_length(sequences))
    elif model_name == "CNN-LSTM":
        model = cnn_lstm_model(calculate_max_length(sequences))
    elif model_name == "Attention":
        model = attention_model(calculate_max_length(sequences))
    else:
        raise ValueError("Unknown model name!")

    # Configure the optimizer
    optimizer = optimizers.SGD(learning_rate=0.0005, decay=1e-6, 
                                momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=optimizer)
    
    # Configure callbacks
    callbacks = [
        EarlyStopping(patience=10),
        ModelCheckpoint(f'weights_{model_name}.keras', save_best_only=True)
    ]
    
    # Train the model
    model.fit(X_train, y_train,
              epochs=100,
              batch_size=16,
              validation_split=0.1,
              callbacks=callbacks,
              verbose=1)  # Enable verbosity for debugging
    
    # Load the best weights
    model.load_weights(f'weights_{model_name}.keras')
    
    # Predict on test data
    predictions = model.predict(X_test)
    
    predictions = predictions.squeeze()
    data = data.squeeze()
    
    # Evaluate metrics
    pearson = scipy.stats.pearsonr(predictions, y_test)[0]
    spearman = scipy.stats.spearmanr(predictions, y_test)[0]
    
    print(f"Finished training {model_name}.")
    print(f"Pearson Correlation: {pearson}")
    print(f"Spearman Correlation: {spearman}")
    
    return model


# Update the train and evaluate function to include these new structures
def train_and_evaluate_models(data, labels, model_names, num_iterations=90):
    """Train and evaluate multiple models (CNN, CNN-LSTM, Attention).
    
    Args:
        data (numpy.ndarray): Input features.
        labels (numpy.ndarray): Target labels.
        model_names (list): Names of models to use.
        num_iterations (int): Number of different training sizes to test.
    
    Returns:
        dict: Contains evaluation metrics for each model structure.
    """
    results = {}
    
    for model_key in model_names:
        print(f"Training model: {model_key}")
        pearson_coeffs = []    
        spearman_coeffs = []      
        training_sizes = []    
        for iteration in range(num_iterations):
            num_repeats = 3
            current_pearson = []   
            current_spearman = []
            for _ in range(num_repeats):
                # Split data with increasing training size
                train_size = (iteration + 5) / 100
                (X_train, y_train, 
                 X_test, y_test) = split_train_test(data, labels, 
                                                    test_size=1-train_size)
                
                # Create corresponding models
                if model_key == "CNN":
                    model = cnn_model(calculate_max_length(sequences))
                elif model_key == "CNN-LSTM":
                    model = cnn_lstm_model(calculate_max_length(sequences))
                elif model_key == "Attention":
                    model = attention_model(calculate_max_length(sequences))
                else:
                    raise ValueError("Unknown model name!")
                
                optimizer = optimizers.SGD(learning_rate=0.0005, decay=1e-6, 
                                          momentum=0.9, nesterov=True)
                model.compile(loss='mse', optimizer=optimizer)
                
                # Callbacks configuration
                callbacks = [
                    EarlyStopping(patience=10),
                    ModelCheckpoint(f'weights_{model_key}.keras', save_best_only=True)
                ]
                
                # Model training
                model.fit(X_train, y_train,
                          epochs=1000,
                          batch_size=8,
                          validation_split=0.1,
                          callbacks=callbacks,
                          verbose=0)
                
                # Model evaluation
                model.load_weights(f'weights_{model_key}.keras')
                predictions = model.predict(X_test).flatten()
                
                # Calculate metrics
                pearson = scipy.stats.pearsonr(predictions, y_test)[0]
                spearman = scipy.stats.spearmanr(predictions, y_test)[0]
                
                current_pearson.append(pearson)
                current_spearman.append(spearman)
            
            # Store iteration results
            pearson_coeffs.append(np.mean(current_pearson))
            spearman_coeffs.append(np.mean(current_spearman))
            training_sizes.append(len(X_train))
        
        results[model_key] = {
            "pearson_coeffs": pearson_coeffs,
            "spearman_coeffs": spearman_coeffs,
            "training_sizes": training_sizes
        }
    
    return results


def visualize_results(all_model_results, x_label, y_label, title, filename):
    """Visualize performance metrics of multiple models in one plot with error bars.
    
    Args:
        all_model_results (dict): Dictionary containing results for each model.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Plot title.
        filename (str): Name of the file to save the plot.
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, results in all_model_results.items():
        training_sizes = results["training_sizes"]
        pearson_coeffs = results["pearson_coeffs"]
        spearman_coeffs = results.get("spearman_coeffs", None)  # Optional
        pearson_stds = results.get("pearson_stds", None)        # Optional
        spearman_stds = results.get("spearman_stds", None)      # Optional
        
        # Plot Pearson coefficients
        plt.errorbar(training_sizes, pearson_coeffs,
                     yerr=pearson_stds if pearson_stds else None,
                     fmt='o', label=f'{model_name} - Pearson',
                     capsize=3, alpha=0.7)
        
        # # Plot Spearman coefficients (optional, if available)
        # if spearman_coeffs:
        #     plt.errorbar(training_sizes, spearman_coeffs,
        #                  yerr=spearman_stds if spearman_stds else None,
        #                  fmt='s', label=f'{model_name} - Spearman',
        #                  capsize=3, alpha=0.7)
    
    # Global plot settings
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy import stats
    import numpy as np

    NUM_REPEATS = 3
    r2_scores = []

    all_pred = []
    all_experiment = None  # We'll set this once

    for i in range(NUM_REPEATS):
        print(f"--- Run {i+1} ---")

        # Load data
        (sequences, lb_values, m9_values,
         controls, experiments, averages) = load_experiment_data()
        
        lb_processed = np.array(lb_values) - np.array(controls)
        lb_processed = lb_processed - np.min(lb_processed) + 1
        lb_processed = np.log2(lb_processed)

        seq_encoded = sequence_to_onehot(sequences)
        seq_encoded = seq_encoded.reshape(seq_encoded.shape[0], 
                                          seq_encoded.shape[1], 
                                          1, 4)
        
        model_CNN = train_one_time(seq_encoded, lb_processed, "CNN")

        sequence, predictexp, _, experiment_exp, experiment_std, control = read_experiment_random_ecoli(control_out=True)

        Test_encode = sequence_to_onehot(sequence)
        Test_encode = Test_encode.reshape(Test_encode.shape[0], 
                                          Test_encode.shape[1], 
                                          1, 4)
        
        predictexp = model_CNN.predict(Test_encode).squeeze()
        predictexp = 2**predictexp

        max_std_threshold = 10000
        mask = experiment_std < max_std_threshold
        filtered_predictexp = predictexp[mask]
        filtered_experiment_exp = experiment_exp[mask]
        filtered_experiment_std = experiment_std[mask]
        filtered_control = control[mask]
        mean = np.mean(filtered_control)
        filtered_predictexp = filtered_predictexp + mean

        # Log transform
        log_pred = np.log10(filtered_predictexp)
        log_exp = np.log10(filtered_experiment_exp)
        log_std = filtered_experiment_std / (filtered_experiment_exp * np.log(10))

        # Calculate Pearson R^2
        r_val = stats.pearsonr(filtered_predictexp, filtered_experiment_exp)[0]
        r2 = r_val ** 2
        r2_scores.append(r2)
        print("Pearson R^2 =", r2)

        # Store for plotting
        all_pred.append(log_pred)
        if all_experiment is None:
            all_experiment = log_exp
            all_std = log_std

    # Average predictions across runs
    all_pred = np.array(all_pred)
    mean_pred = np.mean(all_pred, axis=0)
    std_pred = np.std(all_pred, axis=0)

    # Plot mean prediction with error bars
    fig, ax = plt.subplots(figsize=(1.2 * FIG_WIDTH * 1.2, FIG_HEIGHT))

    ax.errorbar(
        mean_pred,
        all_experiment,
        xerr=std_pred,
        yerr=all_std,
        fmt="",
        capsize=2,
        capthick=0.5,
        linestyle="",
        linewidth=1,
        color="black",
    )
    ax.scatter(
        mean_pred,
        all_experiment,
        s=5,
        color="black",
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xticks(np.log10([10, 100, 1000, 10000, 100000]))
    ax.set_xticklabels([10, 100, 1000, 10000, 100000])
    ax.set_yticks(np.log10([10, 100, 1000, 10000, 100000]))
    ax.set_yticklabels([10, 100, 1000, 10000, 100000])

    csfont = {"family": "Arial"}
    ax.set_xlabel("Predicted", fontdict=csfont)
    ax.set_ylabel("Experiment", fontdict=csfont)
    ax.set_xlim([np.log10(10), np.log10(100000)])
    ax.set_ylim([np.log10(10), np.log10(100000)])
    fig.tight_layout()
    plt.savefig("predicted_experiment_random_ecoli_log_transformed_avg3.pdf")

    # Report average R^2
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    print(f"\nAverage Pearson R^2 over {NUM_REPEATS} runs: {mean_r2:.4f} ± {std_r2:.4f}")


    
    
    
    
    
    
    
    
    
