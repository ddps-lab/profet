# profet: train anchor prediction model

# Load Python Packages
# 1. Basic Data Processing Packages
import pandas as pd
import numpy as np
import pickle
# 2. Machine Learning Packages (Modeling, Clustering, Errors)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
# 3. Deep Learning Package
import tensorflow as tf
# 4. Other Packages
from tqdm import tqdm
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--anchor_instance', type=str)
args = parser.parse_args()


# 01-1. Experiments Setting
# Global Variables
INSTANCE_LIST = ['g3s.xlarge', 'g4dn.xlarge', 'p2.xlarge', 'p3.2xlarge']
ANCHOR_INSTANCE = args.anchor_instance
PRED_INSTANCES = [x for x in INSTANCE_LIST if x != ANCHOR_INSTANCE]
ANCHOR_NAME = ANCHOR_INSTANCE[:2]
CLUSTER_METHOD = 'average'
CLUSTER_THRESHOLD = 6

# Load and check anchor dataset
anchor_data = pickle.load(open(f"../data/anchor_data/anchor_{ANCHOR_NAME}.pickle", 'rb'))
anchor_data.head()

# Get variables from anchor dataset for clustering and modeling
columns = list(anchor_data.columns)
host_cols = [x for x in columns if x.startswith('Host_')]
device_cols = [x for x in columns if x.startswith('Device_')]
latency_cols = ['epoch_latency', 'batch_latency']
workload_cols = ['instance_name', 'model', 'dataset', 'optimizer', 'batchsize', 'exp_name']
model_list = sorted(list(anchor_data['model'].value_counts().index))
dataset_list = sorted(list(anchor_data['dataset'].value_counts().index))
batchsize_list = sorted(list(anchor_data['batchsize'].value_counts().index))
exp_list = sorted(list(anchor_data['exp_name'].value_counts().index))

# 01-2. Feature Engineering with NLP Clustering
# Function-1: levenshtein
# calculate levenshtein distance of two string (str_x, str_y)
def levenshtein(str_x, str_y):
    size_x = len(str_x) + 1
    size_y = len(str_y) + 1
    matrix = np.zeros((size_x, size_y))
    
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if str_x[x-1] == str_y[y-1]:
                matrix[x, y] = min(matrix[x-1, y]+1, matrix[x-1, y-1], matrix[x, y-1]+1)
            else:
                matrix[x, y] = min(matrix[x-1, y]+1, matrix[x-1, y-1]+1, matrix[x, y-1]+1)
    return matrix[size_x-1, size_y-1]

# Calculate levenshtein distance matrix for every device operation pairs
feature_names = [x[7:] for x in device_cols]
dist_matrix = pd.DataFrame(0, index=feature_names, columns=feature_names)
for x in feature_names:
    for y in feature_names:
        dist_matrix[x][y] = levenshtein(x, y)

# Apply hierarchical clustering to distance matrix
cluster = fcluster(linkage(squareform(dist_matrix), CLUSTER_METHOD), CLUSTER_THRESHOLD, criterion='distance')
cluster_feature = {i: [] for i in range(len(pd.DataFrame(cluster).value_counts()))}
for index, value in enumerate(feature_names):
    cluster_feature[cluster[index]-1].append(value)

# Apply feature aggregation to anchor dataset
for key, value in cluster_feature.items():
    value = ["Device_" + x for x in value]
    anchor_data["&".join(value)] = 0
    for feature in value:
        anchor_data["&".join(value)] += anchor_data[feature]
    anchor_data.drop(value, axis=1, inplace=True)

# 01-3. Train and Save Median Ensemble Models
# define deep neural network regression model with custom learning rate, loss, layers
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9)

def build_dnn_model(input_shape):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation="relu", input_shape=input_shape))
    model.add(tf.keras.layers.Dense(64, activation="relu"))  
    model.add(tf.keras.layers.Dense(32, activation="relu"))  
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=['mean_absolute_percentage_error', 'mean_squared_error'],
                  loss_weights=[1., 1.])
    return model

# Function-2: train_save_model
def train_save_model():
    drop_cols = latency_cols + workload_cols + PRED_INSTANCES
    for pred_instance in tqdm(PRED_INSTANCES):
        train_x = anchor_data.drop(drop_cols + PRED_INSTANCES, axis=1)
        train_simple_x = np.array(anchor_data['batch_latency']).reshape(-1, 1)
        train_y = anchor_data[pred_instance]

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        model_dnn = build_dnn_model((train_x.shape[1],))
        model_dnn.fit(train_x, train_y, epochs=200, callbacks=[callback], batch_size=16, verbose=0)
        model_rfr = RandomForestRegressor()
        model_rfr.fit(train_x, train_y)
        model_simple = LinearRegression()
        model_simple.fit(train_simple_x, train_y)

        PRED_NAME = pred_instance[:2]
        Path('./model/anchor').mkdir(parents=True, exist_ok=True)
        pickle.dump(model_rfr, open(f'./model/anchor/anchor_{ANCHOR_NAME}_{PRED_NAME}_rfr.model', 'wb'))
        pickle.dump(model_simple, open(f'./model/anchor/anchor_{ANCHOR_NAME}_{PRED_NAME}_linear.model', 'wb'))
        model_dnn.save(f'./model/anchor/anchor_{ANCHOR_NAME}_{PRED_NAME}_dnn.model', save_format="tf")

train_save_model()
