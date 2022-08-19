# profet: train and validate anchor prediction model

# Load Python Packages
# 1. Basic Data Processing Packages
import pandas as pd
import numpy as np
import math
import pickle
# 2. Machine Learning Packages (Modeling, Clustering, Errors)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
# 3. Deep Learning Package
import tensorflow as tf
# 4. Other Packages
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--anchor_instance', type=str)
args = parser.parse_args()

# 01-1. Experiments Setting
# Global Variables
ANCHOR_INSTANCE = args.anchor_instance
INSTANCE_LIST = ['g3s.xlarge', 'g4dn.xlarge', 'p2.xlarge', 'p3.2xlarge']
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

# 01-3. Modeling and Validation
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

# Function-2: train_test_model
# 1. Get train/test dataset and target instance(pred_instance)
# 2. Split dataset into feature and target (x, y)
# 3. train three single model (linear, random forest, dnn)
# 4. predict with three model, then median ensemble to results
# 5. return prediction results
def train_test_model(train_data, test_data, pred_instance, drop_cols):
    train_x = train_data.drop(drop_cols + PRED_INSTANCES, axis=1)
    train_simple_x = np.array(train_data['batch_latency']).reshape(-1, 1)
    train_y = train_data[pred_instance]
    
    test_x = test_data.drop(drop_cols + PRED_INSTANCES, axis=1)
    test_simple_x = np.array(test_data['batch_latency']).reshape(-1, 1)
    test_y = test_data[[pred_instance]].to_numpy()
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    model_dnn = build_dnn_model((train_x.shape[1],))
    model_dnn.fit(train_x, train_y, epochs=200,
              callbacks=[callback],
              batch_size=16,
              verbose=0)
    model_rfr = RandomForestRegressor()
    model_rfr.fit(train_x, train_y)
    model_simple = LinearRegression()
    model_simple.fit(train_simple_x, train_y)
    
    dnn_pred_y = model_dnn.predict(test_x)
    dnn_pred_y = dnn_pred_y.reshape(-1, 1)
    rfr_pred_y = model_rfr.predict(test_x)
    rfr_pred_y = rfr_pred_y.reshape(-1, 1)
    simple_pred_y = model_simple.predict(test_simple_x)
    simple_pred_y = simple_pred_y.reshape(-1, 1)
    median_pred_y = np.median(np.stack([
        dnn_pred_y, rfr_pred_y, simple_pred_y
    ]), axis=0)
    return test_y, median_pred_y, test_data

# Function-3: model_validation
# 1. run model-validation and save result to dictionary of dictionary
#    {'g3s.xlarge': {'AlexNet': ... , 'VGG16': ...}, 'g4dn.xlarge': {...}}
# 2. split traing and test dataset by model(dnn-architecture) condition
# 3. execute train_test_model function, the save result to dictionary
def model_validation():
    pred_instance_dict = {}
    for pred_instance in PRED_INSTANCES:
        pred_model_dict = {}
        for val_model in model_list:
            train_data = anchor_data[anchor_data['model'] != val_model]
            test_data = anchor_data[anchor_data['model'] == val_model]
            print(f"Validation Model: {val_model}")
            print(f"Train Data Size: {len(train_data)}")
            print(f"Test Data Size: {len(test_data)}")
            
            test_y, pred_y, test_data = train_test_model(
                train_data, test_data, pred_instance,
                latency_cols + workload_cols + PRED_INSTANCES)
            
            pred_model_dict[val_model] = (test_y, pred_y, test_data)
        pred_instance_dict[pred_instance] = pred_model_dict
    return pred_instance_dict

pred_instance_dict = model_validation()

# 01-4. Preprocessing Validation Results
# Function-4: build_true_pred_df
# convert dictionary to dataframe
def build_true_pred_df(pred_instance_dict, instance, model):
    true_y = pred_instance_dict[instance][model][0]
    pred_y = pred_instance_dict[instance][model][1]
    test_df = pred_instance_dict[instance][model][2]
    pred_df = test_df[['model', 'dataset', 'batchsize', 'exp_name']]
    
    pred_df[f"{instance}_true"] = true_y
    pred_df[f"{instance}_pred"] = pred_y

    pred_df['dataset'] = pred_df['dataset'].str.split('dataset').str[0]
    pred_df['batchsize'] = pd.to_numeric(pred_df['batchsize'])
    pred_df['dataset'] = pd.to_numeric(pred_df['dataset'])
    pred_df['anchor_mape'] = abs((true_y - pred_y) / true_y) * 100

    return pred_df

all_result_true_pred_df = []
for test_instance in PRED_INSTANCES:
    single_target_true_pred_df_list = []
    for test_model in model_list:
        true_pred_df = build_true_pred_df(pred_instance_dict, test_instance, test_model)
        single_target_true_pred_df_list.append(true_pred_df)
    single_target_true_pred_df = pd.concat(single_target_true_pred_df_list)
    all_result_true_pred_df.append(single_target_true_pred_df)  


# 01-5. Print and Save Results
all_true = []
all_pred = []
for i in range(len(PRED_INSTANCES)):
    print(f'{ANCHOR_INSTANCE} to {PRED_INSTANCES[i]} result')
    true = all_result_true_pred_df[i][f'{PRED_INSTANCES[i]}_true']
    pred = all_result_true_pred_df[i][f'{PRED_INSTANCES[i]}_pred']
    all_true.append(true)
    all_pred.append(pred)

    print(f'MAPE: {mean_absolute_percentage_error(true, pred) * 100}')
    print(f'R2: {r2_score(true, pred)}')
    print(f'RMSE: {math.sqrt(mean_squared_error(true, pred))} us')
    print(f'RMSE: {math.sqrt(mean_squared_error(true, pred))/1000} ms')
    print()

print(f'{ANCHOR_INSTANCE} Instance All Results')
all_true_df = pd.concat(all_true)
all_pred_df = pd.concat(all_pred)
print(f'MAPE: {mean_absolute_percentage_error(all_true_df, all_pred_df) * 100}')
print(f'R2: {r2_score(all_true_df, all_pred_df)}')
print(f'RMSE: {math.sqrt(mean_squared_error(all_true_df, all_pred_df))} us')
print(f'RMSE: {math.sqrt(mean_squared_error(all_true_df, all_pred_df))/1000} ms')

pickle.dump(all_result_true_pred_df, open(f'{ANCHOR_INSTANCE}_anchor_median.pickle', 'wb'))
