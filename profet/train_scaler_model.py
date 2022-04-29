# profet: train scaler prediction model

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Load dataset and setting for training scaler models
Path('./model/scaler').mkdir(parents=True, exist_ok=True)
ANCHOR_INSTANCE = 'g3s.xlarge'
SCALER_MODE = 'MinMax-2nd'
merge_data = pickle.load(open('../data/pre_data/merge_preprocessing_g3.pickle', 'rb'))
merge_data['dataset'] = merge_data['dataset'].str.split('dataset').str[0]
merge_data['dataset'] = pd.to_numeric(merge_data['dataset'])
dataset_list = list(merge_data['dataset'].value_counts().index)
model_list = list(merge_data['model'].value_counts().index)
batchsize_list = list(merge_data['batchsize'].value_counts().index)
instance_list = list(merge_data['instance_name'].value_counts().index)

# Function-1: build_batchsize_scaler
# Train and save batchsize scaler model by condition.
# We train different model by instance and dataset size, then find most accurate model.
def build_batchsize_scaler(merge_data, instance_name, dataset_size):
    condition = ((merge_data['instance_name'] == instance_name)&
                 (merge_data['dataset'] == dataset_size))
    
    groupby_data = merge_data[condition].groupby(
        by=['batchsize', 'model']).mean()['batch_latency']
    print(len(groupby_data), end=' ')
    
    latency_dict = {}
    for i in range(len(groupby_data)):
        batchsize = groupby_data.index[i][0]
        model_name = groupby_data.index[i][1]
        if model_name not in latency_dict.keys():
            latency_dict[model_name] = {}
        latency_dict[model_name][batchsize] = groupby_data.iloc[i]
    
    bs_list = []
    latency_list = []
    for model_name in latency_dict.keys():
        batchsize_dict = latency_dict[model_name]
        if len(batchsize_dict) != 5:
            continue
        bs_list += list(batchsize_dict.keys())
        origin_latency = np.array(list(batchsize_dict.values())).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_latency = scaler.fit(origin_latency).transform(origin_latency).flatten().tolist()
        latency_list += scaled_latency
        
    X = np.array(bs_list).reshape(-1, 1)
    y = np.array(latency_list).reshape(-1, 1)
    
    poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_model.fit(X, y)
    
    predictor_name = f"./model/scaler/scaler_{instance_name}_{dataset_size}dataset_BSpred.model"
    pickle.dump(poly_model, open(predictor_name, 'wb'))

# Function-2: build_dataset_scaler
# Train and save dataset scaler model by condition.
# We train different model by instance and batchsize, then find most accurate model.
# Same process as batchsize scaler
def build_dataset_scaler(merge_data, instance_name, batchsize):
    condition = ((merge_data['instance_name'] == instance_name)&
                 (merge_data['batchsize'] == batchsize))
    groupby_data = merge_data[condition].groupby(
        by=['dataset', 'model']).mean()['batch_latency']
    print(len(groupby_data), end=' ')
    
    latency_dict = {}
    for i in range(len(groupby_data)):
        dataset_size = groupby_data.index[i][0]
        model_name = groupby_data.index[i][1]
        if model_name not in latency_dict.keys():
            latency_dict[model_name] = {}
        latency_dict[model_name][dataset_size] = groupby_data.iloc[i]
    
    ds_list = []
    latency_list = []
    for model_name in latency_dict.keys():
        dataset_size_dict = latency_dict[model_name]
        if len(dataset_size_dict) != 5:
            continue
        ds_list += list(dataset_size_dict.keys())
        origin_latency = np.array(list(dataset_size_dict.values())).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_latency = scaler.fit(origin_latency).transform(origin_latency).flatten().tolist()
        latency_list += scaled_latency
    
    X = np.array(ds_list).reshape(-1, 1)
    y = np.array(latency_list).reshape(-1, 1)
    
    poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_model.fit(X, y)
    
    predictor_name = f"./model/scaler/scaler_{instance_name}_{batchsize}batchsize_DSpred.model"
    pickle.dump(poly_model, open(predictor_name, 'wb'))    

# 02-1-1. Train Batchsize Scaler & Save Models
for instance_name in sorted(instance_list):
    for dataset_size in sorted(dataset_list):
        build_batchsize_scaler(merge_data, instance_name, dataset_size)

# 02-1-2. Train Dataset Size Scaler & Save Models
for instance_name in sorted(instance_list):
    for batchsize in sorted(batchsize_list):
        build_dataset_scaler(merge_data, instance_name, batchsize)
