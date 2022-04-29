# profet: validate scaler prediction model

import pandas as pd
import numpy as np
import pickle
import math
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target_instance', type=str)
args = parser.parse_args()

# Global Variables for validation
ANCHOR_INSTANCE = args.target_instance
INSTANCE_LIST = ['g3s.xlarge', 'g4dn.xlarge', 'p2.xlarge', 'p3.2xlarge']
PRED_INSTANCES = [x for x in INSTANCE_LIST if x != ANCHOR_INSTANCE]
ANCHOR_NAME = ANCHOR_INSTANCE[:2]
SCALER_MODE = 'MinMax-2nd'

# Load Dataset from anchor validation
anchor_pred = pickle.load(open(f'./{ANCHOR_INSTANCE}_anchor_median.pickle', 'rb'))
model_list = sorted(list(anchor_pred[0]['model'].value_counts().index))
dataset_list = sorted(list(anchor_pred[0]['dataset'].value_counts().index))
batchsize_list = sorted(list(anchor_pred[0]['batchsize'].value_counts().index))
exp_list = sorted(list(anchor_pred[0]['exp_name'].value_counts().index))

instance_index = {}
for i in range(len(anchor_pred)):
    instance_name = [x.split('_pred')[0] for x in list(anchor_pred[i].columns) if '_pred' in x][0]
    instance_index[instance_name] = i

# Function-1: inference_batch_dataset
# This function inference scaled prediction value(0~1) from polynomial model,
# then convert scaled prediction value to real latency
def inference_batch_dataset(instance_name, model_name, latency_min, latency_max, size_pred, b_or_d):
    if b_or_d == 'batchsize':
        scaler_size = 256
        scaler = pickle.load(open(f"./model/scaler/{SCALER_MODE}/{instance_name[:2]}/scaler_{instance_name}_{model_name}_{scaler_size}dataset_BSpred.pickle", "rb"))
    else: # if b_or_d == 'dataset'
        scaler_size = 64
        scaler = pickle.load(open(f"./model/scaler/{SCALER_MODE}/{instance_name[:2]}/scaler_{instance_name}_{model_name}_{scaler_size}batchsize_DSpred.pickle", "rb"))
    
    scaled_pred = scaler.predict(np.array([size_pred]).reshape(-1, 1))
    latency_pred = (scaled_pred * (latency_max - latency_min) + latency_min)
    latency_pred = latency_pred[0][0]
    return latency_pred

# Function-2: scaler_validation
# b_or_d: what you wan to predict (batchsize or dataset)
# t_or_p: setting of min-max values (true or pred, pred is anchor prediction value)
# 1. setting variables for scaler model-validation
# 2. loop target instances, models, val_sizes, and exps, get values and inference for validation
#    (val_size is size of opposit condition of b_or_d, so if b_or_d is batchsize, val_size_list is dataset_list)
# 3. convert values into dataframe, and return it
def scaler_validation(anchor_pred, b_or_d, t_or_p):
    if b_or_d == 'batchsize':
        size_min = 16
        size_max = 256
        condition_size = 'dataset'
        val_size_list = dataset_list
    else: # if b_or_d == 'dataset'
        size_min = 32
        size_max = 256
        condition_size = 'batchsize'
        val_size_list = batchsize_list
    
    val_result = {}
    key_list = ['instance_name', 'model', 'exp_name', 'true',
                'scaler_pred', 'anchor_pred', 'size_min', 'size_max', 'size_pred',
                'latency_min', 'latency_max', 'b_or_d', 'b_or_d_size', 't_or_p']
    for key in key_list:
        val_result.setdefault(key, [])
    
    for val_instance in tqdm(PRED_INSTANCES):
        for val_model in model_list:
            for val_size in val_size_list:
                for val_exp in exp_list:
                    true_pred_df = anchor_pred[instance_index[val_instance]]
                    cond = ((true_pred_df['model'] == val_model) &
                            (true_pred_df[condition_size] == val_size) &
                            (true_pred_df['exp_name'] == val_exp))
                    val_df = true_pred_df[cond]
                    if len(val_df) != 5:
                        continue
                    
                    latency_min = val_df[val_df[b_or_d] == size_min][f'{val_instance}_{t_or_p}'].values[0]
                    latency_max = val_df[val_df[b_or_d] == size_max][f'{val_instance}_{t_or_p}'].values[0]
                    size_pred_list = [x for x in sorted(list(val_df[b_or_d].values)) if x not in [size_min, size_max]]
                    
                    for size_pred in size_pred_list:
                        latency_true = val_df[val_df[b_or_d] == size_pred][f'{val_instance}_true'].values[0]
                        latency_anchor_pred = val_df[val_df[b_or_d] == size_pred][f'{val_instance}_pred'].values[0]
                        latency_scaler_pred = inference_batch_dataset(
                            val_instance, val_model, latency_min, latency_max, size_pred, b_or_d)
                        val_result['instance_name'].append(val_instance)
                        val_result['model'].append(val_model)
                        val_result['exp_name'].append(val_exp)
                        val_result['true'].append(latency_true)
                        val_result['anchor_pred'].append(latency_anchor_pred)
                        val_result['scaler_pred'].append(latency_scaler_pred)
                        val_result['size_min'].append(size_min)
                        val_result['size_max'].append(size_max)
                        val_result['size_pred'].append(size_pred)
                        val_result['latency_min'].append(latency_min)
                        val_result['latency_max'].append(latency_max)
                        val_result['b_or_d'].append(b_or_d)
                        val_result['b_or_d_size'].append(val_size)
                        val_result['t_or_p'].append(t_or_p)
                        
    val_result_df = pd.DataFrame.from_dict(val_result, orient='columns')
    return val_result_df

# Function-3: print_error
# print MAPE, R^2, and RMSE
def print_error(true, pred):
    print(f'MAPE: {mean_absolute_percentage_error(true, pred) * 100}')
    print(f'R2: {r2_score(true, pred)}')
    print(f'RMSE: {math.sqrt(mean_squared_error(true, pred))} us')
    print(f'RMSE: {math.sqrt(mean_squared_error(true, pred))/1000} ms')
    print()

# 02-2-1. Model Validation of Scaler Prediction
# - setting 1: batchsize prediction with true min-max values
# - setting 2: batchsize prediction with anchor predicted min-max values
# - setting 3: dataset prediction with true min-max values
# - setting 4: dataset prediction with anchor predicted min-max values
result_batchsize_true = scaler_validation(anchor_pred, 'batchsize', 'true')
print_error(result_batchsize_true['true'], result_batchsize_true['scaler_pred'])
result_batchsize_pred = scaler_validation(anchor_pred, 'batchsize', 'pred')
print_error(result_batchsize_pred['true'], result_batchsize_pred['scaler_pred'])
result_dataset_true = scaler_validation(anchor_pred, 'dataset', 'true')
print_error(result_dataset_true['true'], result_dataset_true['scaler_pred'])
result_dataset_pred = scaler_validation(anchor_pred, 'dataset', 'pred')
print_error(result_dataset_pred['true'], result_dataset_pred['scaler_pred'])

pickle.dump([result_batchsize_true, result_batchsize_pred, result_dataset_true, result_dataset_pred],
            open(f'scaler_result_{ANCHOR_NAME}_{SCALER_MODE}.pickle', 'wb'))
