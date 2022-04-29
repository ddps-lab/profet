# profet: scaler prediction

import pandas as pd
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target_instance', type=str)
parser.add_argument('--latency_min', type=float)
parser.add_argument('--latency_max', type=float)
parser.add_argument('--size_pred', type=int)
parser.add_argument('--batch_or_dataset', type=str)
args = parser.parse_args()

TARGET_INSTANCE = args.target_instance
LATENCY_MIN = args.latency_min
LATENCY_MAX = args.latency_max
SIZE_PRED = args.size_pred
BATCH_OR_DATASET = args.batch_or_dataset
INSTANCE_LIST = ['g3s.xlarge', 'g4dn.xlarge', 'p2.xlarge', 'p3.2xlarge']

def inference_batch_dataset(instance_name, latency_min, latency_max, size_pred, b_or_d):
    if b_or_d == 'batchsize':
        scaler_size = 256
        scaler = pickle.load(open(f"./model/scaler/scaler_{instance_name}_{scaler_size}dataset_BSpred.model", "rb"))
    else: # if b_or_d == 'dataset'
        scaler_size = 64
        scaler = pickle.load(open(f"./model/scaler/scaler_{instance_name}_{scaler_size}batchsize_DSpred.model", "rb"))
    
    scaled_pred = scaler.predict(np.array([size_pred]).reshape(-1, 1))
    latency_pred = (scaled_pred * (latency_max - latency_min) + latency_min)
    latency_pred = latency_pred[0][0]
    return latency_pred

latency_pred = inference_batch_dataset(TARGET_INSTANCE, LATENCY_MIN, LATENCY_MAX, SIZE_PRED, BATCH_OR_DATASET)
print(f'Predicted latency: {latency_pred}')
