# profet: anchor-data preprocessing

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

# Build anchor dataframe
# 1. call function with 'anchor instance type', preprocessed data, 
#    and type of target value (batch latency or epoch latency)
# 2. select feature of anchor instance (anchor_df)
# 3. get list of target instances (pred_instances)
# 4. loop each anchor feature rows, select same workload from target instances by condition
# 5. finally, there are anchor feature dataframe and target value dataframe 
#    (with multiple target instances) and inner join them with workload informations
def build_anchor_df(anchor_instance, merge_data, batch_or_epoch):
    s_to_us = 1000000
    anchor_df = merge_data[merge_data['instance_name'] == anchor_instance]
    
    instance_list = list(merge_data['instance_name'].value_counts().index)
    pred_instances = [x for x in instance_list if x != anchor_instance]
    
    pred_dict_list = []
    for idx, row in tqdm(anchor_df.iterrows()):
        pred_dict = {}
        for column in ['exp_name', 'dataset', 'model', 'batchsize', 'optimizer']:
            pred_dict[column] = row[column]
        for instance in pred_instances:
            pred_cond = (merge_data['instance_name'] == instance)
            for column in ['exp_name', 'dataset', 'model', 'batchsize', 'optimizer']:
                pred_cond = pred_cond & (merge_data[column] == row[column])
            
            pred_latency = merge_data[pred_cond][batch_or_epoch]
            if len(pred_latency) == 1:
                pred_dict[instance] = list(pred_latency)[0] * s_to_us
            else:
                print("error")
        pred_dict_list.append(pred_dict)
    pred_df = pd.DataFrame(pred_dict_list)
    
    anchor_df = anchor_df.merge(
        pred_df, how='inner',
        left_on=['dataset', 'model', 'optimizer', 'batchsize', 'exp_name'],
        right_on=['dataset', 'model', 'optimizer', 'batchsize', 'exp_name']
    )
    return anchor_df

# 02-1. Build Anchor DataFrame
merge_data = pickle.load(open('./pre_data/merge_preprocessing_g3.pickle', 'rb'))
print("Build Anchor Dataset: g3s.xlarge")
anchor_g3 = build_anchor_df('g3s.xlarge', merge_data, 'batch_latency')
print("Build Anchor Dataset: g4dn.xlarge")
anchor_g4 = build_anchor_df('g4dn.xlarge', merge_data, 'batch_latency')
print("Build Anchor Dataset: p2.xlarge")
anchor_p2 = build_anchor_df('p2.xlarge', merge_data, 'batch_latency')
print("Build Anchor Dataset: p3.2xlarge")
anchor_p3 = build_anchor_df('p3.2xlarge', merge_data, 'batch_latency')

# 02-2. Save Anchor DataFrame
pickle.dump(anchor_g3, open(f'./anchor_data/anchor_g3.pickle', 'wb'))
pickle.dump(anchor_g4, open(f'./anchor_data/anchor_g4.pickle', 'wb'))
pickle.dump(anchor_p2, open(f'./anchor_data/anchor_p2.pickle', 'wb'))
pickle.dump(anchor_p3, open(f'./anchor_data/anchor_p3.pickle', 'wb'))
print("anchor-data preprocessing done.")
