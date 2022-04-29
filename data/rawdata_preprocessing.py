# profet: raw-data preprocessing

from collections import defaultdict
from statistics import median
from tqdm import tqdm
import pandas as pd
import os
import pickle

# list of experiments name and instances
EXP_NAMES = ['exp01', 'exp02', 'exp03', 'exp04', 'exp05', 'exp06', 'exp07', 'exp08', 'exp09']
INSTANCE_NAMES = ['g3s.xlarge', 'g4dn.xlarge', 'p2.xlarge', 'p3.2xlarge']

# get filenames from raw-data directories
# 'raw-data' directory contains csv and pickle files
# csv file is profiling feature data, and pickle file is batch and epoch latency data
# directory looks like this
# ./raw-data/exp01/g3s.xlarge-g3s.xlarge-128dataset-AlexNet-SGD-16-20210202-1426402021_02_02_14_29_22.csv
def get_filename(filetype):
    filename_list = []
    for exp in EXP_NAMES:
        for instance in INSTANCE_NAMES:
            exp_path = f"./raw_data/{exp}/{instance}"
            filenames = [f"./raw_data/{exp}/{instance}/{filename}" for filename in os.listdir(exp_path) if (filetype in filename)]
            filename_list.append(filenames)
    filename_list = [item for sublist in filename_list for item in sublist]
    return filename_list

# from list of feature filenames(.csv),
# 1. convert csv file to pandas dataframe
# 2. groupby operation name (HD_Type) and aggregation
# 3. return dataframes and all column names (union set of operations)
def get_feature_dflist(filename_list):
    df_list = []
    for filename in tqdm(filename_list):
        df = pd.read_csv(filename)
        df['HD_Type'] = df['Host/device'] + '_' + df['Type']
        df = df.groupby(['HD_Type']).sum()[['Total time (us)']]
        df = df.rename(columns={'Total time (us)': filename})
        df_list.append(df)
    
    all_columns = []
    for df in df_list:
        all_columns += list(df.index)
    all_columns = list(set(all_columns))
    
    return df_list, all_columns

# from list of dataframes, build single feature dataframe with union operation set
# 1. create dictionary variable 'col_dict' and initialize with empty list ([])
# 2. fill 'col_dict' with profile value or zero(0) or experiment informations
# 3. change dictionary to dataframe, then return dataframe
def build_feature_df(df_list, all_columns):
    col_dict = {}
    all_columns += ['tmp_index', 'exp_name', 'instance_name', 'dataset', 'model', 'optimizer', 'batchsize']
    
    for col in all_columns:
        col_dict.setdefault(col, [])
    
    for i in tqdm(range(0, len(df_list))):
        for col in all_columns:
            select_df_cols = list(df_list[i].index)
            if col in select_df_cols:
                col_dict[col].append(df_list[i].loc[col].values[0])
            elif col in ['tmp_index', 'exp_name', 'instance_name', 'dataset', 'model', 'optimizer', 'batchsize']:
                continue
            else:
                col_dict[col].append(0)
                
        col_dict['tmp_index'].append(df_list[i].columns[0].split('/')[-1])
        col_dict['exp_name'].append(df_list[i].columns[0].split('/')[2])
        col_dict['instance_name'].append(df_list[i].columns[0].split('/')[3])
        col_dict['dataset'].append(df_list[i].columns[0].split('/')[-1].split('-')[2])
        col_dict['model'].append(df_list[i].columns[0].split('/')[-1].split('-')[3])
        col_dict['optimizer'].append(df_list[i].columns[0].split('/')[-1].split('-')[4])
        col_dict['batchsize'].append(int(df_list[i].columns[0].split('/')[-1].split('-')[5]))
    
    feature_df = pd.DataFrame.from_dict(col_dict).set_index('tmp_index')
    return feature_df

# from list of pickle filenames, build single target dataframe
# 1. initialize lists for dataframe columns
# 2. append experiments information and target value to lists
# 3. convert lists to single dictionary, then build dataframe
def build_target_df(filename_list):
    exp_list = []
    instance_list = []
    dataset_list = []
    model_list = []
    optimizer_list = []
    batchsize_list = []
    epoch_latency_list = []
    batch_latency_list = []
    
    for filename in tqdm(filename_list):
        exp_string = filename.split('/')[4]
        raw_data = pickle.load(open(filename, 'rb'))

        exp_list.append(filename.split('/')[2])
        instance_list.append(filename.split('/')[3])
        dataset_list.append(exp_string.split('-')[2])
        model_list.append(exp_string.split('-')[3])
        optimizer_list.append(exp_string.split('-')[4])
        batchsize_list.append(int(exp_string.split('-')[5]))
        
        epoch_latency = raw_data[2]
        batch_latency = median(raw_data[3])
        epoch_latency_list.append(epoch_latency)
        batch_latency_list.append(batch_latency)
        
    target_dict = {'exp_name' : exp_list,
                    'instance_name' : instance_list,
                    'dataset' : dataset_list,
                    'model' : model_list,
                    'optimizer' : optimizer_list,
                    'batchsize' : batchsize_list,
                    'epoch_latency' : epoch_latency_list,
                    'batch_latency' : batch_latency_list}
    
    target_df = pd.DataFrame(target_dict)
    return target_df

# 01-1. Build Feature DataFrame
# 1. get only 'csv' file which contains profiling data
# 2. build feature dataframe
# 3. save feature dataframe to preprocessed dataset directory
print("Build feature dataframe")
feature_filename_list = get_filename('.csv')
df_list, all_columns = get_feature_dflist(feature_filename_list)
feature_df = build_feature_df(df_list, all_columns)
pickle.dump(feature_df, open('./pre_data/feature_preprocessing.pickle', 'wb'))

# 01-2. Build Target DataFrame
# 1. get only 'pickle' file which contains every single batch latencies and epoch latencies
# 2. build target dataframe
# 3. save target dataframe to preprocessed dataset directory
print("Build target dataframe")
target_filename_list = get_filename('.pickle')
target_df = build_target_df(target_filename_list)
pickle.dump(target_df, open('./pre_data/target_preprocessing.pickle', 'wb'))

# 01-3. Merge Feature DataFrame and Target DataFrame with Inner Join
# use merge method for join opearation between feature dataframe and target dataframe
print("Build feature-target dataframe")
merge_df = feature_df.merge(target_df, how='inner', left_on=['exp_name', 'instance_name', 'dataset', 'model', 'batchsize', 'optimizer'], right_on=['exp_name', 'instance_name', 'dataset', 'model', 'batchsize', 'optimizer'])
pickle.dump(merge_df, open('./pre_data/merge_preprocessing.pickle', 'wb'))

# 01-4. Extract Workloads that Intersection of g3s, g4dn, p2, p3
instance_list = list(target_df['instance_name'].value_counts().index)
model_list = list(target_df['model'].value_counts().index)
dataset_list = list(target_df['dataset'].value_counts().index)
batchsize_list = list(target_df['batchsize'].value_counts().index)
g3_workloads = target_df[(target_df['instance_name'] == 'g3s.xlarge') &
                         (target_df['exp_name'] == 'exp01')][['dataset', 'model', 'batchsize']].values

print("Extract intersection workloads(minimum workloads on g3s.xlarge)")
feature_list = []
target_list = []
for instance_name in tqdm(instance_list):
    for workload in g3_workloads:
        feature_cond = ((feature_df['instance_name'] == instance_name) &
                        (feature_df['dataset'] == workload[0]) &
                        (feature_df['model'] == workload[1]) &
                        (feature_df['batchsize'] == workload[2]))
        feature_list.append(feature_df[feature_cond])
        
        target_cond = ((target_df['instance_name'] == instance_name) &
                       (target_df['dataset'] == workload[0]) &
                       (target_df['model'] == workload[1]) &
                       (target_df['batchsize'] == workload[2]))
        target_list.append(target_df[target_cond])

feature_df_g3 = pd.concat(feature_list)
target_df_g3 = pd.concat(target_list)
merge_df_g3 = feature_df_g3.merge(target_df_g3, how='inner', left_on=['exp_name', 'instance_name', 'dataset', 'model', 'batchsize', 'optimizer'], right_on=['exp_name', 'instance_name', 'dataset', 'model', 'batchsize', 'optimizer'])
pickle.dump(merge_df_g3, open('./pre_data/merge_preprocessing_g3.pickle', 'wb'))
print("Raw-data preprocessing done.")
