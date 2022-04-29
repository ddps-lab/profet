# profet: anchor prediction

import tensorflow as tf
import pandas as pd 
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str)
parser.add_argument('--anchor_instance', type=str)
parser.add_argument('--anchor_latency', type=float)
args = parser.parse_args()

FILENAME = args.filename
ANCHOR_INSTANCE = args.anchor_instance
ANCHOR_LATENCY = args.anchor_latency

INSTANCE_LIST = ['g3s.xlarge', 'g4dn.xlarge', 'p2.xlarge', 'p3.2xlarge']
PRED_INSTANCES = [x for x in INSTANCE_LIST if x != ANCHOR_INSTANCE]
FEATURE_COLUMNS = ['Device_SoftmaxCrossEntropyWithLogits', 'Host_IDLE',
                  'Device_Conv2DBackpropFilter', 'Device_RealDiv', 'Host__Send',
                  'Host_FlushSummaryWriter', 'Device_MaxPool',
                  'Device_FusedBatchNormGradV3', 'Device_AvgPoolGrad', 'Device_AddN',
                  'Device_Slice', 'Device_ReluGrad', 'Host_IteratorGetNext',
                  'Host_Identity', 'Device_AvgPool', 'Device_Relu', 'Host_GatherV2',
                  'Device_Conv2D', 'Host__HostSend', 'Device__Recv', 'Host_Dataset',
                  'Device_Transpose', 'Device_Mean', 'Device_ConcatV2',
                  'Device_MaxPoolGrad', 'Device_FusedBatchNormV3', 'Device_IDLE',
                  'Device_Conv2DBackpropInput', 'Host_LogicalAnd', 'Host_WriteSummary',
                  'Device_DepthwiseConv2dNative', 'Device_ResourceApplyGradientDescent',
                  'Device_AssignSubVariableOp', 'Device_Relu6Grad', 'Device_BiasAddGrad',
                  'Device_AddV2', 'Device_MatMul', 'Device_RsqrtGrad', 'Device_BiasAdd',
                  'Device_Pad', 'Device_Equal', 'Device_Sum', 'Device_Neg',
                  'Device_RandomUniform', 'Device_Sub',
                  'Device_DepthwiseConv2dNativeBackpropInput',
                  'Device_AssignAddVariableOp', 'Device_BroadcastTo',
                  'Device_GreaterEqual', 'Device_LogicalAnd', 'Device_Cast',
                  'Device_Softmax', 'Device_Relu6', 'Device_Mul', 'Device__HostRecv',
                  'Device_DynamicStitch', 'Device_DepthwiseConv2dNativeBackpropFilter',
                  'Device__FusedConv2D', 'Device_ArgMax', 'Device_DivNoNan',
                  'Device_Rsqrt', 'Device__Send', 'Device_SquaredDifference', 'Device_Tile', 'Device_Square']
CLUSTER_FEATURES={0: ['DepthwiseConv2dNativeBackpropInput', 'DepthwiseConv2dNativeBackpropFilter'], 
                    1: ['DepthwiseConv2dNative'], 
                    2: ['Conv2DBackpropInput', 'Conv2DBackpropFilter'], 
                    3: ['AssignSubVariableOp', 'AssignAddVariableOp'], 
                    4: ['FusedBatchNormV3', 'FusedBatchNormGradV3'],
                    5: ['BroadcastTo'], 
                    6: ['LogicalAnd'], 
                    7: ['BiasAddGrad', 'BiasAdd'], 
                    8: ['MaxPoolGrad', 'AvgPoolGrad'], 
                    9: ['Relu6Grad', 'RsqrtGrad', 'ReluGrad'], 
                    10: ['Conv2D', 'ConcatV2'], 
                    11: ['MatMul', 'MaxPool', 'AvgPool'], 
                    12: ['Softmax', 'ArgMax'], 
                    13: ['Relu', 'AddV2', 'IDLE', 'Pad', 'Mean', 'Equal', 'Sum', 'Neg', 'Sub', 'Slice', 'RealDiv', 'Cast', 'Relu6', 'Mul', '_Recv', 'Rsqrt', 'AddN', '_Send', 'Tile', 'Square'], 
                    14: ['DivNoNan'], 
                    15: ['_HostRecv'], 
                    16: ['Transpose'], 
                    17: ['GreaterEqual'], 
                    18: ['_FusedConv2D'], 
                    19: ['RandomUniform'], 
                    20: ['DynamicStitch'], 
                    21: ['SquaredDifference'], 
                    22: ['ResourceApplyGradientDescent'], 
                    23: ['SoftmaxCrossEntropyWithLogits']}

def median_ensemble(test_x, anchor_latency, anchor_instance, pred_instance):
    anchor_name = anchor_instance[:2]
    pred_name = pred_instance[:2]
    
    model_linear = pickle.load(open(f'./model/anchor/anchor_{anchor_name}_{pred_name}_linear.model', 'rb'))
    model_rfr = pickle.load(open(f'./model/anchor/anchor_{anchor_name}_{pred_name}_rfr.model', 'rb'))
    model_dnn = tf.keras.models.load_model(f'./model/anchor/anchor_{anchor_name}_{pred_name}_dnn.model')
    
    linear_pred = model_linear.predict(np.array(anchor_latency).reshape(-1, 1)).reshape(-1, 1)
    rfr_pred = model_rfr.predict(test_x).reshape(-1, 1)
    dnn_pred = model_dnn.predict(test_x).reshape(-1, 1)
    median_pred = np.median(np.stack([dnn_pred, rfr_pred, linear_pred]), axis=0)
    print(f'{anchor_instance} - {pred_instance} : {median_pred}')
    return median_pred

def anchor_inference(filename, anchor_latency, anchor_instance, pred_instance):
    test_x = pd.read_json(filename)
    missing_columns_list = [x for x in FEATURE_COLUMNS if x not in test_x.columns]
    for i in missing_columns_list:
        test_x[i] = 0
    
    for key, value in CLUSTER_FEATURES.items():
        value = ["Device_" + x for x in value]
        test_x["&".join(value)] = 0
        for feature in value:
            test_x["&".join(value)] += test_x[feature]
        test_x.drop(value, axis=1, inplace=True)
        
    median_ensemble(test_x, anchor_latency, anchor_instance, pred_instance)

for pred_instance in PRED_INSTANCES:
    anchor_inference(FILENAME, ANCHOR_LATENCY, ANCHOR_INSTANCE, pred_instance)
