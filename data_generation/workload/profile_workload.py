import math
import time
import pickle
import argparse
from datetime import datetime

import tensorflow as tf
import numpy as np

import dataset_info
import model_info

# Check num of gpus
gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
for gpu in gpus:
    print('Name:', gpu.name, ' Type:', gpu.device_type)

# Get arguments for job
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='VGG19', type=str)
parser.add_argument('--dataset', default=224, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--optimizer', default='SGD', type=str)
parser.add_argument('--prof_or_latency', default='profiling', type=str)
parser.add_argument('--use_gpu_num', default=1, type=int)
parser.add_argument('--prof_point', default=1.5, type=float)
parser.add_argument('--prof_len', default=1, type=int)
parser.add_argument('--instance_type', default='EC2', type=str)
args = parser.parse_args()

# Dataset Info
dataset = dataset_info.select_dataset(args.dataset)
model_name = args.model

num_classes = dataset['num_classes']
img_rows = dataset['img_rows']
img_cols = dataset['img_cols']
img_channels = dataset['img_channels']
num_data = dataset['num_data']
num_test = dataset['num_test']

use_gpu_num = args.use_gpu_num
if use_gpu_num > num_gpus:
    raise Exception(f'Detected gpu num is {num_gpus}, but using {use_gpu_num} gpus')
batch_size = args.batch_size * use_gpu_num
prof_point = args.prof_point
batch_num = math.ceil(num_data/batch_size)
epochs = math.ceil(prof_point)
prof_start = math.floor(batch_num * prof_point)
prof_len = args.prof_len
prof_range = '{}, {}'.format(prof_start, prof_start + prof_len)
optimizer = 'SGD'

###################### Build Synthetic Dataset ######################
x_train_shape = (num_data, img_rows, img_cols, img_channels)
y_train_shape = (num_data, 1)

x_test_shape = (num_test, img_rows, img_cols, img_channels)
y_test_shape = (num_test, 1)

x_train = np.random.rand(*x_train_shape)
y_train = np.random.randint(num_classes, size=y_train_shape)
x_test = np.random.rand(*x_test_shape)
y_test = np.random.randint(num_classes, size=y_test_shape)
###############################################################

if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    input_shape = (img_rows, img_cols, img_channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Select model from model info module
device_names = [f'/device:GPU:{x}' for x in range(num_gpus)]
use_device_names = device_names[:use_gpu_num]
strategy = tf.distribute.MirroredStrategy(devices=use_device_names)
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


if use_gpu_num == 1:
    model = model_info.select_model(model_name, input_shape, num_classes)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
    optimizer=optimizer,
    metrics=['accuracy'])
else:
    with strategy.scope():
        model = model_info.select_model(model_name, input_shape, num_classes)
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
        optimizer=optimizer,
        metrics=['accuracy'])
    
time_now = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
job_name = f'{args.instance_type}-{args.dataset}-{model_name}-{optimizer}-{batch_size}-{use_gpu_num}-{time_now}'
logs = f'/home/ubuntu/profet/data_generation/logs/{job_name}'
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = prof_range)

# Setting for latency check callback
class BatchTimeCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.all_times = []

    def on_train_end(self, logs=None):
        time_filename = f'/home/ubuntu/profet/data_generation/tensorstats/times-{job_name}.pickle'
        time_file = open(time_filename, 'ab')
        pickle.dump(self.all_times, time_file)
        time_file.close()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_times = []
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_time_end = time.time()
        self.all_times.append(self.epoch_time_end - self.epoch_time_start)
        self.all_times.append(self.epoch_times)

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_time_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.epoch_times.append(time.time() - self.batch_time_start)

latency_callback = BatchTimeCallback()

prof_model_select = {'profiling': [tboard_callback], 'latency': [latency_callback], 'vanilla': []}
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test), callbacks=prof_model_select[args.prof_or_latency])
