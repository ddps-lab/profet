#!bin/bash
INSTANCE_TYPE=$1
MODEL_NAME=$2
PIXEL_SIZE=$3
BATCH_SIZE=$4
USING_GPU_NUM=$5
PROF_MODE=$6

#Training CMD
TRAIN_CMD="/home/ubuntu/profet/data_generation/workload/profile_workload.py \
--model $MODEL_NAME --dataset $PIXEL_SIZE --batch_size $BATCH_SIZE --prof_or_latency $PROF_MODE \
--use_gpu_num $USING_GPU_NUM --instance_type $INSTANCE_TYPE"

sudo -i -u root bash << EOF
python3.7 $TRAIN_CMD
EOF
