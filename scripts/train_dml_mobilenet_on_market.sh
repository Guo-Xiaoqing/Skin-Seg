#!/bin/bash
#
# This script performs the following operations:
# Training 2 MobileNets with DML on Market-1501
#
# Usage:
# cd Deep-Mutual-Learning
# ./scripts/train_dml_mobilenet_on_market.sh


# Where the TFRecords are saved to.
DATASET_DIR=./datasets/tfrecord/

# Where the checkpoint and logs will be saved to.
DATASET_NAME=train
SAVE_NAME=./logs/skin_unet_semi #WCE_densenet, WCE_bi_densenet, WCE_saliency, WCE_orig_densenet, WCE_orig_bi_densenet, WCE_orig_saliency          orig refers to no crop black region, without means without blak region(floder name)
SAL_DIR=${SAVE_NAME}/saliency_map
CKPT_DIR=${SAVE_NAME}/checkpoint
LOG_DIR=${SAVE_NAME}/logs

# Model setting
MODEL_NAME=unet,unet
SPLIT_NAME=train

# Run training.
python3 train_image_classifier.py \
    --dataset_name=${DATASET_NAME}\
    --split_name=${SPLIT_NAME} \
    --dataset_dir=${DATASET_DIR} \
    --saliency_map=${SAL_DIR} \
    --checkpoint_dir=${CKPT_DIR} \
    --log_dir=${LOG_DIR} \
    --model_name=${MODEL_NAME} \
    --preprocessing_name=reid \
    --num_networks=2
