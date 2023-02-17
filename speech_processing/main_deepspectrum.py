#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Esther Rituerto-González
Contact: erituert [at] ing [dot] uc3m [dot] es 
         esther [dot] rituerto [dot] g [at] gmail [dot] com
Last updated: February 2023

Ref: https://github.com/lstappen/MuSe-data-base/blob/master/feature_extraction/audio/baseline/extractSMILE-XBOW-DS.py

"""

# =============================================================================
# Extracting Deepspectrum Embeddings
# =============================================================================
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, help="directory of audios", required=True)
parser.add_argument("--target_file", type=str, help="Path for the deepspectrum csv", default="./deepspectrum.csv")
parser.add_argument("--window_size", type=int, help="window size for deepspectrum", default=1)
parser.add_argument("--hop_size", type=float, help="hop size for deepspectrum", default=1)
parser.add_argument("--net", type=str, help="network to extract embeddings with", default='vgg19') # vgg19 and resnet50
parser.add_argument("--fl", type=str, help="final layer to extract embeddings with", default='fc2') # fc2 and avg_pool, respectively

# Parameters
args = parser.parse_args()
SOURCE_DIR = args.source_dir
TARGET_FILE = args.target_file
WS = args.window_size
HS = args.hop_size
NET = args.net
FL = args.fl

# Avoid cuDNN problems
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# =============================================================================
# Main DeepSpectrum
# =============================================================================
os.system("CUDA_VISIBLE_DEVICES= deepspectrum -v features " + SOURCE_DIR + " -t " + str(WS) +" "+ str(HS)+ " -en " + NET + " -fl " + FL + " -o " + TARGET_FILE)

# =============================================================================
# Processing output file and saving one pandas df per video
# =============================================================================
import pandas as pd
import numpy as np

input_path = TARGET_FILE.split('/')[0:-2]
input_path = '/'.join(input_path)
output_path = input_path+'/deepspectrum_'+NET+'_per_user/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
ds_df = pd.read_csv(TARGET_FILE)

filenames = np.unique(ds_df.name)

for filename in filenames:
    user_df = ds_df[(ds_df.name == filename)]
    output_filename = filename.split('_BINLABEL_')[0]
    user_df = user_df.rename(columns={"timeStamp":"timestamp"})
    user_df['timestamp'] = user_df['timestamp']+1
    user_df = user_df.drop(columns = ['name', 'class'])
    user_df.to_csv(output_path+'/'+output_filename+'.csv', index=False)
