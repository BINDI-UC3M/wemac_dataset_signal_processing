#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Esther Rituerto-González
Contact: erituert@ing.uc3m.es
Release date: December 2021

"""

# Define paths
preprocessed_audios_path = '...'
output_path = '...'
working_path = '...'


# Import libraries
import os
os.chdir(working_path)
import numpy as np
from scipy.io import wavfile
# import matplotlib.pyplot as plt
import pandas as pd
from audio_tools import natural_sort
from unsupervised_vad import process_vad
from librosa_feature_extraction import librosa_feature_extraction
from opensmile_feat_extraction import opensmile_feat_extraction


# Define variables
fs = 16000
win_sec = 20e-3
step_sec = 10e-3
win = int(fs*win_sec)
step = int(fs*step_sec)
n_mfcc = 13
NFFT = 320
percent_high_nrg = 0.1 # VAD energy theshold to consider a segment like "voiced" 
frame_len_s = 1 # Segment window size for VAD 

# =============================================================================
# MAIN
# =============================================================================
filenames = natural_sort(os.listdir(preprocessed_audios_path))

for i, filename in enumerate(filenames):
        
    # Load audio
    sampling_rate, y = wavfile.read(preprocessed_audios_path+'/'+filename) # scipy reads int
    if y.dtype == 'int16':
        y = y/32767 # Preprocessed .wav were "int16" but in 'inf' it is float64 so no need
    
    len_y_seconds = int(np.floor(len(y)/fs))
    timestamps = np.arange(1, len_y_seconds+1, 1)
    
    # Output filename
    output_filename = filename.split('BINLABEL')[0][0:-1]

    # Extract VAD vector
    vad_vector = process_vad(y, sampling_rate, win, step, \
                              frame_len_s, percent_high_nrg)
    vad_df = pd.DataFrame({'timestamp': timestamps, 'VAD': vad_vector})
    vad_df.to_csv(output_path+'/vad/'+output_filename+'.csv', index=False)

    ## Extract librosa (38 features)
    librosa_df = librosa_feature_extraction(y, fs, win, step, n_mfcc, NFFT)
    librosa_df.insert(0, 'timestamp', timestamps)
    librosa_df.to_csv(output_path+'/features/librosa/'+output_filename+'.csv', index=False)

    ## Extract openSMILE features (egemaps 88 and compare 6k)
    compare_df, egemaps_df  = opensmile_feat_extraction(y, sampling_rate)
    compare_df.insert(0, 'timestamp', timestamps)
    egemaps_df.insert(0, 'timestamp', timestamps)
    compare_df.to_csv(output_path+'/features/compare/'+output_filename+'.csv', index=False)
    egemaps_df.to_csv(output_path+'/features/egemaps/'+output_filename+'.csv', index=False)
        

