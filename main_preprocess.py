#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Esther Rituerto-González
Contact: erituert@ing.uc3m.es
Release date: February 2022


This script aims to pre-process the speech signals of the first 47 volunteer 
from the dataset in order to obtain normalised 
audios per user, clean, @16kHz mono

"""

# Define paths
input_path = '...'
audios_path = input_path+'Audios/'
labels_path = input_path+'Labels/'
output_path = '...'
# os.mkdir(output_path)
signals_output_path = output_path+'signals/'

# Libraries
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import glob
from scipy.io import wavfile
import librosa
from audio_tools import natural_sort, butter_lowpass_filter, butter_highpass_filter
 

# Variables
new_rate = 16000
win_sec = 20e-3
step_sec = 10e-3
win = int(new_rate*win_sec)
step = int(new_rate*step_sec)

order_lpf = 16
order_hpf = 6
fs = 48000       # sample rate, Hz
cutoff_lpf = 8000  # desired cutoff frequency of the low pass filter, Hz
cutoff_hpf = 50  # desired cutoff frequency of the high pass filter, Hz


# =============================================================================
# Main
# =============================================================================

# Load usernames
user_audio_folders = natural_sort(glob.glob(audios_path+"*"))
user_ids = [i.split('/')[-1] for i in user_audio_folders]

# Load labels
labels = pd.read_csv(labels_path+'....csv')

## Load audios

# For each user
for idx, user_name in enumerate(user_audio_folders): #idx va de 0 al num de voluntarias
    
    # Do not load demo audio
    user_audios_names = natural_sort(glob.glob(user_name+"/"+user_ids[idx]+"_VIDEO*.wav"))
    videos_num = [i.split('/')[-1].split('_')[1].split(' ')[1] for i in user_audios_names]
    
    last_vid = int(videos_num[-1])
    
    # Load all audios together and find the maximum value for normalizing later
    if idx == 10: # User num 25 que no tiene audio 1
        wi = 1
        ji = 2
    else:
        wi = 0
        ji = 1
        
    for w in range(wi,last_vid):
        if (((w == 0)  and (idx!= 10)) or ((w==1) and (idx==10))):
            samplerate, data_full = wavfile.read(user_audios_names[w])
        else:
            if idx == 10:
                _ , data_aux = wavfile.read(user_audios_names[w-1])
                data_full = np.hstack((data_full, data_aux))
            else:
                _ , data_aux = wavfile.read(user_audios_names[w])
                data_full = np.hstack((data_full, data_aux))
    
    max_value = np.max(np.abs(data_full)) 
    del data_full    
            
    ## For each audio
    for j in range(ji,last_vid+1): # j va de 1 a 14
        pos = [k for k, e in enumerate(videos_num) if e == str(j)] 
        
        # If there is more than one audiodescription for this video
        if len(pos) > 1: 
            o = 0
            for l in pos:
                if o == 0:
                    samplerate, data = wavfile.read(user_audios_names[l])
                else:
                    _ , data_aux = wavfile.read(user_audios_names[l])
                    data = np.hstack((data, data_aux))
        
        # If there is only one audiodescription for this video
        else:
            samplerate, data = wavfile.read(user_audios_names[pos[0]])
        
        #print(len(data))
            
        ## Start pre-processing
          
        # 1. Filtering
        # Low pass at 8kHz, order 16
        y_lpf = butter_lowpass_filter(data, cutoff_lpf, fs, order_lpf) 
        # High pass at 50Hz, order 6
        y_hpf = butter_highpass_filter(y_lpf.astype(np.int16), cutoff_hpf, fs, order_hpf) 
        data_filtered = y_hpf
            
        # 2. Normalizing , do we want it per user?)
        #data_normalized = data/np.max(np.abs(data_filtered)) # at audio level
        data_normalized = data_filtered/max_value # at user/sesion level

        # 3. Downsampling to 16kHz
        data_resampled = librosa.resample(data_normalized, samplerate, new_rate)
        y = data_resampled
        
        # 4. Padding signals with 0's until full seconds
        padding = np.zeros((new_rate-len(data_resampled)%new_rate,))
        y = np.hstack((data_resampled, padding))
          
        # Output filename
        binary_label = labels['EmocionReportada'].iloc[(idx*14)+j-1]
        output_filename = user_ids[idx]+'_VIDEO_'+str(j)+'_BINLABEL_'+str(binary_label)
    
        # Saving audiofile     
        y = (y*32767).astype(np.int16)
        wavfile.write(signals_output_path+'/'+output_filename+'.wav', new_rate, y)
        