#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Esther Rituerto-González
Contact: erituert [at] ing [dot] uc3m [dot] es 
         esther [dot] rituerto [dot] g [at] gmail [dot] com
Last updated: February 2023


This script aims to pre-process the speech signals of the WEMAC
database in order to obtain normalised 
audios per user @16kHz mono

"""

# Define paths
import os
input_path =  '...'
audios_path = input_path+'Audios/'
labels_path = input_path + 'Labels/'
output_path = '...'
signals_output_path = output_path+'signals/'

if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(output_path+'signals/'):
    os.makedirs(output_path+'signals/')

# Libraries
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import glob
from scipy.io import wavfile
import librosa
from audio_tools import natural_sort, butter_lowpass_filter, butter_highpass_filter
 

# Variables
vvg=0
new_rate = 16000
win_sec = 20e-3
step_sec = 10e-3
win = int(new_rate*win_sec)
step = int(new_rate*step_sec)

percent_high_nrg = 0.1 # Porcentaje de voz que está dispuesto a asumir para marcar silencio en una trama
frame_len_s = 0.1 # Tamaño de ventana que marca cmo voz o silencio

order_lpf = 16
order_hpf = 6
fs = 48000       # sample rate, Hz
cutoff_lpf = 8000  # desired cutoff frequency of the low pass filter, Hz
cutoff_hpf = 50  # desired cutoff frequency of the high pass filter, Hz


# =============================================================================
# Main
# =============================================================================


if (vvg == 0):
    
    # Load users
    valid_users = pd.read_excel(labels_path+'....xlsx')

    valid_users = valid_users.drop(valid_users[valid_users['Nivel1 (0-Fallo/1-OK)'] != 1].index)
    user_ids = valid_users['Voluntaria ID'].reset_index(drop=True)
    
    # Skip incomplete users
    user_ids = user_ids.drop(user_ids[user_ids=='V025'].index)
    user_ids = user_ids.drop(user_ids[user_ids=='V098'].index)
    user_ids = user_ids.drop(user_ids[user_ids=='V124'].index)
    user_ids = user_ids.drop(user_ids[user_ids=='V129'].index)
    
    user_ids = user_ids.reset_index(drop=True)
    
    user_audio_folders = input_path+'Audios/'+user_ids.astype(str)
    
    # Load labels
    labels = pd.read_csv(labels_path+'....csv',sep=";") 
    
elif (vvg == 1):
    
    # Load users
    valid_users = pd.read_csv(labels_path+'/Vol_nombreG.csv', sep=";")
    
    # Skip incomplete users
    valid_users = valid_users.drop(valid_users[valid_users['Voluntaria']!='G30'].index)
    
    valid_users_wona=valid_users.dropna()
    valid_users_wona = valid_users_wona['Voluntaria'].tolist()
    user_ids = np.unique(valid_users_wona).tolist()
    user_audio_folders= [input_path+'Audios/'+user_id for user_id in user_ids]
    
    # Load labels
    labels = valid_users
    
## Load audios

# For each user
for idx, user_name in enumerate(user_audio_folders): 
    
    # Do not load demo audio
    user_audios_names = natural_sort(glob.glob(user_name+"/"+user_ids[idx]+"_VIDEO*.wav"))
    videos_num = [i.split('/')[-1].split('_')[1].split(' ')[1] for i in user_audios_names]
    
    last_vid = int(videos_num[-1])
    
    # Load all audios together and find the maximum value for normalizing later        
    wi = 0
    ji = 1
    
    for w in range(wi,last_vid):
        if (w == 0):
            samplerate, data_full = wavfile.read(user_audios_names[w])
        else:
            _ , data_aux = wavfile.read(user_audios_names[w])
            data_full = np.hstack((data_full, data_aux))
    
    max_value = np.max(np.abs(data_full)) 
    del data_full    
        
    #fixing users g30 and v098   
    if ((vvg == 1) and (user_name[-3:] == "G30")):
        range_ = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]
    elif ((vvg == 0) and (user_name[-4:] == "V098")):
        range_ = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
    else:
        range_ = range(ji,last_vid+1)
        
    ## For each audio
    for j in range_: # j from 1 to 14
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
        if (vvg==0):
            binary_label = labels['Emocion.Reportada'].iloc[(idx*14)+j-1]
        elif (vvg==1):
            binary_label = int(labels['Reportada.Binarizado'].iloc[(idx*14)+j-1])

        output_filename = user_ids[idx]+'_VIDEO_'+str(j)+'_BINLABEL_'+str(binary_label)
    
        # Saving audiofile     
        y = (y*32767).astype(np.int16)
        wavfile.write(signals_output_path+'/'+output_filename+'.wav', new_rate, y)
        print('Preprocessed volunt '+user_ids[idx]+' VIDEO '+str(j))