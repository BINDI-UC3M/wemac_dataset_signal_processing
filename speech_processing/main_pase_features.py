#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Esther Rituerto-González
Contact: erituert [at] ing [dot] uc3m [dot] es 
         esther [dot] rituerto [dot] g [at] gmail [dot] com
Last updated: February 2023

"""

# Define paths
preprocessed_audios_path = '...'
output_path = '...'

# Import libraries
import sys
import os
sys.path.append(os.getcwd())


import numpy as np
from scipy.io import wavfile
import pandas as pd
from audio_tools import natural_sort
from audio_tools import reshape1sec

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Loading PASE Model
from pase.models.frontend import wf_builder
pase = wf_builder('cfg/frontend/PASE+.cfg').eval()
pase.load_pretrained('FE_e199.ckpt', load_last=True, verbose=True)

import torch
filenames = natural_sort(os.listdir(preprocessed_audios_path))

for i, filename in enumerate(filenames):
        
    # Load audio
    fs, y = wavfile.read(preprocessed_audios_path+'/'+filename) # scipy reads int
    if y.dtype == 'int16':
        y = y/32767 # Preprocessed .wav were "int16" but in 'inf' it is float64 so no need
    
    len_y_seconds = int(np.floor(len(y)/fs))
    timestamps = np.arange(1, len_y_seconds+1, 1)
    #print('Length of Timestamps ='+str(len_y_seconds+1))

    # Output filename
    output_filename = filename.split('BINLABEL')[0][0:-1]

    # convert array to tensor
    y=torch.reshape(torch.from_numpy(y), (1, 1, len(y)))

    # Locate file and model in GPU
    y = y.type(torch.FloatTensor)
    y = y.cuda()
    pase = pase.cuda()

    # Use PASE Encoder to extract embedding
    pase_embedding = pase(y) #pase_embedding size will be (1, 256, len_y_seconds*fs/160)
    pase_embedding_n = pase_embedding.detach().cpu().numpy()

    # Reshaping o average per second
    pase_embedding_r = np.reshape(pase_embedding_n[0,:,:], (256, int(np.round(pase_embedding_n.shape[2]/100)),100))
    pase_embedding_m = np.mean(pase_embedding_r, axis = 2).T

    # To DataFrame
    pase_df_m = pd.DataFrame(data = pase_embedding_m, columns = ['Feat_'+ str(x) for x in np.arange(1,257)])
    pase_df_m.insert(0, 'timestamp', timestamps)

    # Saving dataframe
    pase_df_m.to_csv(output_path+'/'+output_filename+'.csv', index=False)

