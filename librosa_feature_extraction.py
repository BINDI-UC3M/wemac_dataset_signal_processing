# Import libraries
import numpy as np
from audio_tools import reshape1sec
import librosa
import pandas as pd

def librosa_feature_extraction(y, fs, win, step, n_mfcc, NFFT):
    
    seconds = int(len(y)/fs)
    
    # =============================================================================
    # Features 
    # =============================================================================
   
    # =============================================================================
    # Compute MFCC features from the raw signal (uses hannning window). 
    # Librosa starts computing mfcc 1
    # =============================================================================
    raw_mfcc = librosa.feature.mfcc(y=y, sr=fs, hop_length=step, n_mfcc=n_mfcc, n_fft=NFFT) # 160000 samples en ventanas de 20ms con hop size de 10ms
    
    mean_mfcc, std_mfcc = reshape1sec(raw_mfcc, seconds)
    
    # =============================================================================
    # Other features (use rectangular window)
    # =============================================================================
    
    # RMS or Energy
    raw_rms = librosa.feature.rms(y=y, frame_length=win, hop_length=step, center=True, \
                              pad_mode='reflect')
    mean_rms, std_rms = reshape1sec(np.ravel(raw_rms), seconds)
    
    # Zero Crossing Rate
    raw_zcr = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=step, \
                                             center=True)
    mean_zcr, std_zcr = reshape1sec(np.ravel(raw_zcr),seconds)
    
    # Spectral Centroid
    raw_centroid = librosa.feature.spectral_centroid(y=y, sr=fs, n_fft=NFFT, \
                                                 hop_length=step, win_length=win, \
                                                 window='hann', center=True, \
                                                 pad_mode='reflect')
    mean_centroid, std_centroid = reshape1sec(np.ravel(raw_centroid),seconds)
    
    # Spectral Flux (not found)
    ## raw_flux = librosa.onset.onset_strength(y=y, sr=fs, lag=1, max_size=1, \
    #                                             detrend=False, center=True)
    # mean_flux, std_flux = np.zeros([1,seconds]),np.zeros([1,seconds])
    
    
    # Spectral Rolloff
    raw_rolloff = librosa.feature.spectral_rolloff(y=y, sr=fs, n_fft=NFFT, \
                                               hop_length=step, win_length=win, \
                                               window='hann', center=True, \
                                               pad_mode='reflect', freq=None, \
                                               roll_percent=0.95)
    mean_rolloff, std_rolloff = reshape1sec(raw_rolloff, seconds)
    
    # Spectral Flatness
    raw_flatness = librosa.feature.spectral_flatness(y=y,n_fft=NFFT, hop_length=step, \
                                                     win_length=win, window='hann', \
                                                     center=True, pad_mode='reflect', \
                                                     amin=1e-10, power=2.0)
    mean_flatness, std_flatness = reshape1sec(raw_flatness, seconds)
    
    # # Band Energy Ratio (not found)
    # mean_ber, std_ber = np.zeros([1,seconds]),np.zeros([1,seconds])
    
    # =============================================================================
    # Pitch extraction
    # =============================================================================
    pitch_track, mag = librosa.piptrack(y=y, sr=fs, n_fft=NFFT, hop_length=step, \
                 fmin=0.0, fmax=4000.0, threshold=0.1, \
                 win_length=win, window='hann', center=True, pad_mode='reflect', \
                 ref=None)
    
    # Only count magnitude where frequency is > 0
    raw_pitch = np.zeros([mag.shape[1]])
    for i in range(0,mag.shape[1]):
        index = mag[:,i].argmax()
        raw_pitch[i] = pitch_track[index,i]
    
    mean_pitch, std_pitch = reshape1sec(raw_pitch, seconds)
    
    
    # =============================================================================
    # Concat features
    # =============================================================================
    features_names = ['MFCC0', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6',\
                'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'Energy', \
                'ZCR', 'Centroid', 'Rolloff', 'Flatness', 'Pitch']
    # 'ZCR', 'Centroid', 'Flux', 'Rolloff', 'Flatness', 'BER', 'Pitch']
    mean_features = [mean_mfcc[0,:], mean_mfcc[1,:], mean_mfcc[2,:], mean_mfcc[3,:], \
                     mean_mfcc[4,:], mean_mfcc[5,:], mean_mfcc[6,:], mean_mfcc[7,:], \
                     mean_mfcc[8,:], mean_mfcc[9,:], mean_mfcc[10,:], mean_mfcc[11,:], \
                     mean_mfcc[12,:], mean_rms[0,:], mean_zcr[0,:], mean_centroid[0,:],  \
                     mean_rolloff[0,:], mean_flatness[0,:], mean_pitch[0,:]]
        # mean_flux[0,:], mean_rolloff[0,:], mean_flatness[0,:], mean_ber[0,:],\
    std_features = [std_mfcc[0,:], std_mfcc[1,:], std_mfcc[2,:], std_mfcc[3,:], \
                     std_mfcc[4,:], std_mfcc[5,:], std_mfcc[6,:], std_mfcc[7,:], \
                     std_mfcc[8,:], std_mfcc[9,:], std_mfcc[10,:], std_mfcc[11,:], \
                     std_mfcc[12,:], std_rms[0,:], std_zcr[0,:], std_centroid[0,:],  \
                     std_rolloff[0,:], std_flatness[0,:], std_pitch[0,:]]
    # std_flux[0,:], std_rolloff[0,:], std_flatness[0,:], std_ber[0,:], \
    
    
    # =============================================================================
    # Save Features 1s
    # =============================================================================  
    df_mean = pd.DataFrame(data = np.array(mean_features).T, columns = ['Mean '+ x for x in features_names])
    df_std = pd.DataFrame(data = np.array(std_features).T, columns = ['Std '+ x for x in features_names])
    features_df = pd.concat([df_mean, df_std], sort=False,axis=1)
    
    return features_df        
    
    
