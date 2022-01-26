#! /usr/bin/python

# Voice Activity Detection (VAD) tool.
# use the vad_help() function for instructions.
# Navid Shokouhi December 2012.
# https://github.com/idnavid/py_vad_tool
# Updated: May 2017 for Speaker Recognition collaboration.

from audio_tools import enframe, deframe
import numpy as np


##Function definitions:
def vad_help():
    """Voice Activity Detection (VAD) tool.
	
	Navid Shokouhi May 2017.
    """
    print("Usage:")
    print("python unsupervised_vad.py")


#### Energy tools
def zero_mean(xframes):
    """
        remove mean of framed signal
        return zero-mean frames.
        """
    m = np.mean(xframes,axis=1)
    xframes = xframes - np.tile(m,(xframes.shape[1],1)).T
    return xframes

def compute_nrg(xframes):
    # calculate per frame energy
    n_frames = xframes.shape[1]
    return np.diagonal(np.dot(xframes,xframes.T))/float(n_frames)

def compute_log_nrg(xframes):
    # calculate per frame energy in log
    n_frames = xframes.shape[1]
    raw_nrgs = np.log(compute_nrg(xframes+1e-5))/float(n_frames)
    return (raw_nrgs - np.mean(raw_nrgs))/(np.sqrt(np.var(raw_nrgs)))

def power_spectrum(xframes):
    """
        x: input signal, each row is one frame
        """
    X = np.fft.fft(xframes,axis=1)
    X = np.abs(X[:,:X.shape[1]/2])**2
    return np.sqrt(X)

def nrg_vad(xframes,percent_thr,nrg_thr=0.,context=5):
    """
        Picks frames with high energy as determined by a 
        user defined threshold.
        
        This function also uses a 'context' parameter to
        resolve the fluctuative nature of thresholding. 
        context is an integer value determining the number
        of neighboring frames that should be used to decide
        if a frame is voiced.
        
        The log-energy values are subject to mean and var
        normalization to simplify the picking the right threshold. 
        In this framework, the default threshold is 0.0
        """
    xframes = zero_mean(xframes)
    n_frames = xframes.shape[0]
    
    # Compute per frame energies:
    xnrgs = compute_log_nrg(xframes)
    xvad = np.zeros((n_frames,1))
    for i in range(n_frames):
        start = max(i-context,0)
        end = min(i+context,n_frames-1)
        n_above_thr = np.sum(xnrgs[start:end]>nrg_thr)
        n_total = end-start+1
        xvad[i] = 1.*((float(n_above_thr)/n_total) > percent_thr)
    return xvad


def process_vad(s, fs, win_len, hop_len, frame_len_s, percent_high_nrg):
        
    sframes = enframe(s,win_len,hop_len) # rows: frame index, cols: each frame
    
    vad = nrg_vad(sframes,percent_high_nrg)
    vad_signal = deframe(vad,win_len,hop_len)
        
    # Windowing of 1 sec
    windowing_len = int(frame_len_s*fs);
    
    ini = 0
    fin = windowing_len
    n_frames = int(np.floor(len(s)/windowing_len))
    vad_vector = np.zeros(n_frames)
    for i in range(0,n_frames-1):
        frame = vad_signal[ini:fin]
        value = np.median(frame)
        vad_vector[i] = value
        
        ini = fin+1
        fin = fin+windowing_len
    
    return vad_vector


