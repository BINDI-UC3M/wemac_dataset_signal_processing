#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Esther Rituerto-González
Contact: erituert [at] ing [dot] uc3m [dot] es 
         esther [dot] rituerto [dot] g [at] gmail [dot] com
Last updated: February 2023

"""

# Import libraries
import opensmile
import pandas as pd
import numpy as np
    
# =============================================================================
#  Features openSMILE
# =============================================================================
def opensmile_feat_extraction(y, fs):
    
    smile_compare = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    smile_egemaps = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    n_frames = int(len(y)/fs)
    
    for n_frame in range(0, n_frames):
        
        i = n_frame*fs
        j = i+fs
        
        compare_feats = smile_compare.process_signal(y[i:j], sampling_rate=fs)
        egemaps_feats = smile_egemaps.process_signal(y[i:j], sampling_rate=fs)
        
        # Concat features
        if (n_frame == 0):
            compare_columns = compare_feats.columns
            egemaps_columns = egemaps_feats.columns

            full_compare_feats = compare_feats.values
            full_egemaps_feats = egemaps_feats.values
        else:
            full_compare_feats = np.vstack((full_compare_feats, compare_feats.values))
            full_egemaps_feats = np.vstack((full_egemaps_feats, egemaps_feats.values))
            
    compare_df = pd.DataFrame(data = full_compare_feats, columns = compare_columns)
    egemaps_df = pd.DataFrame(data = full_egemaps_feats, columns = egemaps_columns)

    return compare_df, egemaps_df
    
    
    