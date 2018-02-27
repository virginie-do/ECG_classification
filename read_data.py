#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:38:39 2018

@author: virginiedo
"""

import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import librosa


srate = 300

def open_data(filename):   
    mat = scipy.io.loadmat(filename)
    # Conversion microvolts --> mV 
    data = mat['val'][0]/1000
    # Conversion mV --> dBV
    #data = 20*np.log(data/1000)
    return data


def visualization():  
    data = open_data('training2017/A00022.mat')
    length = len(data)
    end = length/srate
    t = np.linspace(0, end, num=length)    
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude (mV)')
    #plt.ylabel('Amplitude (dBV)')
    plt.plot(t,data,label='noisy')
    


def mfcc_features(filename):
    """
    Compute MFCC features
    """
    signal = open_data(filename)
    mfcc_feat = mfcc(signal, srate, numcep=13, appendEnergy=True)
    # make mean of all rows
    features = mfcc_feat.mean(axis=0)
    return features
    

def logfbank_features(filename):
    """
    Compute log Mel-filterbank energy features
    """
    signal = open_data(filename)
    fbank_beat = logfbank(signal, srate)
    # take mean of all rows
    features = fbank_beat.mean(axis=0)
    return features

#open_data('training2017/A00003.mat')
visualization()




