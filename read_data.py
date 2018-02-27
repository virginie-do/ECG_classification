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
import librosa.display


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
    


def logspec_features(filename):
    signal = open_data(filename)    
    features = librosa.amplitude_to_db(librosa.stft(signal), ref=np.max)
    # take mean of all rows
    features = features.T.mean(axis=0)
    return features


def mfcc_features(filename):
    """
    Compute MFCC features
    """
    signal = open_data(filename)
    mfcc_feat = librosa.feature.mfcc(signal, srate)
    # take mean of all rows
    features = mfcc_feat.T.mean(axis=0)
    return features
    

def tempo_features(filename):
    """
    Compute MFCC features
    """
    signal = open_data(filename)
    tempo_feat = librosa.feature.tempogram(signal, srate)
    # take mean of all rows
    features = tempo_feat.T.mean(axis=0)
    return features
    

#open_data('training2017/A00003.mat')
#visualization()


def plot_spectrogram(filename):
    plt.figure(figsize=(12, 8))
    signal = open_data(filename)
    D = librosa.amplitude_to_db(librosa.stft(signal), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.subplot(4, 2, 2)
    librosa.display.specshow(D, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')


#plot_spectrogram('training2017/A00003.mat')
