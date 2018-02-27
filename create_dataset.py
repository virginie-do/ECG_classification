
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:07:33 2018

@author: virginiedo
"""

import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from read_data import open_data, logspec_features, mfcc_features, tempo_features


#selection de 50 noms de fichiers depuis le fichier labels.csv
def select(class1,class2):
    
    data=pd.read_csv('labels.csv',sep=',')
    counter1=0
    counter2=0
    step=0
    files1,files2=[],[]
    while (counter1<50 or counter2<50):
        
        if(data.values[step][1]==class1):
            counter1+=1
            files1.append(data.values[step][0])
        elif(data.values[step][1]==class2):
            counter2+=1
            files2.append(data.values[step][0])
        step+=1
        
    print('Selection done')        
    return files1, files2


#creation des sets d'apprentissage pour deux classes
def create_db(files1, files2, class1, class2):
    
    data = []
    labels = []
    for file in files1:
        
        ft = mfcc_features('training2017/'+file)
        data.append(ft)
        labels.append(class1)
        
    for file in files2:
        
        ft = mfcc_features('training2017/'+file)
        data.append(ft)
        labels.append(class2)
        
    print('Database created') 
    
    return data, labels


def create_fnames_df(class1, class2):
    
    files1, files2 = select(class1, class2)
    df = []
    for file in files1:
        df.append([file, class1])
    for file in files2:
        df.append([file, class2])
    training_df = pd.DataFrame(df)    
    training_df.columns = ['filename', 'label']      
    return training_df
    


def create_feature_df(class1, class2):

    """
    Create feature matrix
    with a combination of selected signal processing features
    """
    
    training_df = create_fnames_df(class1, class2)
    features_df = pd.DataFrame()

    for i in range(0, len(training_df)):
        file = "training2017/{0}".format(training_df.iloc[i]["filename"])
        mfcc_feat = mfcc_features(file)
        logspec_feat = logspec_features(file)
        tempo_feat = tempo_features(file)
        comb_feat = np.append(mfcc_feat, logspec_feat)
        comb_feat = np.append(comb_feat, tempo_feat)
        
        #comb_feat = mfcc_features(file)
        features_df = features_df.append([comb_feat], ignore_index=True)

    # labels
    labels = pd.DataFrame()
    for i in range(0, len(training_df)):
        labels = labels.append([training_df.iloc[i]["label"]], ignore_index=True)
        
    print("Feature matrix created")

    return features_df, labels