#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:07:33 2018

@author: virginiedo
"""

import numpy as np

from sklearn.preprocessing import LabelEncoder, scale, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from keras.utils import np_utils

from create_dataset import create_feature_df

import cma


class Objective_Function(object):
# to be able to inerhit from the function class in Python 2, we need (object)
# f and history_f are attributes of the instance
# we have two instances methods __init__ and __call__
# the __call__ allows to then use fun(x)

    def __init__(self, f):
        self.f = f
        self.history_f = []
        self.fbest = np.inf
        self.history_fbest = []
    
    def __call__(self, x):
        """ Calls the actual objective function in f
            and keeps track of the evaluated search
            points within the history_f list.
        """
        
        f = self.f(x)  # evaluate
        self.history_f.append(f)  # store
        if f < self.fbest:
            self.fbest = f
            
        self.history_fbest.append(self.fbest)

        return f

def objective_XGB(params):
    eta = params[0]**2
    gamma = params[1]**2
    model = XGBClassifier(eta=eta, gamma=gamma)
    model.fit(X_train,y_train)
    return 1 - model.score(X_test, y_test)


def tuned_XGB():
    fun = Objective_Function(objective_XGB)
    res = cma.fmin(fun, [0.3, 0], 1)
    cma.plot()
    params = res[0]
    eta = params[0]**2
    gamma = params[1]**2
    model = XGBClassifier(eta=eta, gamma=gamma)
    return model
    
    

data, labels = create_feature_df('A','N')

labels = labels.values.ravel()
  
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20)

print('Train sets created')

##########################################################################
 
# Random forest
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
print('Random forest score : {}'.format(clf.score(X_test,y_test)))  


# XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
print('XGBoost score : {}'.format(xgb.score(X_test,y_test)))

# XGBoost tuned
xgb2 = tuned_XGB()
xgb2.fit(X_train, y_train)
print('Tuned XGBoost score : {}'.format(xgb.score(X_test,y_test)))



