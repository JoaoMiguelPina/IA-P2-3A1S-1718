# 83488 - Joao Nogueira; 85080 - Joao Pina; Grupo 13

# -*- encoding: utf-8 -*-

import numpy as np
from sklearn import neighbors, datasets, tree, linear_model

from sklearn.externals import joblib
import timeit

from sklearn.model_selection import cross_val_score

def features(X):
    
    F = np.zeros((len(X),5))
    for x in range(0,len(X)):
	
        suma = 0
        for char in X[x]:
            suma += ord(char)

        i = round(len(X[x]) / 2)
		
        F[x,0] = len(X[x])
        F[x,1] = suma
        F[x,2] = ord(X[x][0])
        F[x,3] = ord(X[x][-1])
        F[x,4] = ord(X[x][i])

    return F       

def mytraining(f,Y):
    
    min_samples_split = 2
    clf = tree.DecisionTreeClassifier(min_samples_split=min_samples_split)
    clf = clf.fit(f, Y)    
   
    return clf
    
def mytrainingaux(f,Y,par):
    
    return clf

def myprediction(f, clf):
    Ypred = clf.predict(f)

    return Ypred

