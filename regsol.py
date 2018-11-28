# 83488 - Joao Nogueira; 85080 - Joao Pina; Grupo 13

import numpy as np
from sklearn import datasets, tree, linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
import timeit

def mytraining(X,Y):
    
    reg = KernelRidge(kernel='rbf', gamma=0.06202, alpha = 0.00001)
    reg.fit(X,Y)      
   
    return reg

def mytrainingV2(X,Y):
    
    reg = KernelRidge(kernel='polynomial', gamma=0.046, alpha = 0.001, degree=5)
    reg.fit(X,Y)      
   
    return reg
    
def mytrainingaux(X,Y,par):
    
    reg.fit(X,Y)
                
    return reg

def myprediction(X,reg):

    Ypred = reg.predict(X)

    return Ypred
