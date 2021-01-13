#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:11:04 2018
@author: owais

z1,z2 = Weighted input
x,y = Input/Output
c = Context/Hidden Layer
a1 = First activation layer
a2 = Second activation layer/ yhat, the estimated output
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x)*(1-sigmoid(x))

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu(x):
    return np.maximum(0,x)

def relu_backward(dA, x):
    dZ = np.array(dA, copy = True)
    dZ[x <= 0] = 0
    return dZ

#Read and Parse data into y, the input
ya = pd.read_csv('a.csv')
y = np.zeros((len(ya),2))
for i in range(len(ya)):
    y[i] = ya.values[i]     #Input layer

#x = np.arange(10)
#np.roll(x, 2)
#array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
#np.roll(x, -2)
#array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1]

#Build output matrix for prediction    
x = np.roll(y,1,axis=0)  
x[0] = [0.65,0.7]        #Output layer

#Input to Hidden weight and bias matrix
w1 = np.random.rand(2,5)

#Context to Hidden weight matrix
w2 = np.random.rand(5,5)
b1 = np.random.rand(1,5)

# Hidden to output weight matrix
w3 = np.random.rand(5,2)
b2 = np.random.rand(1,2)    #Second bias

#Full forward Propagation

c = np.zeros((1,5))
a1 = np.zeros((len(x),5))
z1 = np.zeros((len(x),5))
z2 = np.zeros((len(x),2))
a2 = np.zeros((len(x),2))
mse = np.zeros((len(x),2))
eta = 0.01
delta2 = np.zeros((len(x),2))
delta1 = np.zeros((len(x),5))
e = np.zeros((len(x),2))

for k in range(10000):     #Epochs
    for i in range(len(x)):
        #Input to hidden layer
        z1[i] = np.dot(y[i],w1)+ np.dot(c,w2) + b1  #Weighted Input 1st layer
        a1[i] = sigmoid(z1[i])
        c = a1[i]        #Context layer = Hidden layer
        #Hidden to output layer
        z2[i] = np.dot(c,w3) + b2     #Second Weighted Input values
        a2[i] = sigmoid(z2[i])
        #Mean Squared Error
        e[i] = np.subtract(a2[i], x[i])
        mse[i] = 0.5*np.square(e[i])
        
        #BackProp   
        delta2[i] = e[i] * sigmoid_prime(z2[i])   # [1*2]
        delta1[i] = np.dot(delta2[i], w3.T) * sigmoid_prime(z1[i])   #[1*5]
        
    for j in reversed(range(len(x)-1)):
        delta1[j] += sigmoid_prime(z1[j+1])* np.dot(delta1[j+1],w2.T)   #[1*5]  
    
    w3 -= (eta/2)*(np.dot(a1.T,delta2))     #[5*2]
    w2 -= (eta/2)*(np.dot(a1.T,delta1))     #[5*5]
    w1 -= (eta/2)*(np.dot(x.T,delta1))      #[2*5]
    
    b2 -= (eta/2) * np.sum(delta2,axis = 0)
    b1 -= (eta/2) * np.sum(delta1, axis = 0)
    print('Loss for epoch '+str(k)+ '  is  ' +str(np.sum(mse)))
    
plt.plot(x[:,0], x[:,1])
plt.plot(a2[:,0], a2[:,1])