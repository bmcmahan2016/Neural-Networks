# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 18:20:33 2017
@author: brandonjamesmcmahan
Project: Create a Neural Network to learn target function
in lecture 10 slide 9 of the Learning from Data Caltech course
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from neuroide import *


#define the target function--this is for non-linear target functions (disabled)
#def target_f(x1,x2):
#    #plot the target function
#    plt.figure(1)
#    plt.title('Target Function')
#    plt.xlabel('X1')
#    plt.ylabel('X2')
#    #we are going to store the result of classification in y
#    y = np.zeros(len(x1))
#    #now classify all points
#    for i in range(0,len(x1)):
#        if x2[i]>x1[i] and x2[i]>(1-x1[i]):
#            y[i] = 1
#            plt.scatter(x1[i],x2[i],c='r')
#        elif x2[i]<x1[i] and x2[i] > (1-x1[i]):
#            y[i] = -1
#            plt.scatter(x1[i],x2[i],c='b')
#        elif x2[i] < (1 - x1[i]) and x2[i] > x1[i]:
#            y[i] = -1
#            plt.scatter(x1[i],x2[i],c='b')
#        elif x2[i] < x1[i] and x2[i] < (1 - x1[i]):
#            y[i] = 1
#            plt.scatter(x1[i],x2[i],c='r')
#        #if point can't be clasified leave it as zero
#        else:
#            y[i] = 0
#    return y
#end target function

def target_f(x1,x2):
    #we are going to store the result of classification in y
    y = np.zeros(len(x1))
    plt.figure(1)
    for i in range(0,len(y)):
        if x2[i] > x1[i]:
            y[i] = 1
            plt.scatter(x1[i],x2[i],c='r')
        else:
            y[i] = 0
            plt.scatter(x1[i],x2[i],c='b')
        plt.title('Training Data')
        plt.xlabel('X1')
        plt.ylabel('X2')
    return y
    
#generate N random samples
N = 100
xd = np.zeros([3,N])
for i in range(0,N):
    xd[0][i] = 1   #threshold needed for perceptron
    xd[1][i] = random.random()
    xd[2][i] = random.random()
#end creation of training data

#classify training data
y = target_f(xd[1],xd[2])


#construct a single layer neural network with logistic neurons
n1 = Neuron('logistic')
n2 = Neuron('logistic')
n3 = Neuron('logistic')

neurons = [[n1,n2],[n3]]
my_net = Network(neurons)



#train the network
for t in range(0,1000):
    #choose a random data point
    i = random.random()*N
    i = int(i)
    #send that point into net
    my_net.feed_input_layer([xd[1][i],xd[2][i]])
    my_net.forward_prop()
    #use SGD and backpropogation to update weights
    my_net.backpropogate([y[i]])
#end network training

#show how well network did
plt.figure(2)
for i in range(N):
    my_net.feed_input_layer([xd[1][i],xd[2][i]])
    my_net.forward_prop()
    clsfctn = my_net.get_output()
    clsfctn = clsfctn[0]
    if np.round(clsfctn) == 1:
        plt.scatter(xd[1][i],xd[2][i],c='r')
    else:
        plt.scatter(xd[1][i],xd[2][i],c='b')
plt.title('Neural Networks Classification of Data')
plt.xlabel('X1')
plt.ylabel('X2')

