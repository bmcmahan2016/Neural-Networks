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





#define the target function
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

n1 = Neuron('logistic')
n2 = Neuron('logistic')
n3 = Neuron('logistic')
#n4 = Neuron('logistic')
#n5 = Neuron('logistic')
#n6 = Neuron('logistic')
#n7 = Neuron('logistic')

#backpropogation algorithm fails for this input
#need to debug backpropogation algorithm for this input
neurons = [[n1,n2],[n3]]

my_net = Network(neurons)

#x = [1,1]
#my_net.feed_input_layer(x)
#my_net.forward_prop()
my_net.backpropogate([1])






#display initial network

#w.display()

#send in an input
for tt in range(0,10):
    x = Neurons(layers)
    for t in range(0,1000):
        i = random.random()*N
        i = int(i)
        my_net.feed_input_layer([xd[1][i],xd[2][i]])
        my_net.forward_prop()
        #print "Sending input to neural network"
        #x.Show_Status()

        #test backpropogation
        #print "\nDelta is " #%x.compute_delta(1)
        #print "Begining backpropagation..."
        my_net.backpropogate([y[i]])
    xxx = my_net.get_output()

    print "neural net: %f ans: %f" %(xxx[0],y[i])
