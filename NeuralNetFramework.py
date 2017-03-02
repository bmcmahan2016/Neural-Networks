# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 18:20:33 2017

@author: brandonjamesmcmahan
"""
import numpy as np
#import matplotlib.pyplot as plt
import random

#create an object that will store all weights
class Weights(object):
    #construct weight matrix
    #we will index each weight matrix by the layer number starting at one
    def __init__(self,layers):
        self.weights = {}
        for i in range(1,len(layers)+1):
            #to each layer, assign a numpy array
            self.weights[i] = random.random()*np.ones(layers[i])
    
    #this function will return projections to output neuron in layer
    def weight(self,layer,outpt):
        return self.weights[layer][:,outpt]
        
    #this function will update all weights using SGD    
    def update_weights(self,x):
        #update each layer one at a time
        for l in range(1,len(x.x)):
            #print "Updating layer %.2d" %l
            for j in range(1,len(x.x[l])-1):
                #print "Updating neuron %.2d" %j
                #need to translate this formula
                w.weights[l][:,j-1]=w.weights[l][:,j-1]-(0.1)*x.x[l-1][:]*x.delta[l][j]
            #update weights for final layer
            if l == len(x.x)-1:
                for j in range(len(x.x[l])):
                    w.weights[l][:,j]=w.weights[l][:,j]-(0.1)*x.x[l-1]*x.delta[l]
                
    def display(self):
        print "\nWEIGHTS"
        for i in range(1,len(self.weights)+1):
            print "\nlayer %.2d" %i
            print self.weights[i]
    
#create neuron object
class Neurons(object):
    #construct a neural network
    #neuron will be indexed by non-zero because 1 threshold is the zero input
    def __init__(self, layers):
        self.x = {}
        self.delta = {}
        self.x[0] = np.zeros(layers[1]+1)   #we need d+1 inputs to first layer
        self.x[0][0] = 1                    #theshold
        for i in range(1,len(layers)):
            self.x[i] = np.zeros(layers[i]+1)   #we need an extra input in each layer for threshold
            self.x[i][0] = 1                    #threshold
            self.delta[i] = np.zeros(layers[i])
        i = len(layers)
        self.x[i] = np.zeros(layers[i])
        self.delta[i] = np.zeros(layers[i])
        
    #this function will update the status of all neurons in the network    
    def Feed_Forward(self, inpt, w):
        #update one layer at a time, starting at layer 1 (0 = inputs)
        self.x[0][1:] = inpt        
        for l in range(1,len(self.x)):
            #in the current layer, update each neuron one at a time
            for j in range(1,len(self.x[l])):
                self.x[l][j] = np.tanh(np.dot(w.weight(l,j-1),self.x[l-1]))
            if len(self.x[l]) == 1:     #final neuron
                self.x[l][0] = np.tanh(np.dot(w.weight(l,0),self.x[l-1]))
                
    def Show_Status(self):
        #print"\nNETWORK STATUS"
        for i in range(0,len(self.x)):
            print "\nLayer %.2d" %i
            print self.x[i]
            
    def compute_delta(self,y_n):
        l = len(self.x)-1   #final layer
        self.delta[len(self.x)-1] = 2*(self.x[l]-y_n)*(1-self.x[l]**2)
        
    def backpropogate(self,delta):
        L = len(self.x)-1
        #compute delta one layer at a time, starting with L-1 to the first layer
        for l in range(1,L+1):
            #for each neuron j in layer l we compute delta
            #we skip the zero neuron as this is only a threshold
            #print "in layer %.2d" %l
            for j in range(1,len(self.x[l])):
                #print "Computing delta for neuron %.2d" %j
                #need to check indexing on this statement
                self.delta[L-l][j-1] = (1-self.x[L-l][j]**2)*np.dot(w.weights[L-l+1][j-1],self.delta[L-l+1]) 


##create delta object  
#class Delta(object):
#    #construct deltas
#    def __init__(self,layers):
#        self.delta = {}
#        for i in range(1,len(layers)+1):
#            self.delta[i] = np.zeros(layers[i])
#    
#    def backpropogate(self,delta):
#        #first get delta for largest layer
#        delta = 2*(y_n)*(1-(neuron)**2)
        

#construct objects
layers = {1: [4,3], 2:[4,3],3:[4,1]}
w = Weights(layers)
layers = {1:3,2:3,3:1}
x = Neurons(layers)

#display initial network

#w.display()

#send in an input
for tt in range(0,10):
    x = Neurons(layers)
    for t in range(0,10):
        x.Feed_Forward([10,10,10],w)
        #print "Sending input to neural network"
        #x.Show_Status()

        #test backpropogation
        #print "\nDelta is " #%x.compute_delta(1)
        #print "Begining backpropagation..."
        x.backpropogate(x.compute_delta(-0.3))
            
#       update network weights
        w.update_weights(x)  

    print x.x[3]      

 
            
'''
#define the target function
def target_f(x1,x2):
    #plot the target function
    plt.figure(1)
    plt.title('Target Function')
    plt.xlabel('X1')
    plt.ylabel('X2')
    #we are going to store the result of classification in y
    y = np.zeros(len(x1))
    #now classify all points
    for i in range(0,len(x1)):
        if x2[i]>x1[i] and x2[i]>(1-x1[i]):
            y[i] = 1
            plt.scatter(x1[i],x2[i],c='r')
        elif x2[i]<x1[i] and x2[i] > (1-x1[i]):
            y[i] = -1
            plt.scatter(x1[i],x2[i],c='b')
        elif x2[i] < (1 - x1[i]) and x2[i] > x1[i]:
            y[i] = -1
            plt.scatter(x1[i],x2[i],c='b')
        elif x2[i] < x1[i] and x2[i] < (1 - x1[i]):
            y[i] = 1
            plt.scatter(x1[i],x2[i],c='r')
        #if point can't be clasified leave it as zero
        else:
            y[i] = 0
    return y
#end target function
    

print "\n\n\n-----------------------------------"
print   "begining Neural Network Algorithm" 
print "-----------------------------------\n"   
#generate N random samples
N = 100
x = np.zeros([3,N])
for i in range(0,N):
    x[0][i] = 1   #threshold needed for perceptron
    x[1][i] = random.random()
    x[2][i] = random.random()
#end creation of training data

#classify training data
y = target_f(x[1],x[2])

#initialize weight matrix
#3 inputs --> 3 neurons --> 2 neurons --> 1 neuron
layer_1 = random.random()*np.ones([3,3])
layer_2 = random.random()*np.ones([4,2])
layer_3 = random.random()*np.ones([3,1])
w = {1:layer_1,2:layer_2,3:layer_3}
#end initialization of weight matrix

#we initialize all neurons in the network
#initialize dictionary that will store delta values

delta = {1:np.array([0.0,0.0,0.0]) , 2:np.array([0.0,0.0]) , 3:np.array([0.0])}

error = 1
epochs = 0
#SART ITERATION
for t in range(0,100):
    #epochs += 1
    #print "EPOCH: %f" %epochs
    #we pick an initial random point as input to the network
    i = random.random()*N
    i = int(i)
    N0 = [x[0][i],x[1][i],x[2][i]]
    y_n = y[i]
    neurons = {0:N0, 1:np.array([1,0.1,0.1,0.1]), 2:N0, 3:N0[1]}
    
    #FEEDFORWARD
    #first layer
    for j in range(1,len(neurons[1])-1):
        neurons[1][j] = np.tanh(np.dot(w[1][:,j],neurons[0]))
    #compute second layer
    for j in range(1,len(neurons[2])-1):
        neurons[2][j] = np.tanh(np.dot(w[2][:,j],neurons[1]))
    #compute third layer
    neurons[3] = np.tanh(np.dot(w[3][:,0],neurons[2]))
    #END FEEDFORWARD

    #BACKPROPOGATION
    #compute delta for final layer
    delta[3] = 2*(neurons[3]-y_n)*(1-(neurons[3])**2)
    #compute delta for 2nd layer
    for i in range(0,len(delta[2])):
        delta[2][i] = (1 - neurons[2][i+1]**2)*np.dot(w[3][i+1,:],delta[3])
    #compute delta for first layer
    for i in range(0,len(delta[1])):
        delta[1][i] = (1 - neurons[1][i+1]**2)*np.dot(w[2][i+1,:],delta[2][:])
    #END BACKPROPOGATION

    #UPDATE WEIGHTS
    #for first layer
    for j in range(0,len(delta[1])):
        for i in range(0,len(neurons[0])):
            w[1][i,j] = w[1][i,j]-0.1*neurons[0][i]*delta[1][j]
    #for second layer
    for j in range(0,len(delta[2])):
        for i in range(0,len(neurons[1])):
            w[2][i,j] = w[2][i,j]-0.1*neurons[1][i]*delta[2][j]
    #for third layer
    for i in range(0,len(neurons[2])):
        w[3][i] = w[3][i]-0.1*neurons[2][i]*delta[3]
    #END UPDATE WEIGHTS
#END ITERATION
        
#get in sample error
error_0=0
for i in range(0,N):
    y_n = y[i]
    #first layer
    for j in range(1,len(neurons[1])-1):
        neurons[1][j] = np.tanh(np.dot(w[1][:,j],neurons[0]))
    #compute second layer
    for j in range(1,len(neurons[2])-1):
        neurons[2][j] = np.tanh(np.dot(w[2][:,j],neurons[1]))
    #compute third layer
    neurons[3] = np.tanh(np.dot(w[3][:,0],neurons[2]))
    error_0 = error_0 + (np.sign(neurons[3])-y_n)**2
error = error_0/N
print "In sample error: %f" %error

#classify this data using the neural network and then plot this data as a scatter
#classified_data = np.zeros(N)
#for i in range(0,N):
#    classified_data[i] = np.tanh(np.dot(w,[x[0][i],x[1][i],x[2][i]]))
#    plt.figure(2)
#    if np.sign(classified_data[i]) > 0:
#        plt.scatter(x[1][i],x[2][i],c='r')
#    elif np.sign(classified_data[i]) < 0:
#        plt.scatter(x[1][i],x[2][i],c='b')
#plt.title('Data Classified by Neural Network')
#plt.xlabel('X1')
#plt.ylabel('X2')



#generate new data for testing
#generate N random samples
N = 1000
x = np.zeros([3,N])
for i in range(0,N):
    x[0][i] = 1   #threshold needed for perceptron
    x[1][i] = random.random()
    x[2][i] = random.random()
#end creation of training data

#classify training data
y = target_f(x[1],x[2], 0)

#get out of sample error
error_0=0
for i in range(0,N):
    N0 = [x[0][i],x[1][i],x[2][i]]
    y_n = y[i]
    neurons = {0:N0, 1:np.array([1,0.1,0.1,0.1]), 2:N0, 3:N0[1]}
    #first layer
    for j in range(1,len(neurons[1])-1):
        neurons[1][j] = np.tanh(np.dot(w[1][:,j],neurons[0]))
    #second layer
    for j in range(1,len(neurons[2])-1):
        neurons[2][j] = np.tanh(np.dot(w[2][:,j],neurons[1]))
    #Third layer
    neurons[3] = np.tanh(np.dot(w[3][:,0],neurons[2]))    
    error_0 = error_0 + (neurons[3]-y_n)**2
error = error_0/N
print "out of sample error: %f" %error


















#     
#    #get in sample error
#    error_0=0
#    for i in range(0,N):
#        N0 = [x[0][i],x[1][i],x[2][i]]
#        y_n = y[i]
#        neurons = {0:N0, 1:np.array([1,0.1,0.1,0.1]), 2:N0, 3:N0[1]}
#        #first layer
#        for j in range(1,len(neurons[1])-1):
#            neurons[1][j] = np.tanh(np.dot(w[1][:,j],neurons[0]))
#        #second layer
#        for j in range(1,len(neurons[2])-1):
#            neurons[2][j] = np.tanh(np.dot(w[2][:,j],neurons[1]))
#        #Third layer
#        neurons[3] = np.tanh(np.dot(w[3][:,0],neurons[2]))    
#        error_0 = error_0 + (neurons[3]-y_n)**2
#    error = error_0/N
#    print "In sample error: %f" %error


#plot training data as a scatter
#for i in range(0,N):
#    if y[i] > 0:
#        plt.scatter(x[1][i],x[2][i],c='r')
#    elif y[i] < 0:
#        plt.scatter(x[1][i],x[2][i],c='b')
#    plt.title('Target Function')
'''
