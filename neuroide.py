#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:14:32 2017

@author: Brandon McMahan
NeuroIDE (neuroide)
Neural Network Development Module
Description: This provides module provides defines classess that allow for easy creation and implementation of Neural Networks 
"""
import numpy as np

#BEGIN NEURON CLASS
class Neuron(object):
    #construct a neuron object
    def __init__(self,fcn):
        self.state = 0  #start the neuron with an initial state of zero
        self.fcn = fcn  #user enters type of neurons they would like to use
    
    #method to update neuron activity
    def inpt(self,x,w):
        x = np.insert(x,[0],1)      #append a one to input data to take care of bias
        if self.fcn == 'logistic':
            z = np.dot(w,x)
            self.state = (1 + np.exp(-z))**(-1)
        elif self.fcn == 'linear':
            self.state = np.dot(w,x)
        elif self.fcn == 'softmax':
            pass
        else:
            self.state = 0
#END NEURON CLASS   
    
#BEGIN NETWORK CLASS
class Network(object): 
    #BEGIN CONSTRUCTION OF NETWORK OBJECT
    def __init__(self,neurons):
        #create a network with neurons in each layer
        self.neurons = []
        self.w = []
        self.dE_by_dw = []
        noLayers = len(neurons)     #get the number of layers our network will have
        #for each layer, we need to append a single neuron to the front of the list that will act as a place holder for the bias
        #create network layers
        for i in range(noLayers):
            self.neurons.append(neurons[i])
        #create network weights
        for i in range(noLayers):
            if i >= 1:
                self.w.append(0.1*np.ones([len(self.neurons[i-1])+1,len(self.neurons[i])]))
                self.dE_by_dw.append(0*np.ones([len(self.neurons[i-1])+1,len(self.neurons[i])]))
                self.w[i][0] = 1    #the first row stands in for the bias
                self.dE_by_dw[i][0] = 0
            elif i == 0:
                self.w.append(np.array([]))
                self.dE_by_dw.append(np.array([]))
    #END CONSTRUCTION OF NETWORK OBJECT
    
    def feed_input_layer(self,x):
        for i in range(len(self.neurons[0])):
            neurons[0][i].state = x[i]
    
    #method to forward propagate input through the network
    #BEGIN FORWARD PROPOGATION
    def forward_prop(self):
        noLayers = len(self.neurons)
        #propagate through layers starting at layer one
        #there is no need to update the input layer (layer 0)
        for l in range(1,noLayers):
            
            #extract outputs of layer l-1 as input to layer l
            x = []
            for ii in range(len(self.neurons[l-1])):
                x.append(self.neurons[l-1][ii].state)
            #end extraction of output from layer l-1
            #print x
            #update each neuron in layer l
            for i in range(len(self.neurons[l])):
                self.neurons[l][i].inpt(x,self.w[l][:,i])
            #end update each neuron in layer l
            
    #END FORWARD PROPOGATION
    
    
    
    #method to compute squared error on a training point
    def error_fcn(self,t):
        #first we need to get the network output
        self.get_output()
        self.error = 0.5*(t - self.y)**2    #returns error for each neuron in output layer
        print self.error
    #end error function
    
    #method to backpropagate the output layer
    #BACKPROPOGATE FOR OUTPUT LAYER
    def start_backprop(self,t):
        #compute dE/dy for all neurons in output layer
        outLayer = len(self.neurons)-1  #this is output layer number

        #for each neuron, j, in the output layer compute partials
        for j in range(len(self.neurons[outLayer])):
            #dE/dy
            self.dE_by_dy[outLayer][j] = self.neurons[outLayer][j].state-t[j]  
            #dE/dz
            self.dE_by_dz[outLayer][j] = self.neurons[outLayer][j].state*(1-self.neurons[outLayer][j].state)*self.dE_by_dy[outLayer][j] 
        #get all weight derivatives
        #loop over all neurons in output layer
        for j in range(len(self.neurons[outLayer])):
            #loop over all inputs to neuron j in layer l (this include all neurons in layer l-1 AND a bias input)
            for i in range(len(self.neurons[outLayer-1])+1):
                #print "updating weight from neuron %.2d to output neuron %.2d" %(i,j)
                if i == 0:  #bias input
                    self.dE_by_dw[outLayer][i,j] = 1*self.dE_by_dz[outLayer][j]
                else:   #input from another neuron
                    self.dE_by_dw[outLayer][i,j] = self.neurons[outLayer-1][i-1].state*self.dE_by_dz[outLayer][j]
    #END BACKPROPOGATE FOR OUTPUT LAYER
    
    def back_one(self,l):
        #get partial derivitives for layer l
        #for j in range(len(self.layer[l+1])):
        #    dE_by_dz_above[j] = self.layer[l+1][j].state()*(1-self.layer[l+1][j].state())*dE_by_dy_above[j]
        
        #loop over all neurons in the current layer to get dE/dy for each neuron in this layer
        for i in range(len(self.neurons[l])): 
            #print "Computing partials for neuron %.2d in layer %.2d" %(i+1, l)
            self.dE_by_dy[l][i] = np.dot(self.w[l+1][i+1,:],self.dE_by_dz[l+1])
            self.dE_by_dz[l][i] = self.neurons[l][i].state*(1-self.neurons[l][i].state)*self.dE_by_dy[l][i]
        #now update all weights from layer l-1 to l
        
        #get all weight derivatives
        #loop over all neurons in layer output layer
        for j in range(len(self.neurons[l])):
            #loop over all inputs to neuron j in layer l (this include all neurons in layer l-1 AND a bias input)
            for i in range(len(self.neurons[l-1])+1):
                #print "updating weight from neuron %.2d to output neuron %.2d" %(i,j)
                if i == 0:  #bias input
                    self.dE_by_dw[l][i,j] = 1*self.dE_by_dz[l][j]
                else:   #input from another neuron
                    self.dE_by_dw[l][i,j] = self.neurons[l-1][i-1].state*self.dE_by_dz[l][j]    #swaped l and j in last term

    def backpropogate(self,t):
        #initialize all derivitives
        self.dE_by_dy = []          #this will become a list of np arrays
        self.dE_by_dz = []          #this will become a list of np arrays
        
        #loop through all layers
        for l in range(len(self.neurons)):
            self.dE_by_dy.append(np.zeros([len(self.neurons[l])]))
            self.dE_by_dz.append(np.zeros([len(self.neurons[l])]))
        
        #now that derivitives have been initialized get them for output layer
        self.start_backprop(t)
        #now propogate through remaining layers
        #we need to go through layers in reverse order from top down
        for l in reversed(range(1,len(self.neurons)-1)):
            self.back_one(l)
        #now we must update all the weights 
        #loop through all layers
        for l in range(1,len(self.neurons)):  
            #loop through each neuron in layer l
            for i in range(len(self.neurons[l-1])+1):
                #for each neuron i in layer l, loop through all inputs (neurons in layer below AND bias)
                for j in range(len(self.neurons[l])):
                    #may want to make learning rate more general in future
                    self.w[l][i,j] = self.w[l][i,j] - 0.1*self.dE_by_dw[l][i,j]
    #END BACKPROPOGATION
    
    #method to get the output of the network
    def get_output(self):
        outLayer = len(self.neurons)-1    #this is the output layer
        self.y = []
        for i in range(len(self.neurons[outLayer])):
            self.y.append(self.neurons[outLayer][i].state)
        return self.y
    #end method to get the output of the network
#END NETWORK CLASS
