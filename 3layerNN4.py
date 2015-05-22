#!/usr/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import scipy.io as sio
import math as ma
import random as rnd

def column(X,j):
  return [row[j] for row in X]

def columns(X):
    for j in range(0, len(X[0])):
        yield column(X,j)


def normalise_data(data):
    data_transpose = []
    normalised_data = []

    for col in columns(data):
        tmp = map(lambda x: (x - np.mean(col))/np.std(col,ddof=1), col)
        data_transpose.append(tmp)
    for col in columns(data_transpose): normalised_data.append(col)  

    return normalised_data

def g(x):
     return 1.0/(1.0 + np.exp(-x))

def grad(theta,X,Y,l):
    m = len(X)
    th = np.array([theta]).transpose()
    grad = (X.transpose().dot(g(X.dot(th)) - Y)/m).transpose()[0]
    grad[1:len(theta)] = grad[1:len(theta)] + l/m * theta[1:len(theta)]
    return grad

def J(theta, X, Y,l):
    m = len(X)
    th = theta.transpose()
    t1 = 1.0/m * (-(1-Y).transpose().dot(np.log(1-g(X.dot(th)))) - Y.transpose().dot(np.log(g(X.dot(th)))))
    reg = l/(2*m) * theta[1:len(theta)].dot(theta[1:len(theta)])
    return t1 + reg

def create_features(data, deg):
    no_of_fts = len(row)
    data_with_fts = []
    for r in data:
        data_with_fts.append([(r[0]**(i-j))*(r[1]**j) for i in range(0,deg+1) for j in range(0,i+1)])
    return data_with_fts

def check_gradient(nnet, l):
    epsilon = 0.0000001

    test_data = np.random.rand(5,nnet.sizes[0])
    test_out = np.random.rand(5,nnet.sizes[nnet.no_of_layers - 1])

    num_grad = []
    for layer_weight in nnet.weights:
        num_grad.append(np.zeros(layer_weight.shape))
        for j, k in np.ndindex(layer_weight.shape):
            layer_weight[j][k] = layer_weight[j][k] + epsilon
            Jplus = nnet.cost(test_data, test_out, l)
            layer_weight[j][k] = layer_weight[j][k] - 2*epsilon
            Jmin = nnet.cost(test_data, test_out, l)
            layer_weight[j][k] = layer_weight[j][k] + epsilon
            num_grad[-1][j][k] = (Jplus - Jmin)/(2*epsilon)
    return num_grad, nnet.grad(test_data, test_out, l)

def cost_wrapper(theta, X, Y, l, nnet):
    #print "T: ", theta[1:30]
    nnet.set_weights(theta)
    ret = nnet.cost(X,Y,l)
    print ret
    return ret

def grad_wrapper(theta, X, Y, l, nnet):
    #print "T: ", theta[1:30]
    nnet.set_weights(theta)
    ret = nnet.grad(X,Y,l)
    ret2 = np.hstack([layer.flatten() for layer in ret])
    return ret2
 
class NNet():
    def __init__(self, layer_sizes, weight_range):
        # Not trying to be slick here, so don't take any short cuts
        # In particular lets define variables a and z that map directly
        # to the variables in the neural network.

        self.sizes = layer_sizes
        self.no_of_layers = len(layer_sizes)
        self.a = [None] * self.no_of_layers
        self.z = [None] * self.no_of_layers
        self.weights = []

        #Create a weight matrix for each layer which we initialise
        #by choosing each entry independently and uniformly from [-weight_range,weight_range]

        i = 1
        while i < self.no_of_layers:
        # Create a random matrix with iid uniform entries of size layer_sizes[i-1]+1 by layer_sizes[i].
        # The addition of one to layer_sizes[i-1]+1 is to account for the bias unit in each layer.
            rnd_matrix = 2.0 * weight_range * (np.random.rand(layer_sizes[i],layer_sizes[i-1]+1) - 0.5)

            self.weights.append(rnd_matrix)
            i = i + 1

    #Do the forward pass through the network. At the end of the loop a[] and z[]
    #will be filled with activations of each neuron.
    #We expect the data to be a numpy array in which each row is an input vector (training example)

    def forward_pass(self, data):
        norows, nocols = data.shape
        #It is easier to represent the activations in each layer as column vectors
        #The activation in layer i for the jth training example is the jth column of z[i] 
        #Note that due to this convention we must transpose the initial data matrix.
        #We also insert a row of ones in the first row of z[0] to act as the bias units.

        self.z[0] = np.vstack((np.ones((1,norows)), data.T))

        for i in range(1, self.no_of_layers):
            self.a[i] = np.dot(self.weights[i-1], self.z[i-1])
            self.z[i] = self.g(self.a[i])
            if i + 1 < self.no_of_layers: self.z[i] = np.vstack((np.ones((1,self.z[i].shape[1])), self.z[i]))

    #This function computes the cost function for a set of training examples.
    #This function expects each row of X to be a training example
    #and each row of Y to be the classification label. The classification
    #label in each row of Y is assumed to be a
    #binary vector in which the class label is determined by
    #which column contains a one.
    #l is a real valued parameter specifying the regularisation strength

    def cost(self, X, Y, l):
        m = X.shape[0]
        self.forward_pass(X)
        final_layer = self.no_of_layers - 1
        #Split up terms in the cost function for better readability
        t1 = -np.trace(np.dot(np.log(self.z[final_layer]), Y))
        t2 = -np.trace(np.dot(np.log(1 - self.z[final_layer]), 1 - Y))
        reg = 0

        #Add regularisation term. Use the L2 norm from linalg.
        #weights[i-1][:,1:] chops off the first column of weights, which are the biases
        for i in range(1, self.no_of_layers):
            reg = reg + np.linalg.norm(self.weights[i-1][:,1:])**2

        return 1.0/m * (t1 + t2 + l*reg/2.0)
    
    #Set the weights using an unrolled vector
    def set_weights(self, weight_vector):
        j = 0
#        print "b: ", self.weights[0][0,1:30]
        for i in range(0, len(self.sizes) - 1):
            layer_size = self.sizes[i+1]*(self.sizes[i]+1)
#            print np.array(weight_vector[j:j + layer_size])[1:30]
            self.weights[i] = np.array(weight_vector[j:j + layer_size])
            self.weights[i].shape = (self.sizes[i+1], self.sizes[i]+1)
            j = j + layer_size
#        print "a: ", self.weights[0][0,1:30]


    #Use back propagation to compute gradient.
    #This function expects each row of X to be a training example
    #and each row of Y to be the classification label. The classification
    #label in each row of Y is assumed to be a
    #binary vector in which the class label is determined by
    #which column contains a one.

    def grad(self, X, Y, l):
        self.forward_pass(X)

        m = X.shape[0]
        final_layer = self.no_of_layers - 1
        delta = [None] * self.no_of_layers
        gradient = []
        delta[final_layer] = self.z[final_layer ] - Y.T
        for i in range(final_layer, 0, -1):
            #Compute gradient using back propagated errors. Note that we matrix multiply
            #in order to sum over the training examples.
            gradient.append(np.dot(delta[i], self.z[i-1].T)/m)
            #Add regularisation term.
            gradient[-1][:,1:] = gradient[-1][:,1:] + l/m*self.weights[i-1][:,1:]

            #Back propagate the errors to the next layer.
            if i > 1 : delta[i-1] = np.dot(self.weights[i-1][:,1:].T, delta[i]) * self.dg(self.a[i-1])
        return gradient[::-1]

    def load_weights(self, weights):
        self.weights = weights

    def train(self, X, Y, l):
        current_grad = self.grad(X,Y,l)
        for i in range(0, self.no_of_layers-1):
            #print current_grad[i][1:2,1:2]
            self.weights[i] = self.weights[i] - 1.1 * current_grad[i]

    def g(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def dg(self, x):
        return self.g(x)*(1.0 - self.g(x))



#### Main ####
data = sio.loadmat('ex3data1.mat')
weights = sio.loadmat('ex3weights.mat')
nnet = NNet([400,25,10],0.12)
#nnet.load_weights([weights['Theta1'], weights['Theta2']])

#Convert data into required form
X = data['X']
Y = np.zeros((X.shape[0], 10))
for row, n in zip(Y,data['y']): row[n-1] = 1

init = np.hstack([layer.flatten() for layer in nnet.weights])
theta = sp.fmin_l_bfgs_b(cost_wrapper, init, fprime = grad_wrapper, args = (X,Y,1,nnet))


'''
for i in range(0,300):
    nnet.train(X,Y,1.0)
    print i,":1000 ",nnet.cost(X,Y,1.0) 
'''

fig=plt.figure()
for k in range(0,25):
    fig.add_subplot(5,5,k+1)
    rand_int = int(4000*rnd.random())
    Xim=[[data['X'][rand_int][j*20 + i] for j in range(0,20)] for i in range(0,20)]
    nnet.forward_pass(np.array([data['X'][rand_int]]))
    print (np.argmax(nnet.z[nnet.no_of_layers - 1]) + 1) % 10,
    if((k+1)%5 == 0): print "\n"
    plt.imshow(Xim)

plt.show()

correct = 0
for example, answer in zip(X,data['y']):
    nnet.forward_pass(np.array([example]))
    pred = (np.argmax(nnet.z[nnet.no_of_layers - 1]) + 1)
    if pred == answer: correct = correct + 1
    print pred, answer

print 'Percentage correct: ', float(correct)/X.shape[0]
