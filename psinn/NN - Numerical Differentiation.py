#!/usr/bin/env python
# coding: utf-8
# Copyright 2019 Lawrence Livermore National Security, LLC and other PSINN
# developers. See the top-level LICENSE file for more details.
#
# SPDX-License-Identifier: MIT


import numpy as np
import tensorflow as tf


# for reproducibility
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


file0 = open("Data/Current.csv", "r")
data_c1= np.loadtxt(file0)

file1 = open("Data/Voltage.csv", "r")
data_l1 = np.loadtxt(file1)


data = np.stack([data_c1, data_l1]).T


# # set parameters for the data
# batch_size = 30
# numbatches = 100
# h = 1
 
# data_temp = np.zeros((numbatches, batch_size, 2))
# data_y = np.zeros((numbatches, batch_size-1, 2)) 

# for i in range(numbatches):
#     data_temp[i,:,0] = data[i*batch_size:(i+1)*batch_size, 0]
#     data_temp[i,:,1] = data[i*batch_size:(i+1)*batch_size, 1]

# set parameters for the data
batch_size = 30
numbatches = 10
h=1
t = np.linspace(1, batch_size*10, batch_size*10)
 

data_temp = np.zeros((numbatches, batch_size, 2))
data_xtemp = np.zeros((numbatches, batch_size*10, 2))
data_y = np.zeros((numbatches, batch_size*10-1, 2)) 

for i in range(numbatches):
    data_temp[i,:,0] = data[i*batch_size:(i+1)*batch_size, 0]
    data_temp[i,:,1] = data[i*batch_size:(i+1)*batch_size, 1]
    
    x = np.arange(30)
    cs0 = CubicSpline(x, data_temp[i,:,0])
    cs1 = CubicSpline(x, data_temp[i,:,1])
    
    xs = np.arange(0, batch_size, .1)
    
    data_xtemp[i,:,0] = cs0(xs)
    data_xtemp[i,:,1] = cs1(xs)
    
for i in range(numbatches):
    for j in range(batch_size*10-1):
        data_y[i,j] = (data_xtemp[i,j+1] - data_xtemp[i,j])/h
        
data_x = np.zeros((numbatches, batch_size*10-1, 2))

for i in range(numbatches):
    for j in range(batch_size*10-1):
        data_x[i,j] = data_xtemp[i,j] 

batch_size = batch_size*10-1


# THIS BLOCK IS FOR NUMERICAL INTEGRATION
# REMEBER THAT IN THE FEED DICT YOU HAVE TO SWITCH
# DATA_X and DATA_Y


# # set parameters for the data
# batch_size = 30
# numbatches = 10
# t = np.linspace(1, batch_size*10, batch_size*10)
 

# data_temp = np.zeros((numbatches, batch_size, 2))
# data_x = np.zeros((numbatches, batch_size*10, 2))
# data_y = np.zeros((numbatches, batch_size*10, 2)) 

# for i in range(numbatches):
#     data_temp[i,:,0] = deriv[i*batch_size:(i+1)*batch_size, 0]
#     data_temp[i,:,1] = deriv[i*batch_size:(i+1)*batch_size, 1]
    
#     x = np.arange(30)
#     cs0 = CubicSpline(x, data_temp[i,:,0])
#     cs1 = CubicSpline(x, data_temp[i,:,1])
    
#     xs = np.arange(0, batch_size, .1)
    
#     data_x[i,:,0] = cs0(xs)
#     data_x[i,:,1] = cs1(xs)
    
    
# for i in range(numbatches):
#     data_y[i,:,0] = integrate.cumtrapz(data_x[i,:,0], t, initial = 0)
#     data_y[i,:,1] = integrate.cumtrapz(data_x[i,:,1], t, initial = 0)

# batch_size = batch_size*10


# set up neural model for vector field
reset_graph()

# set parameters
n_L = batch_size
n_d = 2

# want to incorporate M trajectories, each of length L, and each in R^d;
# the vector field will be the model Y = f(X)
X = tf.placeholder(tf.float32, [None, n_L, n_d], name="X")
Y = tf.placeholder(tf.float32, [None, n_L, n_d], name="Y")


# set up neural model for vector field

# set number of shape functions
n_s = 2

# set depth of network (number of hidden layers)
depth = 2

# set number of neurons in each layer
# this should be a list of integers, with length = depth
numneurons = [16,16]

# create all the shape functions
weights = [[]]*n_s
biases = [[]]*n_s
for i in range(n_s):
    for j in range(depth + 1):
        if (j==0):
            ninp = 1
        else:
            ninp = numneurons[j-1]
        if (j==depth):
            nout = 1
        else:
            nout = numneurons[j]
        wname = "weight_" + str(i) + "_" + str(j)
        bname = "bias_" + str(i) + "_" + str(j)

        if (j==0):
            weights[i] = [tf.Variable(tf.random.normal([ninp, nout]), name=wname)]
            biases[i] = [tf.Variable(tf.zeros([1, nout]), name=bname)]
        else:
            weights[i].append(tf.Variable(tf.random.normal([ninp, nout]), name=wname))
            biases[i].append(tf.Variable(tf.zeros([1, nout]), name=bname))


# define the shape functions
def psi(si, x):
    temp = x
    for j in range(depth):
        temp = tf.nn.tanh(tf.tensordot(temp, weights[si][j], axes=[2, 0]) + biases[si][j])
    
    temp = tf.tensordot(temp, weights[si][depth], axes=[2, 0]) + biases[si][depth]
    return temp


# split input by dimension
# splitX now contains a list of n_d tensors each of the form M x L x 1
splitX = tf.split(X, n_d, axis=2, name='split')

# now for each of these tensors, we pass them through all the shape functions
allouts = [[]]*n_d
for i in range(n_d):
    for j in range(n_s):
        if (j==0):
            allouts[i] = [psi(j, splitX[i])]
        else:
            allouts[i].append(psi(j, splitX[i]))


# now that we have all the 1d shape functions applied to each dimension,
# we form multidim shape functions up to a particular shell

# numpy tensor product matrix
digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# functions to create all multiindices with summed degree < maxdeg
maxdeg = n_s

def padder(x):
    if len(x) < n_d:
        return '0' * (n_d - len(x)) + x
    else:
        return x

def int2str(x):
    if x < 0:
        return "-" + int2str(-x)
    return ("" if x < maxdeg else int2str(x // maxdeg))            + digits[x % maxdeg]

def padint2str(x):
    return padder(int2str(x))

# index mapping
def index_mapping():
    index = 0
    index_map = {}
    allpos = list(map(padint2str, range(maxdeg ** n_d)))
    for d in range(maxdeg):
        for s in allpos:
            y = list(map(int, s))[::-1]
            if sum(y) == d:
                index_map[tuple(y)] = index
                index += 1

    return index_map

# all multiindices
indmap = index_mapping()
inds = list(indmap.keys())

# make list of tensors

# future idea: let the 0-th shape function correspond to a constant
# so in fact we have (n_s + 1) total shape functions, but only n_s trainable ones

tenlist = []
for ind in inds:
    temp = allouts[0][ind[0]]
    for j in range(1, n_d):
        temp = tf.multiply(temp, allouts[j][ind[j]])
    
    tenlist.append(tf.squeeze(temp, axis=2))

# multidimensional shape functions of inputs
n_ms = len(inds)
multishapes = tf.stack(tenlist,axis=2)

# coefficient matrix
beta = tf.Variable(tf.random.normal([n_ms, n_d]))

# neural vector field!
neuralf = tf.tensordot(multishapes, beta, axes=[2, 0])


# loss function and optimization setup
mseloss = tf.reduce_mean(tf.square(neuralf - Y))

# optimization
optimizer = tf.train.AdamOptimizer(learning_rate = .001)
training_op = optimizer.minimize(mseloss)


# typical TF initialization
init = tf.global_variables_initializer()

# set up TF to save model to disk
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    # temp = sess.run(multishapes, feed_dict = {X:X_batch})
    maxsteps = 5000
    for i in range(maxsteps):
        sess.run(training_op, feed_dict = {X:data_x, Y:data_y})
        if (i % 100) == 0:
            print(i, "Training MSE:", 
                  mseloss.eval(feed_dict = {X:data_x, Y:data_y}))
    
    saver.save(sess, "./trained_neuralf")


# set up TF to save model to disk
saver = tf.train.Saver()


with tf.Session() as sess:
    saver.restore(sess, "./trained_neuralf")
    x = tf.linspace(-1.0, 1.0, 201)
    xnum = sess.run(x)
    x = tf.expand_dims(x, axis=0)
    x = tf.expand_dims(x, axis=2)
    psigraphs = []
    for i in range(n_s):
        psigraphs.append(np.squeeze(sess.run(psi(i, x))))
    
    graphtenlist = []
    for ind in inds:
        graphtemp = psigraphs[ind[0]]
        for j in range(1, n_d):
            graphtemp = np.outer(graphtemp, psigraphs[ind[j]])
        
        graphtenlist.append(graphtemp)

    betanum = sess.run(beta)


plt.plot(xnum,psigraphs[0])
plt.show()


plt.plot(xnum,psigraphs[1])
plt.show()


graphf = np.tensordot(np.stack(graphtenlist, axis=2), betanum, axes=[2, 0])


print(graphf.shape)
XX, YY = np.meshgrid(xnum, xnum)
XX = XX
YY = YY.T


plt.contour(XX, YY, graphf[:,:,0])
plt.show()


plt.contour(XX, YY, graphf[:,:,1])
plt.show()




