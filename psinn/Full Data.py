#!/usr/bin/env python
# coding: utf-8
# Copyright 2019 Lawrence Livermore National Security, LLC and other PSINN
# developers. See the top-level LICENSE file for more details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import pandas as pd
from scipy import integrate
from scipy.integrate import solve_ivp
import pywt

# for reproducibility
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# This cleaned data set takes the cos of the angular data
# In angular data there is a switch point, we remove this by taking the average of the points on either side
# The mag data is not edited, we will normalize in this notebook
data = pd.read_csv('all_clean_data_10ms.csv', index_col=0)
column_names = list(data.columns.values.tolist()) 
data.head()

# Lets start by normalizing the magnitude data between -1 and 1
# so we can find the integral of the data 
for name in ['c1_mag', 'c2_mag', 'c3_mag', 'l1_mag', 'l2_mag', 'l3_mag']:
    data[name] = data[name] - np.mean(data[name])
    data[name] = data[name] / max(np.abs(data[name]))
    
# The angular data is already between -1 and 1 because we took the cos

# Integrate all the data streams
t = np.linspace(0, data.shape[0]-1, data.shape[0])/100
for name in column_names:
    data[name+'_int'] = integrate.cumtrapz(data[name], t, initial = 0)

data.drop(data.tail(5).index,inplace=True)

def lowpassfilter(signal, thresh = .63, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

# Lets run all angular data through a low pass filter
# for name in ['c1_ang', 'c2_ang', 'c3_ang', 'l1_ang', 'l2_ang', 'l3_ang']:
#     data[name+'_int'] = lowpassfilter(data[name+'_int'], thresh = 500)[0:-1]

#Set up data to feed into the neural network
# batch_size = data.shape[0]
batch_size = 5000
numbatches = 1
n_d = 6
j=0

data_x = np.zeros((numbatches, batch_size, n_d))
data_y = np.zeros((numbatches, batch_size, n_d))

for i in range(numbatches):
    data_x[i,:,:] = data.iloc[i*batch_size:(i+1)*batch_size,j:n_d+j].values
    data_y[i,:,:] = data.iloc[i*batch_size:(i+1)*batch_size,12+j:12+n_d+j].values

# set up neural model for vector field
reset_graph()

# set parameters
n_L = batch_size

# want to incorporate M trajectories, each of length L, and each in R^d;
# the vector field will be the model Y = f(X)
X = tf.placeholder(tf.float32, [None, n_L, n_d], name="X")
Y = tf.placeholder(tf.float32, [None, n_L, n_d], name="Y")

# set up neural model for vector field

# set number of shape functions
n_s = 2

# set depth of network (number of hidden layers)
depth = 1

# set number of neurons in each layer
# this should be a list of integers, with length = depth
numneurons = [4]

#poly initialization
initvec1 = tf.expand_dims(tf.concat([tf.ones([1]), -tf.ones([1]), tf.zeros([numneurons[0]-2])], axis = 0), 1)
initvec2 = tf.expand_dims(tf.concat([tf.ones([1]), tf.ones([1]), tf.zeros([numneurons[0]-2])], axis = 0), 1)

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
            weights[i] = [tf.Variable(tf.transpose(initvec1), name=wname, dtype=tf.float32)]
            biases[i] = [tf.Variable(tf.zeros([1, nout]), name=bname, dtype=tf.float32)]
        else:
            if (i==0):
                biases[i].append(tf.Variable(tf.ones([1, nout]), name=bname, dtype=tf.float32))
            else:
                biases[i].append(tf.Variable(tf.zeros([1, nout]), name=bname, dtype=tf.float32))
            if (i % 2 == 0):
                weights[i].append(tf.Variable(initvec2, name=wname, dtype=tf.float32))
            else:
                weights[i].append(tf.Variable(initvec1, name=wname, dtype=tf.float32))

# define the shape functions
def psi(si, x):
    psiout = x
    for j in range(depth):
        if si >= 1:
            temp = tf.math.pow(tf.nn.relu(tf.tensordot(psiout, weights[si][j], axes=[2, 0]) + biases[si][j]), tf.constant(si, dtype=tf.float32))
        else:
            temp = tf.nn.relu(tf.tensordot(0*psiout, weights[si][j], axes=[2, 0]) + biases[si][j])

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
training_op = optimizer.minimize(mseloss, var_list = beta)

# typical TF initialization
init = tf.global_variables_initializer()

# set up TF to save model to disk
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    # temp = sess.run(multishapes, feed_dict = {X:X_batch})
    maxsteps = 20000
    for i in range(maxsteps):
        sess.run(training_op, feed_dict = {X:data_y, Y:data_x})
        if (i % 100) == 0:
            print(i, "Training MSE:", 
                  mseloss.eval(feed_dict = {X:data_y, Y:data_x}))
    
    saver.save(sess, "./trained_neuralf")
    betaout, weightsout, biasesout = sess.run([beta, weights, biases], feed_dict = {X:data_y, Y:data_x})

# set up TF to save model to disk
# saver = tf.train.Saver()

# with tf.Session() as sess:
#     saver.restore(sess, "./trained_neuralf")
#     x = tf.linspace(-1.0, 1.0, 201)
#     xnum = sess.run(x)
#     x = tf.expand_dims(x, axis=0)
#     x = tf.expand_dims(x, axis=2)
#     psigraphs = []
#     for i in range(n_s):
#         psigraphs.append(np.squeeze(sess.run(psi(i, x))))
    
#     graphtenlist = []
#     for ind in inds:
#         graphtemp = psigraphs[ind[0]]
#         for j in range(1, n_d):
#             graphtemp = np.outer(graphtemp, psigraphs[ind[j]])
        
#         graphtenlist.append(graphtemp)

#     betaout, weightsout, biasesout = sess.run([beta, weights, biases], feed_dict = {X:data_x, Y:data_y})

# ReLU fucntion for NumPy purposes
def reluNP(x):
    y = np.zeros(x.shape)
    y[x>=0] = x[x>=0]
    return y

#one dimensional shape functions for NumPy purposes
def psiNP(si,x):
    psiout = x
    for j in range(depth):
        if si >= 1:
            psiout = np.power(reluNP(np.dot(psiout, weightsout[si][j]) + biasesout[si][j]), si)
        else:
            psiout = reluNP(np.dot(psiout*0, weightsout[si][j]) + biasesout[si][j])

    psiout = np.dot(psiout, weightsout[si][depth]) + biasesout[si][depth]
    return psiout

def nvf(t,x):
    # split unput by dimesion along axis 1
    y = x.T
    splitNP = np.split(y, n_d, axis=1)
    
    # now for each of these tensors, we pass them through shape functions
    alloutsNP = [[]]*n_d
    for i in range(n_d):
        for j in range(n_s):
            if (j==0):
                alloutsNP[i] = [psiNP(j, splitNP[i])]
            else:
                alloutsNP[i].append(psiNP(j, splitNP[i]))
                
    tenlistNP = []
    for ind in inds:
        temp = alloutsNP[0][ind[0]]
        for j in range(1, n_d):
            temp = np.multiply(temp, alloutsNP[j][ind[j]])
    
        tenlistNP.append(np.squeeze(temp, axis=1))

    # multidimensional shape functions of inputs
    n_ms = len(inds)
    multishapesNP = np.stack(tenlistNP,axis=1)

    # neural vector field!
    nvfNP = np.matmul(multishapesNP, betaout)
    return nvfNP.T

romtraj = np.zeros((1, batch_size, n_d))
ft = (batch_size-1)/100
tvec = np.linspace(0, ft, batch_size)


# ic = data_x[[0],0,:].T
ic = data_y[[0],0,:].T
yic = np.squeeze(ic)
sol = solve_ivp(fun=nvf, t_span=[0, ft], y0 = yic, t_eval=tvec, rtol=1e-8, atol=1e-8, vectorized = True)
romtraj[0,:,:] = sol.y.T

f, axarr = plt.subplots(3, 4, figsize=(20, 10))
axarr[0,0].axis('off')
for i in range(1, n_d+1):
    plt.subplot(3, 4, i)
    plt.plot(romtraj[0,:,i-1],'o', color = 'r')
    plt.plot(data_y[0,:,i-1], color = 'k')
