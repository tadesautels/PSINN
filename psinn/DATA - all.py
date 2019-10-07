#!/usr/bin/env python
# coding: utf-8
# Copyright 2019 Lawrence Livermore National Security, LLC and other PSINN
# developers. See the top-level LICENSE file for more details.
#
# SPDX-License-Identifier: MIT


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.signal

# Read in raw data from website
# Only save MEAN columns for Voltage and Current
df = pd.read_csv('Data/mass_data/all_data_10ms.csv', index_col=0, usecols=[0,3,7,11,15,19,23,27,31,35,39,43,47])
# df.index = pd.to_datetime(df.index)
df.head()


# Rename columns with shorter names
df.columns = ['c1_ang','c1_mag','c2_ang','c2_mag','c3_ang','c3_mag', 'l1_ang','l1_mag','l2_ang','l2_mag','l3_ang','l3_mag']
df.head()


# convert the degree measurement to radians
# and take the cos (this makes data continuous!)
df.c1_ang = np.cos(np.pi/180*df.c1_ang)
df.c2_ang = np.cos(np.pi/180*df.c2_ang)
df.c3_ang = np.cos(np.pi/180*df.c3_ang)
df.l1_ang = np.cos(np.pi/180*df.l1_ang)
df.l2_ang = np.cos(np.pi/180*df.l2_ang)
df.l3_ang = np.cos(np.pi/180*df.l3_ang)
data = df

plt.plot(data.c1_ang[0:100000], 'o')

# this code removed the wrap around discontinuous points
# by averaging the points on either side
for name in ['c1_ang', 'c2_ang', 'c3_ang', 'l1_ang', 'l2_ang', 'l3_ang']:
    for i in range(data.shape[0]-1):
        if (( data[name].values[i+1] - data[name].values[i] ) > 0.1 ) :
            data[name].values[i] = (data[name].values[i+1] + data[name].values[i-1])/2
        

# save the data to a CVS file to be read into other sources
data.to_csv('Data/mass_data/all_clean_data_10ms.csv')

# make sure that data is easy to read in
test = pd.read_csv('Data/mass_data/all_clean_data_10ms.csv', index_col=0)
test.head()



