import os
import numpy as np
import sys
import matplotlib.pyplot as plt

lambd = 0.5
mu = 0.3

length = 100000

# First method

race = np.zeros(shape=[length,2])
race[:,0] = np.random.exponential(scale=(1/mu),size=length)
race[:,1] = np.random.exponential(scale=(1/lambd),size=length)

holding_times = np.min(race,axis=1)# Values

tempHist = np.histogram(holding_times,bins=20, range=[0,5])

plt.bar(tempHist[1][1::]-0.2, tempHist[0]/length, width=0.05, label='Method1')

# Second method

holding_times2 = np.random.exponential(scale=(1/(mu + lambd)), size=length)

tempHist2 = np.histogram(holding_times2, bins=20, range=[0,5])

plt.bar(tempHist2[1][1::]-0.15, tempHist2[0]/length, width=0.05, label='Method2')

# Third method

strips = np.concatenate((np.cumsum(race[:,0]), np.cumsum(race[:,1])))
strips = np.sort(strips)
holding_times3 = strips[1::] - strips[0:-1]
holding_times3 = holding_times3[0:length]

tempHist3 = np.histogram(holding_times3, bins=20, range=[0,5])

plt.bar(tempHist3[1][1::]-0.1, tempHist3[0]/length, width=0.05, label='Method3')

plt.legend(loc="upper right")