# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 23:00:42 2025

@author: Trevor
"""



import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib.ticker import MaxNLocator

error = np.array([1e0,1e-1,1e-2,1e-3,1e-4,1e-5])


normal = np.array([2.07E-01, 1.76E-02, 1.05E-03, 7.64E-05, 6.05E-06, 3.71E-06])
sobolev = np.array([5.38E-01, 5.43E-02, 5.38E-03, 6.05E-04, 1.29E-04, 8.99E-05])
runge = np.array([2.30E-02, 3.44E-03, 3.30E-04, 3.92E-05, 3.96E-06, 8.37E-07])
SG = np.array([1.61E-01, 1.64E-02, 1.96E-03, 4.09E-04, 3.81E-04, 3.67E-04])
TVR = np.array([2.20E-02, 5.96E-03, 1.38E-03, 1.90E-04, 1.80E-05, 2.02E-06])

normal2 = np.array([177.95137, 17.28782267, 0.4513657, 0.027376369, 0.011603008, 0.011313863])
sobolev2 = np.array([68.03599333, 7.2508051, 0.691760957, 0.061466871, 0.012609231, 0.010094903])
runge2 = np.array([0.23236054, 0.13035420, 0.021863451, 0.012784048, 0.011009187, 0.010852236])
SG2 = np.array([223.0278933, 22.31922767, 2.24814, 0.240960367, 0.040220007, 0.020138899])
TVR2 = np.array([0.154804645, 0.136574035, 0.11647207, 0.021807778, 0.014461209, 0.012139427])

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))
plt.loglog(error, normal, marker='o', linestyle='-', color='darkblue', label='Standard Training')
plt.loglog(error, sobolev, marker='o', linestyle='-', color='green', label='Sobolev Training')
plt.loglog(error, runge, marker='o', linestyle='-', color='red', label='Runge-Kutta Training')
plt.loglog(error, SG, marker='o', linestyle='-', color='orange', label='S-G Filter')
plt.loglog(error, TVR, marker='o', linestyle='-', color='cyan', label='TVR')
plt.xlabel(r'$\sigma^2$', fontsize=30)
plt.ylabel(r'MSE', fontsize=30)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=20)  # 调整 y 轴刻度字体大小
plt.tight_layout()

plt.savefig('function.pdf', format='pdf')
plt.show()


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))
plt.loglog(error, normal2, marker='o', linestyle='-', color='darkblue', label='Standard Training')
plt.loglog(error, sobolev2, marker='o', linestyle='-', color='green', label='Sobolev Training')
plt.loglog(error, runge2, marker='o', linestyle='-', color='red', label='Runge-Kutta Training')
plt.loglog(error, SG2, marker='o', linestyle='-', color='orange', label='S-G Filter')
plt.loglog(error, TVR2, marker='o', linestyle='-', color='cyan', label='TVR')
plt.xlabel(r'$\sigma^2$', fontsize=30)
plt.ylabel(r'MSE', fontsize=30)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=20)  # 调整 y 轴刻度字体大小
plt.tight_layout()

plt.savefig('derivative.pdf', format='pdf')
plt.show()