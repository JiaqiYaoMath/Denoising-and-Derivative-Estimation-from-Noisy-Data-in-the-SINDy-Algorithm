# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 00:51:29 2025

@author: Jiaqi Yao
"""


import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib.ticker import MaxNLocator

error = np.array([1e0,1e-1,1e-2,1e-3,1e-4,1e-5])

normal = np.array([1.75267035, 0.17528443, 0.017527211, 0.001728152, 0.000240627, 4.69E-05])
sobolev = np.array([0.586814035, 0.072839345, 0.01783365, 0.002311046, 0.000319729, 1.02E-04])
runge = np.array([0.28891423, 0.036707959, 0.005172131, 0.000658577, 8.54824E-05, 1.22E-05])
SG = np.array([0.376360775, 0.065429116, 0.035855482, 0.033378638, 0.033282906, 3.33E-02])
TVR = np.array([0.446418765, 0.110346046, 0.01356475, 0.001736345, 0.000173907, 1.75E-05])

normal2 = np.array([216.047325, 38.007259, 3.9996754, 0.78449613, 0.114956333, 0.063605216])
sobolev2 = np.array([139.8059, 14.233861, 1.3152289, 0.101846285, 0.009705848, 0.000972609])
runge2 = np.array([6.244898, 0.61333916, 0.141217925, 0.041281954, 0.009182191, 0.005596126])
SG2 = np.array([643.53781, 64.73817, 6.8008133, 0.98892918, 0.40200135, 0.34149365])
TVR2 = np.array([12.0555605, 2.1139618, 0.9954819, 0.37025879, 0.161263355,0.047012587
])

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
