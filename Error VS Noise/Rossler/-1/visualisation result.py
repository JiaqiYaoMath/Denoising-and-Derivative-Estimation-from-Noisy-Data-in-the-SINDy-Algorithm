# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 10:48:43 2024

@author: Jiaqi Yao
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib.ticker import MaxNLocator

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
############################################################################
print(os.path.dirname(os.path.abspath("")))
sys.path.append(os.path.dirname(os.path.abspath("")))

from Functions.utiles import initial_cond_generation, set_all_seeds
from Functions.data_gen import generator
from Functions.data_set import train_test_way2
from Functions.network1_library import Siren, NNTanh
from Functions.calculation import  _combinations , Derivative, Numerical_Derivative, Second_Derivative, add_white_noise
from Functions.training import train, pre_train
from Functions.RK4_training import normal_train, Sobolev_train, RK4_train

set_all_seeds(777)
##########################################################################

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(" We are using a " + device + " for the simulation")

############################################################################
class ParametersSetting:
    """
    """

    t_min = 0
    t_max = 21
    sample_density = 0.05
    split_value = 20/21
    select_prob = 1.0
    
    ts = torch.arange(t_min, t_max, sample_density)
    RK_timestep = ts[1] - ts[0]
    time_deriv_coef = 2 / (t_max - t_min)
    
    num_indp_var = 3
    system = 'rosser'
    fun_scaling_factor = 0.1
    initial_condition = [-7.5,2.5,1e-8]

    burnin_iterations = 3000
    device = device

    add_noise = True
    noise_level = 1e-1
    
    def ts_generator(self):
        return self.ts


param_set = ParametersSetting()
index = int(param_set.t_max/param_set.sample_density*param_set.split_value)
#############################################################################3
y = generator(param_set)

(t_scaled, u_scaled) = y.run()

a = 0.2
b = 0.2
c = 5.7
s = param_set.fun_scaling_factor

deri_data = torch.ones_like(u_scaled, dtype= torch.float)
deri_data[:,0] = -u_scaled[:,1] - u_scaled[:,2]
deri_data[:,1] = u_scaled[:,0] + a*u_scaled[:,1]
deri_data[:,2] = b + u_scaled[:,2]*(u_scaled[:,0]/s - c)

sec_data = Numerical_Derivative(deri_data,param_set)
train_sec = sec_data[:index,:]
test_sec = sec_data[index:,:]

if param_set.add_noise:
    u_scaled2, deri_data2 = add_white_noise(u_scaled, deri_data, param_set)
    
train_data, test_data = train_test_way2(
    t_scaled.float(),
    u_scaled.float(),
    deri_data.float(),
    param_set
)

train_data2, test_data2 = train_test_way2(
    t_scaled.float(),
    u_scaled2.float(),
    deri_data2.float(),
    param_set
)
#############################################################################3

network1 = torch.load('model/sobolev.pth')
network2 = torch.load('model/runge-kutta.pth')
network3 = torch.load('model/normal.pth')

filter_fun_train = torch.load("model/filter_fun_train.pt")
filter_deri_train = torch.load("model/filter_deri_train.pt")
filter_fun_test = torch.load("model/filter_fun_test.pt")
filter_deri_test = torch.load("model/filter_deri_test.pt")

train_time, train_output, train_deri = train_data.giveback()
test_time, test_output, test_deri = test_data.giveback()

train_time2, train_output2, train_deri2 = train_data2.giveback()
test_time2, test_output2, test_deri2 = test_data2.giveback()
###########################################################################

with torch.no_grad():
    prediction_train1 = network1(train_time)
    prediction_test1  = network1(test_time)
    
    prediction_train2 = network2(train_time)
    prediction_test2  = network2(test_time)
    
    prediction_train3 = network3(train_time)
    prediction_test3  = network3(test_time)
    
 
deri_train1 = Derivative(network1, train_time, param_set).clone().detach()
deri_test1  = Derivative(network1, test_time, param_set).clone().detach()

deri_train2 = Derivative(network2, train_time, param_set).clone().detach()
deri_test2  = Derivative(network2, test_time, param_set).clone().detach()

deri_train3 = Derivative(network3, train_time, param_set).clone().detach()
deri_test3  = Derivative(network3, test_time, param_set).clone().detach()
# ###################################################################################
sec_train1 = Second_Derivative(network1, train_time, param_set).clone().detach()
sec_test1 = Second_Derivative(network1, test_time, param_set).clone().detach()

sec_train2 = Second_Derivative(network2, train_time, param_set).clone().detach()
sec_test2 = Second_Derivative(network2, test_time, param_set).clone().detach()

sec_train3 = Second_Derivative(network3, train_time, param_set).clone().detach()
sec_test3 = Second_Derivative(network3, test_time, param_set).clone().detach()

# ###################################################################################
#################################################################################################
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 5, figsize=(24, 12))


axes[0, 0].plot(train_output[:,0], train_output[:,1], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
axes[0, 0].scatter(train_output2[:,0], train_output2[:,1])
axes[0, 0].set_ylabel(r'Function', fontsize=34)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[0, 0].set_title('Clean Data', fontsize=34)
axes[0, 0].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[1, 0].plot(train_deri[:,0], train_deri[:,1], linewidth=2, color = 'red',linestyle='--', marker='o', label = r'$\dot{x}_1$')
axes[1, 0].set_ylabel(r'Derivative', fontsize=34)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[1, 0].tick_params(axis='both', labelsize=24)
axes[1, 0].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[1, 0].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[0, 1].plot(prediction_train3[:,0], prediction_train3[:,1], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[0, 1].set_title('Normal', fontsize=34)
axes[0, 1].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[0, 1].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[0, 1].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[1, 1].plot(deri_train3[:,0], deri_train3[:,1], linewidth=2, color = 'red',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[1, 1].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[1, 1].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[1, 1].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[0, 2].plot(prediction_train1[:,0], prediction_train1[:,1], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[0, 2].set_title('Sobolev', fontsize=34)
axes[0, 2].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[0, 2].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[0, 2].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[1, 2].plot(deri_train1[:,0], deri_train1[:,1], linewidth=2, color = 'red',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[1, 2].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[1, 2].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[1, 2].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[0, 3].plot(prediction_train2[:,0], prediction_train2[:,1], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[0, 3].set_title('Runge-Kutta', fontsize=34)
axes[0, 3].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[0, 3].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[0, 3].yaxis.set_major_locator(MaxNLocator(nbins=3))


axes[1, 3].plot(deri_train2[:,0], deri_train2[:,1], linewidth=2, color = 'red',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[1, 3].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[1, 3].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[1, 3].yaxis.set_major_locator(MaxNLocator(nbins=3))


axes[0, 4].plot(filter_fun_train[:,0], filter_fun_train[:,1], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[0, 4].set_title('S-G Filter', fontsize=34)
axes[0, 4].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[0, 4].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[0, 4].yaxis.set_major_locator(MaxNLocator(nbins=3))


axes[1, 4].plot(filter_deri_train[:,0], filter_deri_train[:,1], linewidth=2, color = 'red',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[1, 4].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[1, 4].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[1, 4].yaxis.set_major_locator(MaxNLocator(nbins=3))


plt.tight_layout()


plt.savefig("visualize2.pdf", format="pdf", bbox_inches="tight")
plt.show()

# ###################################################################################
#################################################################################################
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 5, figsize=(24, 12), subplot_kw={'projection': '3d'})


axes[0, 0].plot(train_output[:,0], train_output[:,1], train_output[:,2], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
axes[0, 0].scatter(train_output2[:,0], train_output2[:,1], train_output2[:,2], marker='x', s = 150, color = 'green')
axes[0, 0].set_zlabel(r'Function', fontsize=34)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[0, 0].set_title('Clean Data', fontsize=34)
axes[0, 0].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[1, 0].plot(train_deri[:,0], train_deri[:,1], train_deri[:,2], linewidth=2, color = 'red',linestyle='--', marker='o', label = r'$\dot{x}_1$')
axes[1, 0].scatter(train_deri2[:,0], train_deri2[:,1], train_deri2[:,2], marker='x', s = 50, color = 'green')
#axes[1, 0].set_ylabel(r'Derivative', fontsize=34)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[1, 0].tick_params(axis='both', labelsize=24)
axes[1, 0].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[1, 0].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[0, 1].plot(prediction_train3[:,0], prediction_train3[:,1], prediction_train3[:,2], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[0, 1].set_title('Normal', fontsize=34)
axes[0, 1].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[0, 1].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[0, 1].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[1, 1].plot(deri_train3[:,0], deri_train3[:,1], deri_train3[:,2], linewidth=2, color = 'red',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[1, 1].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[1, 1].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[1, 1].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[0, 2].plot(prediction_train1[:,0], prediction_train1[:,1], prediction_train1[:,2], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[0, 2].set_title('Sobolev', fontsize=34)
axes[0, 2].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[0, 2].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[0, 2].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[1, 2].plot(deri_train1[:,0], deri_train1[:,1], deri_train1[:,2], linewidth=2, color = 'red',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[1, 2].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[1, 2].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[1, 2].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[0, 3].plot(prediction_train2[:,0], prediction_train2[:,1], prediction_train2[:,2], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[0, 3].set_title('Runge-Kutta', fontsize=34)
axes[0, 3].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[0, 3].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[0, 3].yaxis.set_major_locator(MaxNLocator(nbins=3))


axes[1, 3].plot(deri_train2[:,0], deri_train2[:,1], deri_train2[:,2], linewidth=2, color = 'red',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[1, 3].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[1, 3].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[1, 3].yaxis.set_major_locator(MaxNLocator(nbins=3))


axes[0, 4].plot(filter_fun_train[:,0], filter_fun_train[:,1], filter_fun_train[:,2], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[0, 4].set_title('S-G Filter', fontsize=34)
axes[0, 4].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[0, 4].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[0, 4].yaxis.set_major_locator(MaxNLocator(nbins=3))


axes[1, 4].plot(filter_deri_train[:,0], filter_deri_train[:,1], filter_deri_train[:,2], linewidth=2, color = 'red',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[1, 4].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[1, 4].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[1, 4].yaxis.set_major_locator(MaxNLocator(nbins=3))

fig.text(-0.01, 0.75, 'Function', fontsize=34, rotation=90, va='center', ha='center')
fig.text(-0.01, 0.30, 'Derivative', fontsize=34, rotation=90, va='center', ha='center')

plt.tight_layout()


plt.savefig("visualize3.pdf", format="pdf", bbox_inches="tight")
plt.show()