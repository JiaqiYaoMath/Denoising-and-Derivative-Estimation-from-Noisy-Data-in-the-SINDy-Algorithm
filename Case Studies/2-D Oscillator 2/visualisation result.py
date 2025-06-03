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
    t_max = 11
    sample_density = 0.1
    split_value = 10/11
    select_prob = 1.0
    
    ts = torch.arange(t_min, t_max, sample_density)
    RK_timestep = ts[1] - ts[0]
    time_deriv_coef = 2 / (t_max - t_min)
    
    num_indp_var = 2
    system = '2_D_Oscilator'
    fun_scaling_factor = 1.0
    initial_condition = [-2.0, 2.0]

    burnin_iterations = 3000
    device = device
    
    add_noise = True
    noise_level = 1e-3
    
    def ts_generator(self):
        return self.ts


param_set = ParametersSetting()
index = int(param_set.t_max/param_set.sample_density*param_set.split_value)
#############################################################################3
y = generator(param_set)

(t_scaled, u_scaled) = y.run()

deri_data = torch.ones_like(u_scaled, dtype= torch.float)
deri_data[:,0] = -0.1*u_scaled[:,0] + 3.0*u_scaled[:,1]
deri_data[:,1] = -3.0*u_scaled[:,0] - 0.1*u_scaled[:,1]

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
#################################################################################################

network1 = torch.load('model/sobolev.pth')
network2 = torch.load('model/runge-kutta.pth')
network3 = torch.load('model/normal.pth')

filter_fun_train = torch.load("model/filter_fun_train.pt")
filter_deri_train = torch.load("model/filter_deri_train.pt")
filter_fun_test = torch.load("model/filter_fun_test.pt")
filter_deri_test = torch.load("model/filter_deri_test.pt")

TVR_fun_train = torch.load("model/TVR_fun_train.pt")
TVR_deri_train = torch.load("model/TVR_deri_train.pt")

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

#############################################################################3
#################################################################################################
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 6, figsize=(24, 12))


axes[0, 0].plot(train_output[:,0], train_output[:,1], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
axes[0, 0].scatter(train_output2[:,0], train_output2[:,1], marker='x', s=200, color='green')
axes[0, 0].set_ylabel(r'State', fontsize=34)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[0, 0].set_title('Data', fontsize=34)
axes[0, 0].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[1, 0].plot(train_deri[:,0], train_deri[:,1], linewidth=2, color = 'red',linestyle='--', marker='o', label = r'$\dot{x}_1$')
axes[1, 0].scatter(train_deri2[:,0], train_deri2[:,1], marker='x', s=200, color='green')
axes[1, 0].set_ylabel(r'Derivative', fontsize=34)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[1, 0].tick_params(axis='both', labelsize=24)
axes[1, 0].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[1, 0].yaxis.set_major_locator(MaxNLocator(nbins=3))

axes[0, 1].plot(prediction_train3[:,0], prediction_train3[:,1], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[0, 1].set_title('Standard', fontsize=34)
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


axes[0, 5].plot(TVR_fun_train[:,0], TVR_fun_train[:,1], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[0, 5].set_title('TVR', fontsize=34)
axes[0, 5].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[0, 5].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[0, 5].yaxis.set_major_locator(MaxNLocator(nbins=3))


axes[1, 5].plot(TVR_deri_train[:,0], TVR_deri_train[:,1], linewidth=2, color = 'red',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.legend(fontsize=24)
axes[1, 5].tick_params(axis='both', labelsize=24) # 调整 x 轴刻度字体大小
axes[1, 5].xaxis.set_major_locator(MaxNLocator(nbins=3))
axes[1, 5].yaxis.set_major_locator(MaxNLocator(nbins=3))

plt.tight_layout()


plt.savefig("2visualize2.pdf", format="pdf", bbox_inches="tight")
plt.show()