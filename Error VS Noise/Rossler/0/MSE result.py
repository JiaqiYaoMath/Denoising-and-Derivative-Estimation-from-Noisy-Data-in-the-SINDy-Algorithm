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

    add_noise = False
    noise_level = 1e0
    
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
    u_scaled, deri_data = add_white_noise(u_scaled, deri_data, param_set)
    
train_data, test_data = train_test_way2(
    t_scaled.float(),
    u_scaled.float(),
    deri_data.float(),
    param_set
)
#################################################################################################
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
plt.plot(param_set.ts, u_scaled[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
plt.plot(param_set.ts, u_scaled[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$x_2$')
plt.plot(param_set.ts, u_scaled[:,2], linewidth=2, color = 'red',linestyle='--', marker='o', label = r'$x_3$')
plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.show()

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
plt.plot(param_set.ts, deri_data[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', label = r'$\dot{x}_1$')
plt.plot(param_set.ts, deri_data[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$\dot{x}_2$')
plt.plot(param_set.ts, deri_data[:,2], linewidth=2, color = 'red',linestyle='--', marker='o', label = r'$\dot{x}_3$')
plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.show()

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
plt.plot(param_set.ts, sec_data[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', label = r'$\ddot{x}_1$')
plt.plot(param_set.ts, sec_data[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$\ddot{x}_2$')
plt.plot(param_set.ts, sec_data[:,2], linewidth=2, color = 'red',linestyle='--', marker='o', label = r'$\ddot{x}_3$')
plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.show()
#############################################################################3

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
# ###################################################################################

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))


plt.plot(param_set.ts[:index], prediction_train3[:,0], linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], prediction_train1[:,0], linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], prediction_train2[:,0], linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], filter_fun_train[:,0], linewidth=2,linestyle='-',color = 'yellow', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], TVR_fun_train[:,0], linewidth=2,linestyle='-',color = 'black', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], train_output[:,0], linewidth=2,color = 'cyan',linestyle='-', label = r'Real Function')

plt.plot(param_set.ts[index:], prediction_test3[:,0], linewidth=2,color = 'darkblue',linestyle='--', label = r'Normal Training')
plt.plot(param_set.ts[index:], prediction_test1[:,0], linewidth=2,linestyle='--', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[index:], prediction_test2[:,0], linewidth=2,linestyle='--',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[index:], test_output[:,0], linewidth=2,color = 'cyan',linestyle='--', label = r'Real Function')

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5x1.pdf', format='pdf')
plt.show()



plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
plt.plot(param_set.ts[:index], torch.abs(train_output[:,0] - prediction_train3[:,0]), linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,0] - prediction_train1[:,0]), linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,0] - prediction_train2[:,0]), linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,0] - filter_fun_train[:,0]), linewidth=2,linestyle='-',color = 'yellow', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,0] - TVR_fun_train[:,0]), linewidth=2,linestyle='-',color = 'black', label = r'Runge-Kutta Training')

print(np.mean((       (train_output[:,0] - prediction_train3[:,0])**2    ).numpy()))
print(np.mean((       (train_output[:,0] - prediction_train1[:,0])**2    ).numpy()))
print(np.mean((       (train_output[:,0] - prediction_train2[:,0])**2    ).numpy()))
print(np.mean((       (train_output[:,0] - filter_fun_train[:,0])**2    ).numpy()))
print(np.mean((       (train_output[:,0] - TVR_fun_train[:,0])**2    ).numpy()))
print()

print(np.mean((       (test_output[:,0] - prediction_test3[:,0])**2    ).numpy()))
print(np.mean((       (test_output[:,0] - prediction_test1[:,0])**2    ).numpy()))
print(np.mean((       (test_output[:,0] - prediction_test2[:,0])**2    ).numpy()))
print()

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5x1diff.pdf', format='pdf')
plt.show()

##########################################################################################################
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))


plt.plot(param_set.ts[:index], prediction_train3[:,1], linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], prediction_train1[:,1], linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], prediction_train2[:,1], linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], filter_fun_train[:,1], linewidth=2,linestyle='-',color = 'yellow', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], TVR_fun_train[:,1], linewidth=2,linestyle='-',color = 'black', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], train_output[:,1], linewidth=2,color = 'cyan',linestyle='-', label = r'Real Function')

plt.plot(param_set.ts[index:], prediction_test3[:,1], linewidth=2,color = 'darkblue',linestyle='--', label = r'Normal Training')
plt.plot(param_set.ts[index:], prediction_test1[:,1], linewidth=2,linestyle='--', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[index:], prediction_test2[:,1], linewidth=2,linestyle='--',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[index:], test_output[:,1], linewidth=2,color = 'cyan',linestyle='--', label = r'Real Function')


plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5x2.pdf', format='pdf')
plt.show()


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

plt.plot(param_set.ts[:index], torch.abs(train_output[:,1] - prediction_train3[:,1]), linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,1] - prediction_train1[:,1]), linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,1] - prediction_train2[:,1]), linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,1] - filter_fun_train[:,1]), linewidth=2,linestyle='-',color = 'yellow', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,1] - TVR_fun_train[:,1]), linewidth=2,linestyle='-',color = 'black', label = r'Runge-Kutta Training')

print(np.mean((       (train_output[:,1] - prediction_train3[:,1])**2    ).numpy()))
print(np.mean((       (train_output[:,1] - prediction_train1[:,1])**2    ).numpy()))
print(np.mean((       (train_output[:,1] - prediction_train2[:,1])**2    ).numpy()))
print(np.mean((       (train_output[:,1] - filter_fun_train[:,1])**2    ).numpy()))
print(np.mean((       (train_output[:,1] - TVR_fun_train[:,1])**2    ).numpy()))
print()

print(np.mean((       (test_output[:,1] - prediction_test3[:,1])**2    ).numpy()))
print(np.mean((       (test_output[:,1] - prediction_test1[:,1])**2    ).numpy()))
print(np.mean((       (test_output[:,1] - prediction_test2[:,1])**2    ).numpy()))
print()

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5x2diff.pdf', format='pdf')
plt.show()

##########################################################################################
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))


plt.plot(param_set.ts[:index], prediction_train3[:,2], linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], prediction_train1[:,2], linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], prediction_train2[:,2], linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], filter_fun_train[:,2], linewidth=2,linestyle='-',color = 'yellow', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], TVR_fun_train[:,2], linewidth=2,linestyle='-',color = 'black', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], train_output[:,2], linewidth=2,color = 'cyan',linestyle='-', label = r'Real Function')

plt.plot(param_set.ts[index:], prediction_test3[:,2], linewidth=2,color = 'darkblue',linestyle='--', label = r'Normal Training')
plt.plot(param_set.ts[index:], prediction_test1[:,2], linewidth=2,linestyle='--', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[index:], prediction_test2[:,2], linewidth=2,linestyle='--',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[index:], test_output[:,2], linewidth=2,color = 'cyan',linestyle='--', label = r'Real Function')

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5x3.pdf', format='pdf')
plt.show()


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
plt.plot(param_set.ts[:index], torch.abs(train_output[:,2] - prediction_train3[:,2]), linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,2] - prediction_train1[:,2]), linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,2] - prediction_train2[:,2]), linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,2] - filter_fun_train[:,2]), linewidth=2,linestyle='-',color = 'yellow', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,2] - TVR_fun_train[:,2]), linewidth=2,linestyle='-',color = 'black', label = r'Runge-Kutta Training')

print(np.mean((       (train_output[:,2] - prediction_train3[:,2])**2    ).numpy()))
print(np.mean((       (train_output[:,2] - prediction_train1[:,2])**2    ).numpy()))
print(np.mean((       (train_output[:,2] - prediction_train2[:,2])**2    ).numpy()))
print(np.mean((       (train_output[:,2] - filter_fun_train[:,2])**2    ).numpy()))
print(np.mean((       (train_output[:,2] - TVR_fun_train[:,2])**2    ).numpy()))
print()

print(np.mean((       (test_output[:,2] - prediction_test3[:,2])**2    ).numpy()))
print(np.mean((       (test_output[:,2] - prediction_test1[:,2])**2    ).numpy()))
print(np.mean((       (test_output[:,2] - prediction_test2[:,2])**2    ).numpy()))
print()

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5x3diff.pdf', format='pdf')
plt.show()

##########################################################################################################
#######################################################################################
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))


plt.plot(param_set.ts[:index], deri_train3[:,0], linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], deri_train1[:,0], linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], deri_train2[:,0], linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], filter_deri_train[:,0], linewidth=2,color = 'yellow',linestyle='-', label = r'Real Function')
plt.plot(param_set.ts[:index], TVR_deri_train[:,0], linewidth=2,color = 'black',linestyle='-', label = r'Real Function')
plt.plot(param_set.ts[:index], train_deri[:,0], linewidth=2,color = 'cyan',linestyle='-', label = r'Real Derivative')

plt.plot(param_set.ts[index:], deri_test3[:,0], linewidth=2,color = 'darkblue',linestyle='--', label = r'Normal Training')
plt.plot(param_set.ts[index:], deri_test1[:,0], linewidth=2,linestyle='--', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[index:], deri_test2[:,0], linewidth=2,linestyle='--',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[index:], test_deri[:,0], linewidth=2,color = 'cyan',linestyle='--', label = r'Real Derivative')

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5dotx1.pdf', format='pdf')
plt.show()


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

plt.plot(param_set.ts[:index], torch.abs(train_deri[:,0] - deri_train3[:,0]), linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,0] - deri_train1[:,0]), linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,0] - deri_train2[:,0]), linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,0] - filter_deri_train[:,0]), linewidth=2,linestyle='-',color = 'yellow', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,0] - TVR_deri_train[:,0]), linewidth=2,linestyle='-',color = 'black', label = r'Runge-Kutta Training')

print(np.mean((       (train_deri[:,0] - deri_train3[:,0])**2    ).numpy()))
print(np.mean((       (train_deri[:,0] - deri_train1[:,0])**2    ).numpy()))
print(np.mean((       (train_deri[:,0] - deri_train2[:,0])**2    ).numpy()))
print(np.mean((       (train_deri[:,0] - filter_deri_train[:,0])**2    ).numpy()))
print(np.mean((       (train_deri[:,0] - TVR_deri_train[:,0])**2    ).numpy()))
print()

print(np.mean((       (test_deri[:,0] - deri_test3[:,0])**2    ).numpy()))
print(np.mean((       (test_deri[:,0] - deri_test1[:,0])**2    ).numpy()))
print(np.mean((       (test_deri[:,0] - deri_test2[:,0])**2    ).numpy()))
print()

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5dotx1diff.pdf', format='pdf')
plt.show()

# ###################################################################################
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))


plt.plot(param_set.ts[:index], deri_train3[:,1], linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], deri_train1[:,1], linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], deri_train2[:,1], linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], filter_deri_train[:,1], linewidth=2,color = 'yellow',linestyle='-', label = r'Real Function')
plt.plot(param_set.ts[:index], TVR_deri_train[:,1], linewidth=2,color = 'black',linestyle='-', label = r'Real Function')
plt.plot(param_set.ts[:index], train_deri[:,1], linewidth=2,color = 'cyan',linestyle='-', label = r'Real Derivative')

plt.plot(param_set.ts[index:], deri_test3[:,1], linewidth=2,color = 'darkblue',linestyle='--', label = r'Normal Training')
plt.plot(param_set.ts[index:], deri_test1[:,1], linewidth=2,linestyle='--', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[index:], deri_test2[:,1], linewidth=2,linestyle='--',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[index:], test_deri[:,1], linewidth=2,color = 'cyan',linestyle='--', label = r'Real Derivative')

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5dotx2.pdf', format='pdf')
plt.show()


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

plt.plot(param_set.ts[:index], torch.abs(train_deri[:,1] - deri_train3[:,1]), linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,1] - deri_train1[:,1]), linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,1] - deri_train2[:,1]), linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,1] - filter_deri_train[:,1]), linewidth=2,linestyle='-',color = 'yellow', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,1] - TVR_deri_train[:,1]), linewidth=2,linestyle='-',color = 'black', label = r'Runge-Kutta Training')

print(np.mean((       (train_deri[:,1] - deri_train3[:,1])**2    ).numpy()))
print(np.mean((       (train_deri[:,1] - deri_train1[:,1])**2    ).numpy()))
print(np.mean((       (train_deri[:,1] - deri_train2[:,1])**2    ).numpy()))
print(np.mean((       (train_deri[:,1] - filter_deri_train[:,1])**2    ).numpy()))
print(np.mean((       (train_deri[:,1] - TVR_deri_train[:,1])**2    ).numpy()))
print()

print(np.mean((       (test_deri[:,1] - deri_test3[:,1])**2    ).numpy()))
print(np.mean((       (test_deri[:,1] - deri_test1[:,1])**2    ).numpy()))
print(np.mean((       (test_deri[:,1] - deri_test2[:,1])**2    ).numpy()))
print()

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5dotx2diff.pdf', format='pdf')
plt.show()

# ###################################################################################
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))


plt.plot(param_set.ts[:index], deri_train3[:,2], linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], deri_train1[:,2], linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], deri_train2[:,2], linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], filter_deri_train[:,2], linewidth=2,color = 'yellow',linestyle='-', label = r'Real Function')
plt.plot(param_set.ts[:index], TVR_deri_train[:,2], linewidth=2,color = 'black',linestyle='-', label = r'Real Function')
plt.plot(param_set.ts[:index], train_deri[:,2], linewidth=2,color = 'cyan',linestyle='-', label = r'Real Derivative')

plt.plot(param_set.ts[index:], deri_test3[:,2], linewidth=2,color = 'darkblue',linestyle='--', label = r'Normal Training')
plt.plot(param_set.ts[index:], deri_test1[:,2], linewidth=2,linestyle='--', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[index:], deri_test2[:,2], linewidth=2,linestyle='--',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[index:], test_deri[:,2], linewidth=2,color = 'cyan',linestyle='--', label = r'Real Derivative')

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5dotx3.pdf', format='pdf')
plt.show()


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

plt.plot(param_set.ts[:index], torch.abs(train_deri[:,2] - deri_train3[:,2]), linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,2] - deri_train1[:,2]), linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,2] - deri_train2[:,2]), linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,2] - filter_deri_train[:,2]), linewidth=2,linestyle='-',color = 'yellow', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,2] - TVR_deri_train[:,2]), linewidth=2,linestyle='-',color = 'black', label = r'Runge-Kutta Training')

print(np.mean((       (train_deri[:,2] - deri_train3[:,2])**2    ).numpy()))
print(np.mean((       (train_deri[:,2] - deri_train1[:,2])**2    ).numpy()))
print(np.mean((       (train_deri[:,2] - deri_train2[:,2])**2    ).numpy()))
print(np.mean((       (train_deri[:,2] - filter_deri_train[:,2])**2    ).numpy()))
print(np.mean((       (train_deri[:,2] - TVR_deri_train[:,2])**2    ).numpy()))
print()

print(np.mean((       (test_deri[:,2] - deri_test3[:,2])**2    ).numpy()))
print(np.mean((       (test_deri[:,2] - deri_test1[:,2])**2    ).numpy()))
print(np.mean((       (test_deri[:,2] - deri_test2[:,2])**2    ).numpy()))
print()

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5dotx3diff.pdf', format='pdf')
plt.show()


##########################################################################################################
#######################################################################################
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))


plt.plot(param_set.ts[:index], sec_train3[:,0], linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], sec_train1[:,0], linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], sec_train2[:,0], linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], train_sec[:,0], linewidth=2,color = 'cyan',linestyle='-', label = r'Real Derivative')

plt.plot(param_set.ts[index:], sec_test3[:,0], linewidth=2,color = 'darkblue',linestyle='--', label = r'Normal Training')
plt.plot(param_set.ts[index:], sec_test1[:,0], linewidth=2,linestyle='--', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[index:], sec_test2[:,0], linewidth=2,linestyle='--',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[index:], test_sec[:,0], linewidth=2,color = 'cyan',linestyle='--', label = r'Real Derivative')

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5ddotx1.pdf', format='pdf')
plt.show()


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

plt.plot(param_set.ts[:index], torch.abs(train_sec[:,0] - sec_train3[:,0]), linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], torch.abs(train_sec[:,0] - sec_train1[:,0]), linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], torch.abs(train_sec[:,0] - sec_train2[:,0]), linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')


print(np.mean((       (train_sec[:,0] - sec_train3[:,0])**2    ).numpy()))
print(np.mean((       (train_sec[:,0] - sec_train1[:,0])**2    ).numpy()))
print(np.mean((       (train_sec[:,0] - sec_train2[:,0])**2    ).numpy()))
print()

print(np.mean((       (test_sec[:,0] - sec_test3[:,0])**2    ).numpy()))
print(np.mean((       (test_sec[:,0] - sec_test1[:,0])**2    ).numpy()))
print(np.mean((       (test_sec[:,0] - sec_test2[:,0])**2    ).numpy()))
print()

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5ddotx1diff.pdf', format='pdf')
plt.show()

# ###################################################################################

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))


plt.plot(param_set.ts[:index], sec_train3[:,1], linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], sec_train1[:,1], linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], sec_train2[:,1], linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], train_sec[:,1], linewidth=2,color = 'cyan',linestyle='-', label = r'Real Derivative')

plt.plot(param_set.ts[index:], sec_test3[:,1], linewidth=2,color = 'darkblue',linestyle='--', label = r'Normal Training')
plt.plot(param_set.ts[index:], sec_test1[:,1], linewidth=2,linestyle='--', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[index:], sec_test2[:,1], linewidth=2,linestyle='--',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[index:], test_sec[:,1], linewidth=2,color = 'cyan',linestyle='--', label = r'Real Derivative')

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5ddotx2.pdf', format='pdf')
plt.show()


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

plt.plot(param_set.ts[:index], torch.abs(train_sec[:,1] - sec_train3[:,1]), linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], torch.abs(train_sec[:,1] - sec_train1[:,1]), linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], torch.abs(train_sec[:,1] - sec_train2[:,1]), linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')

print(np.mean((       (train_sec[:,1] - sec_train3[:,1])**2    ).numpy()))
print(np.mean((       (train_sec[:,1] - sec_train1[:,1])**2    ).numpy()))
print(np.mean((       (train_sec[:,1] - sec_train2[:,1])**2    ).numpy()))
print()

print(np.mean((       (test_sec[:,1] - sec_test3[:,1])**2    ).numpy()))
print(np.mean((       (test_sec[:,1] - sec_test1[:,1])**2    ).numpy()))
print(np.mean((       (test_sec[:,1] - sec_test2[:,1])**2    ).numpy()))
print()

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5ddotx2diff.pdf', format='pdf')
plt.show()

# ###################################################################################

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))


plt.plot(param_set.ts[:index], sec_train3[:,2], linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], sec_train1[:,2], linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], sec_train2[:,2], linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], train_sec[:,2], linewidth=2,color = 'cyan',linestyle='-', label = r'Real Derivative')

plt.plot(param_set.ts[index:], sec_test3[:,2], linewidth=2,color = 'darkblue',linestyle='--', label = r'Normal Training')
plt.plot(param_set.ts[index:], sec_test1[:,2], linewidth=2,linestyle='--', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[index:], sec_test2[:,2], linewidth=2,linestyle='--',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[index:], test_sec[:,2], linewidth=2,color = 'cyan',linestyle='--', label = r'Real Derivative')

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5ddotx3.pdf', format='pdf')
plt.show()


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

plt.plot(param_set.ts[:index], torch.abs(train_sec[:,2] - sec_train3[:,2]), linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], torch.abs(train_sec[:,2] - sec_train1[:,2]), linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], torch.abs(train_sec[:,2] - sec_train2[:,2]), linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')


print(np.mean((       (train_sec[:,2] - sec_train3[:,2])**2    ).numpy()))
print(np.mean((       (train_sec[:,2] - sec_train1[:,2])**2    ).numpy()))
print(np.mean((       (train_sec[:,2] - sec_train2[:,2])**2    ).numpy()))
print()

print(np.mean((       (test_sec[:,2] - sec_test3[:,2])**2    ).numpy()))
print(np.mean((       (test_sec[:,2] - sec_test1[:,2])**2    ).numpy()))
print(np.mean((       (test_sec[:,2] - sec_test2[:,2])**2    ).numpy()))
print()

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/5ddotx3diff.pdf', format='pdf')
plt.show()
