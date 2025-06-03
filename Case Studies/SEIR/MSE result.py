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
    t_max = 176
    sample_density = 1
    split_value = 10/11
    select_prob = 1.0
    
    ts = torch.arange(t_min, t_max, sample_density)
    RK_timestep = ts[1] - ts[0]
    time_deriv_coef = 2 / (t_max - t_min)
    
    num_indp_var = 4
    system = 'seir'
    fun_scaling_factor = 1.0
    initial_condition = [0.999, 0.001, 1e-9, 1e-9]

    burnin_iterations = 3000
    device = device

    add_noise = False
    noise_level = 1e-4
    
    def ts_generator(self):
        return self.ts


param_set = ParametersSetting()
index = int(param_set.t_max/param_set.sample_density*param_set.split_value)
#############################################################################3
y = generator(param_set)

(t_scaled, u_scaled) = y.run()

deri_data = torch.ones_like(u_scaled, dtype= torch.float)
beta = 0.3  
sigma = 1/5  
gamma = 1/10 
s = param_set.fun_scaling_factor

deri_data = torch.ones_like(u_scaled, dtype= torch.float)
deri_data[:,0] = -beta * u_scaled[:,0] * u_scaled[:,2]/s
deri_data[:,1] = beta * u_scaled[:,0] * u_scaled[:,2]/s - sigma * u_scaled[:,1]
deri_data[:,2] = sigma * u_scaled[:,1] - gamma * u_scaled[:,2]
deri_data[:,3] = gamma * u_scaled[:,2]

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
plt.plot(param_set.ts, u_scaled[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$S$')
plt.plot(param_set.ts, u_scaled[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$E$')
plt.plot(param_set.ts, u_scaled[:,2], linewidth=2, color = 'red',linestyle='--', marker='o', label = r'$I$')
plt.plot(param_set.ts, u_scaled[:,3], linewidth=2, color = 'cyan',linestyle='--', marker='o', label = r'$R$')
plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.show()

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
plt.plot(param_set.ts, deri_data[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', label = r'$\dot{S}$')
plt.plot(param_set.ts, deri_data[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$\dot{E}$')
plt.plot(param_set.ts, deri_data[:,2], linewidth=2, color = 'red',linestyle='--', marker='o', label = r'$\dot{I}$')
plt.plot(param_set.ts, deri_data[:,3], linewidth=2, color = 'cyan',linestyle='--', marker='o', label = r'$\dot{R}$')
plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.show()

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
plt.plot(param_set.ts, sec_data[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', label = r'$\ddot{S}$')
plt.plot(param_set.ts, sec_data[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$\ddot{E}$')
plt.plot(param_set.ts, sec_data[:,2], linewidth=2, color = 'red',linestyle='--', marker='o', label = r'$\ddot{I}$')
plt.plot(param_set.ts, sec_data[:,3], linewidth=2, color = 'cyan',linestyle='--', marker='o', label = r'$\ddot{R}$')
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
plt.savefig('figure/3S.pdf', format='pdf')
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
plt.savefig('figure/3Sdiff.pdf', format='pdf')
plt.show()

# ###################################################################################

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
plt.savefig('figure/3E.pdf', format='pdf')
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
plt.savefig('figure/3Ediff.pdf', format='pdf')
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
plt.savefig('figure/3I.pdf', format='pdf')
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
plt.savefig('figure/3Idiff.pdf', format='pdf')
plt.show()

##########################################################################################################

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))


plt.plot(param_set.ts[:index], prediction_train3[:,3], linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], prediction_train1[:,3], linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], prediction_train2[:,3], linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], filter_fun_train[:,3], linewidth=2,linestyle='-',color = 'yellow', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], TVR_fun_train[:,3], linewidth=2,linestyle='-',color = 'black', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], train_output[:,3], linewidth=2,color = 'cyan',linestyle='-', label = r'Real Function')

plt.plot(param_set.ts[index:], prediction_test3[:,3], linewidth=2,color = 'darkblue',linestyle='--', label = r'Normal Training')
plt.plot(param_set.ts[index:], prediction_test1[:,3], linewidth=2,linestyle='--', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[index:], prediction_test2[:,3], linewidth=2,linestyle='--',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[index:], test_output[:,3], linewidth=2,color = 'cyan',linestyle='--', label = r'Real Function')


plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/3R.pdf', format='pdf')
plt.show()


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

plt.plot(param_set.ts[:index], torch.abs(train_output[:,3] - prediction_train3[:,3]), linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,3] - prediction_train1[:,3]), linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,3] - prediction_train2[:,3]), linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,3] - filter_fun_train[:,3]), linewidth=2,linestyle='-',color = 'yellow', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_output[:,3] - TVR_fun_train[:,3]), linewidth=2,linestyle='-',color = 'black', label = r'Runge-Kutta Training')

print(np.mean((       (train_output[:,3] - prediction_train3[:,3])**2    ).numpy()))
print(np.mean((       (train_output[:,3] - prediction_train1[:,3])**2    ).numpy()))
print(np.mean((       (train_output[:,3] - prediction_train2[:,3])**2    ).numpy()))
print(np.mean((       (train_output[:,3] - filter_fun_train[:,3])**2    ).numpy()))
print(np.mean((       (train_output[:,3] - TVR_fun_train[:,3])**2    ).numpy()))
print()

print(np.mean((       (test_output[:,3] - prediction_test3[:,3])**2    ).numpy()))
print(np.mean((       (test_output[:,3] - prediction_test1[:,3])**2    ).numpy()))
print(np.mean((       (test_output[:,3] - prediction_test2[:,3])**2    ).numpy()))
print()

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/3Rdiff.pdf', format='pdf')
plt.show()

#######################################################################################
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
plt.savefig('figure/3dotS.pdf', format='pdf')
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
plt.savefig('figure/3dotSdiff.pdf', format='pdf')
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
plt.savefig('figure/3dotE.pdf', format='pdf')
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
plt.savefig('figure/3dotEdiff.pdf', format='pdf')
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
plt.savefig('figure/3dotI.pdf', format='pdf')
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
plt.savefig('figure/3dotIdiff.pdf', format='pdf')
plt.show()

# ###################################################################################

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))


plt.plot(param_set.ts[:index], deri_train3[:,3], linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], deri_train1[:,3], linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], deri_train2[:,3], linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], filter_deri_train[:,3], linewidth=2,color = 'yellow',linestyle='-', label = r'Real Function')
plt.plot(param_set.ts[:index], TVR_deri_train[:,3], linewidth=2,color = 'black',linestyle='-', label = r'Real Function')
plt.plot(param_set.ts[:index], train_deri[:,3], linewidth=2,color = 'cyan',linestyle='-', label = r'Real Derivative')

plt.plot(param_set.ts[index:], deri_test3[:,3], linewidth=2,color = 'darkblue',linestyle='--', label = r'Normal Training')
plt.plot(param_set.ts[index:], deri_test1[:,3], linewidth=2,linestyle='--', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[index:], deri_test2[:,3], linewidth=2,linestyle='--',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[index:], test_deri[:,3], linewidth=2,color = 'cyan',linestyle='--', label = r'Real Derivative')

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/3dotR.pdf', format='pdf')
plt.show()


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

plt.plot(param_set.ts[:index], torch.abs(train_deri[:,3] - deri_train3[:,3]), linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,3] - deri_train1[:,3]), linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,3] - deri_train2[:,3]), linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,3] - filter_deri_train[:,3]), linewidth=2,linestyle='-',color = 'yellow', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], torch.abs(train_deri[:,3] - TVR_deri_train[:,3]), linewidth=2,linestyle='-',color = 'black', label = r'Runge-Kutta Training')

print(np.mean((       (train_deri[:,3] - deri_train3[:,3])**2    ).numpy()))
print(np.mean((       (train_deri[:,3] - deri_train1[:,3])**2    ).numpy()))
print(np.mean((       (train_deri[:,3] - deri_train2[:,3])**2    ).numpy()))
print(np.mean((       (train_deri[:,3] - filter_deri_train[:,3])**2    ).numpy()))
print(np.mean((       (train_deri[:,3] - TVR_deri_train[:,3])**2    ).numpy()))
print()

print(np.mean((       (test_deri[:,3] - deri_test3[:,3])**2    ).numpy()))
print(np.mean((       (test_deri[:,3] - deri_test1[:,3])**2    ).numpy()))
print(np.mean((       (test_deri[:,3] - deri_test2[:,3])**2    ).numpy()))
print()

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/3dotRdiff.pdf', format='pdf')
plt.show()


#######################################################################################
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
plt.savefig('figure/3ddotS.pdf', format='pdf')
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
plt.savefig('figure/3ddotSdiff.pdf', format='pdf')
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
plt.savefig('figure/3ddotE.pdf', format='pdf')
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
plt.savefig('figure/3ddotEdiff.pdf', format='pdf')
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
plt.savefig('figure/3ddotI.pdf', format='pdf')
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
plt.savefig('figure/3ddotIdiff.pdf', format='pdf')
plt.show()

# ###################################################################################

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))


plt.plot(param_set.ts[:index], sec_train3[:,3], linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], sec_train1[:,3], linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], sec_train2[:,3], linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[:index], train_sec[:,3], linewidth=2,color = 'cyan',linestyle='-', label = r'Real Derivative')

plt.plot(param_set.ts[index:], sec_test3[:,3], linewidth=2,color = 'darkblue',linestyle='--', label = r'Normal Training')
plt.plot(param_set.ts[index:], sec_test1[:,3], linewidth=2,linestyle='--', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[index:], sec_test2[:,3], linewidth=2,linestyle='--',color = 'red', label = r'Runge-Kutta Training')
plt.plot(param_set.ts[index:], test_sec[:,3], linewidth=2,color = 'cyan',linestyle='--', label = r'Real Derivative')

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/3ddotR.pdf', format='pdf')
plt.show()


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

plt.plot(param_set.ts[:index], torch.abs(train_sec[:,3] - sec_train3[:,3]), linewidth=2,color = 'darkblue',linestyle='-', label = r'Normal Training')
plt.plot(param_set.ts[:index], torch.abs(train_sec[:,3] - sec_train1[:,3]), linewidth=2,linestyle='-', color = 'green', label = r'Sobolev Training')
plt.plot(param_set.ts[:index], torch.abs(train_sec[:,3] - sec_train2[:,3]), linewidth=2,linestyle='-',color = 'red', label = r'Runge-Kutta Training')

print(np.mean((       (train_sec[:,3] - sec_train3[:,3])**2    ).numpy()))
print(np.mean((       (train_sec[:,3] - sec_train1[:,3])**2    ).numpy()))
print(np.mean((       (train_sec[:,3] - sec_train2[:,3])**2    ).numpy()))
print()

print(np.mean((       (test_sec[:,3] - sec_test3[:,3])**2    ).numpy()))
print(np.mean((       (test_sec[:,3] - sec_test1[:,3])**2    ).numpy()))
print(np.mean((       (test_sec[:,3] - sec_test2[:,3])**2    ).numpy()))
print()

plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
#plt.title(r'$x_1$', fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.savefig('figure/3ddotRdiff.pdf', format='pdf')
plt.show()