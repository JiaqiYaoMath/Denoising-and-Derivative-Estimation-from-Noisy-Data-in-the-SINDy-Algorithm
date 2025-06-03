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
from Functions.RK4_training import normal_train, Sobolev_train

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

    add_noise = True
    noise_level = 1e-4
    
    def ts_generator(self):
        return self.ts


param_set = ParametersSetting()
#############################################################################3
network = Siren( param_set.num_indp_var+ 1, [80,80,80], int(param_set.num_indp_var) )

optimizer = torch.optim.Adam(
    [
        {
            "params": network.parameters(),
            "lr": 5e-4,
            "weight_decay": 5e-5,
        }
    ]
)


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

###########################################################################
(   
    n1_train_target,
    n1_train_fun,
    n1_train_deri,
    n1_test_fun,
    n1_test_deri,
    network
) = Sobolev_train(
    network,
    optimizer,
    train_data,
    test_data,
    param_set,
    coeffs = [1.0, 0.3]
)
    
optimizer.zero_grad()
network = network.cpu()
torch.cuda.empty_cache() 


torch.save(network, 'model/sobolev.pth')
torch.save(n1_train_fun, 'curve/n1_train_fun.pth')
torch.save(n1_train_deri, 'curve/n1_train_deri.pth')
torch.save(n1_test_fun, 'curve/n1_test_fun.pth')
torch.save(n1_test_deri, 'curve/n1_test_deri.pth')

