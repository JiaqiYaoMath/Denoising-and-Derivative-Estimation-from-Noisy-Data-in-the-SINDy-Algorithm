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

    burnin_iterations = 1500
    device = device
    
    add_noise = True
    noise_level = 1e-0
    
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
            "weight_decay": 1e-3,
        }
    ]
)



y = generator(param_set)

(t_scaled, u_scaled) = y.run()

deri_data = torch.ones_like(u_scaled, dtype= torch.float)
deri_data[:,0] = -0.1*u_scaled[:,0] + 3.0*u_scaled[:,1]
deri_data[:,1] = -3.0*u_scaled[:,0] - 0.1*u_scaled[:,1]

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
plt.figure(figsize=(8, 6))
plt.plot(param_set.ts, u_scaled[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
plt.plot(param_set.ts, u_scaled[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$x_2$')
plt.xlabel(r'$t$', fontsize=30)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=24)
plt.xticks(fontsize=30)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=30)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.show()

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))
plt.plot(param_set.ts, deri_data[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', label = r'$\dot{x}_1$')
plt.plot(param_set.ts, deri_data[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$\dot{x}_2$')
plt.xlabel(r'$t$', fontsize=30)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=24)
plt.xticks(fontsize=30)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=30)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.show()
#################################################################################    


train_time, train_output, train_deri = train_data.giveback()
test_time, test_output, test_deri = test_data.giveback()
###########################################################################
(   
    n3_train_fun,
    n3_train_deri,
    n3_test_fun,
    n3_test_deri,
    network
) = normal_train(
    network,
    optimizer,
    train_data,
    test_data,
    param_set
)

optimizer.zero_grad()
network = network.cpu()
torch.cuda.empty_cache() 

torch.save(network, 'model/normal.pth')
torch.save(n3_train_fun, 'curve/n3_train_fun.pth')
torch.save(n3_train_deri, 'curve/n3_train_deri.pth')
torch.save(n3_test_fun, 'curve/n3_test_fun.pth')
torch.save(n3_test_deri, 'curve/n3_test_deri.pth')
