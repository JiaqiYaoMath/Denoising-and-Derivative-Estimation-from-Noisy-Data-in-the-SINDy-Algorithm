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
from scipy.signal import savgol_filter
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
from Functions.TVR import TVR

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
    noise_level = 1e-4

    def ts_generator(self):
        return self.ts


param_set = ParametersSetting()
index = int(param_set.t_max/param_set.sample_density*param_set.split_value)
#############################################################################3
network = Siren( param_set.num_indp_var+ 1, [80,80,80], int(param_set.num_indp_var) )

optimizer = torch.optim.Adam(
    [
        {
            "params": network.parameters(),
            "lr": 5e-4,
            "weight_decay": 5e-7,
        }
    ]
)


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

############################################################################

train_time, train_output, train_deri = train_data.giveback()
test_time, test_output, test_deri = test_data.giveback()
###########################################################################

train_output = train_output.numpy()
test_output = test_output.numpy()

n = len(train_output)
dx = param_set.sample_density

func_tvr = TVR(n,dx,target = 'function')
deri_tvr = TVR(n,dx,target = 'derivative')

TVR_fun_train = np.ones_like(train_output, dtype = float)
TVR_deri_train = np.ones_like(train_output, dtype = float)

for i in range(TVR_fun_train.shape[1]):
    data = train_output[:,i]
    
    (fun,_) = func_tvr.get_solution(
    data=data, 
    solution_guess=np.full(n,0.0, dtype = float), 
    alpha=1e-3,
    no_opt_steps=100
    )
    
    TVR_fun_train[:,i] = fun


for i in range(TVR_deri_train.shape[1]):
    data = train_output[:,i]
    
    (deri,_) = deri_tvr.get_solution(
    data=data, 
    solution_guess=np.full(n,0.0, dtype = float), 
    alpha=1e-4,
    no_opt_steps=100
    )
    
    TVR_deri_train[:,i] = deri
#################################################################################################
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))
plt.plot(param_set.ts[:index], TVR_fun_train[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
plt.plot(param_set.ts[:index], TVR_fun_train[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$x_2$')
plt.plot(param_set.ts[:index], TVR_fun_train[:,2], linewidth=2, color = 'red',linestyle='--', marker='o', label = r'$x_3$')
plt.xlabel(r'$t$', fontsize=30)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=24)
plt.xticks(fontsize=30)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=30)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.show()

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))
plt.plot(param_set.ts[:index], TVR_deri_train[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', label = r'$\dot{x}_1$')
plt.plot(param_set.ts[:index], TVR_deri_train[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$\dot{x}_2$')
plt.plot(param_set.ts[:index], TVR_deri_train[:,2], linewidth=2, color = 'red',linestyle='--', marker='o', label = r'$\dot{x}_3$')
plt.xlabel(r'$t$', fontsize=30)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=24)
plt.xticks(fontsize=30)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=30)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.show()
#################################################################################    

TVR_fun_train = torch.tensor(TVR_fun_train, dtype=torch.float)
TVR_deri_train = torch.tensor(TVR_deri_train, dtype=torch.float)


torch.save(TVR_fun_train, "model/TVR_fun_train.pt")
torch.save(TVR_deri_train, "model/TVR_deri_train.pt")

