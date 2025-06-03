# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import torch
from scipy.signal import savgol_filter
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
    t_max = 6
    sample_density = 0.1
    split_value = 5/6
    select_prob = 1.0
    
    ts = torch.arange(t_min, t_max, sample_density)
    RK_timestep = ts[1] - ts[0]
    time_deriv_coef = 2 / (t_max - t_min)
    
    num_indp_var = 2
    system = 'osc_show'
    fun_scaling_factor = 1.0
    initial_condition = [1e-8, 1.0]

    burnin_iterations = 3000
    device = device
    
    add_noise = True
    noise_level = 1e-2

    def ts_generator(self):
        return self.ts


param_set = ParametersSetting()
index = int(param_set.t_max/param_set.sample_density*param_set.split_value)
#############################################################################3
network = Siren( param_set.num_indp_var + 1, [80,80,80], int(param_set.num_indp_var) )

optimizer = torch.optim.Adam(
    [
        {
            "params": network.parameters(),
            "lr": 5e-4,
            "weight_decay": 5e-4,
        }
    ]
)


y = generator(param_set)

(t_scaled, u_scaled) = y.run()
(t_scaled1, u_scaled1) = y.run()

deri_data = torch.ones_like(u_scaled, dtype= torch.float)
deri_data[:,0] = -0.5*u_scaled[:,0] + u_scaled[:,1]
deri_data[:,1] = -u_scaled[:,0] - 0.5*u_scaled[:,1]

deri_data1 = torch.ones_like(u_scaled1, dtype= torch.float)
deri_data1[:,0] = -0.5*u_scaled1[:,0] + u_scaled1[:,1]
deri_data1[:,1] = -u_scaled1[:,0] - 0.5*u_scaled1[:,1]


if param_set.add_noise:
    u_scaled, deri_data = add_white_noise(u_scaled, deri_data, param_set)
    
train_data, test_data = train_test_way2(
    t_scaled.float(),
    u_scaled.float(),
    deri_data.float(),
    param_set
)

train_data1, test_data1 = train_test_way2(
    t_scaled1.float(),
    u_scaled1.float(),
    deri_data1.float(),
    param_set
)


train_time, train_output, train_deri = train_data.giveback()
test_time, test_output, test_deri = test_data.giveback()

train_time1, train_output1, train_deri1 = train_data1.giveback()
test_time1, test_output1, test_deri1 = test_data1.giveback()

#################################################################################################
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))
plt.plot(param_set.ts[:index], train_output[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
plt.plot(param_set.ts[:index], train_output[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$x_2$')
plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.show()

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))
plt.plot(param_set.ts[:index], train_deri[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', label = r'$\dot{x}_1$')
plt.plot(param_set.ts[:index], train_deri[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$\dot{x}_2$')
plt.xlabel(r'$t$', fontsize=24)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=24)
plt.xticks(fontsize=24)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=24)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.show()
#################################################################################    

train_output = train_output.numpy()
test_output = test_output.numpy()

filter_fun_train = savgol_filter(train_output, window_length=10, polyorder=3, axis=0)
filter_deri_train = savgol_filter(train_output, window_length=4, polyorder=3, deriv=1, axis=0)/param_set.sample_density

filter_fun_test = savgol_filter(test_output, window_length=10, polyorder=3, axis=0)
filter_deri_test = savgol_filter(test_output, window_length=4, polyorder=3, deriv=1, axis=0)/param_set.sample_density

#################################################################################################
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))
plt.plot(param_set.ts[:index], filter_fun_train[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', markersize = 5, label = r'$x_1$')
plt.plot(param_set.ts[:index], filter_fun_train[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$x_2$')
plt.xlabel(r'$t$', fontsize=30)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=24)
plt.xticks(fontsize=30)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=30)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.show()

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))
plt.plot(param_set.ts[:index], filter_deri_train[:,0], linewidth=2, color = 'darkblue',linestyle='--', marker='o', label = r'$\dot{x}_1$')
plt.plot(param_set.ts[:index], filter_deri_train[:,1], linewidth=2, color = 'green',linestyle='--', marker='o', label = r'$\dot{x}_2$')
plt.xlabel(r'$t$', fontsize=30)
#plt.ylabel(r'Function $x$', fontsize=22)
plt.legend(fontsize=24)
plt.xticks(fontsize=30)  # 调整 x 轴刻度字体大小
plt.yticks(fontsize=30)  # 调整 y 轴刻度字体大小
plt.tight_layout()
plt.show()
#################################################################################    

print(np.mean((       (train_output[:,0] - (train_output1[:,0]).numpy())**2    )))
print(np.mean((       (filter_fun_train[:,0] - (train_output1[:,0]).numpy())**2    )))

filter_fun_train = torch.tensor(filter_fun_train, dtype=torch.float)
filter_deri_train = torch.tensor(filter_deri_train, dtype=torch.float)

filter_fun_test = torch.tensor(filter_fun_test, dtype=torch.float)
filter_deri_test = torch.tensor(filter_deri_test, dtype=torch.float)


torch.save(filter_fun_train, "model/filter_fun_train.pt")
torch.save(filter_deri_train, "model/filter_deri_train.pt")
torch.save(filter_fun_test, "model/filter_fun_test.pt")
torch.save(filter_deri_test, "model/filter_deri_test.pt")
