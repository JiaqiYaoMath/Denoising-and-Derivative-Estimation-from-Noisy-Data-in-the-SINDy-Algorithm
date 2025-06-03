import numpy as np
import torch
from torch.autograd import grad
from itertools import chain,combinations, combinations_with_replacement as combinations_w_r

###########################################################################################

def add_white_noise(u_scaled, deri_data, param_set):
    
    column_variances = torch.var(u_scaled, dim=0, unbiased=False)
    data_variance = torch.mean(column_variances).item()
    
    var1 = data_variance *  (param_set.noise_level)
    var2 = var1 / (param_set.sample_density)**2
    
    function_noise = torch.randn_like(u_scaled) * np.sqrt(var1)
    deri_noise = torch.randn_like(deri_data) * np.sqrt(var2)
    
    u_scaled = u_scaled + function_noise
    deri_data = deri_data + deri_noise
    
    return u_scaled, deri_data


#########################################################################################
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)
    


def library_deriv(data, prediction, time_deriv_coef):
    
    """

    """
    dy = gradients(prediction, data, order = 1)
    time_deriv = dy[:, 0:1]    
    return time_deriv
    

def Derivative(network, data, param_set):
    
    time_deriv_coef = param_set.time_deriv_coef
    
    X = data.clone().detach().requires_grad_(True)
    prediction = network(X)
    
    time_deriv_list = []
      
    for output in np.arange(prediction.shape[1]):
        time_deriv = time_deriv_coef * library_deriv(X, prediction[:, output : output + 1], time_deriv_coef )
        time_deriv_list.append(time_deriv)   
            
    time_deriv_list = torch.t(torch.squeeze(torch.stack(time_deriv_list)))
    
    return time_deriv_list


def second_library_deriv(data, prediction, time_deriv_coef):
    
    """

    """
    dy = gradients(prediction, data, order = 2)
    time_deriv =  time_deriv_coef * dy[:, 0:1]    
    
    return time_deriv


def Second_Derivative(network, data, param_set):
    
    time_deriv_coef = param_set.time_deriv_coef
    
    X = data.clone().detach().requires_grad_(True)
    prediction = network(X)
    
    time_deriv_list = []
    
    for output in np.arange(prediction.shape[1]):
        time_deriv = second_library_deriv(X, prediction[:, output : output + 1], time_deriv_coef )
        time_deriv_list.append(time_deriv)   
            
    time_deriv_list = torch.t(torch.squeeze(torch.stack(time_deriv_list)))
    
    return time_deriv_list
######################################################################################

def Numerical_Derivative(prediction, param_set):
    
    h = param_set.RK_timestep
    # 用来存储每一列的导数
    derivative = torch.zeros_like(prediction)
    m,n = prediction.shape
    
    # 对每一列进行数值微分
    for i in range(n):
        for j in range(1, m - 1):  # 从第二行到倒数第二行
            derivative[j, i] = (prediction[j + 1, i] - prediction[j - 1, i]) / (2 * h)
    
        # 处理边界
        derivative[0, i] = (prediction[1, i] - prediction[0, i]) / h  # 第一行用前向差分
        derivative[m - 1, i] = (prediction[m - 1, i] - prediction[m - 2, i]) / h  # 最后一行用后向差分
    
    return derivative

#####################################################################################
def Library(x,param_set):
    """

    """
    
    degree = param_set.poly_order
    include_interaction = param_set.include_interaction
    include_bias = param_set.include_bias
    interaction_only = param_set.interaction_only
                      
                      
    n_samples, n_features = x.shape
    bias = torch.reshape(torch.pow(x[:,0],0),(n_samples,1))
    to_stack = []
    
    if include_bias:
        to_stack.append(bias)
        
    
    combinations_main_2 =_combinations(param_set)
    
    
    columns=[]
    for i in combinations_main_2:
        if i:
            out_col = 1
            for col_idx in i:
                out_col = x[:, col_idx].multiply(out_col)

            out_col = torch.reshape(out_col,(n_samples,1))
            columns.append(out_col)
        else:
            bias = torch.reshape(torch.pow(x[:,0],0),(n_samples,1))
            columns.append(bias)
    thetas = torch.t(torch.stack(columns).squeeze())

    return thetas



def RK4(network, x, param_set):
    """

    """
    
    
    timestep = param_set.RK_timestep 
    
    poly_dic = Library(
        x,
        param_set
    )

    k1 =  network(poly_dic)

    poly_dic2 = Library(
        x + 0.5 * timestep * k1,
        param_set
    )

    k2 =  network(poly_dic2)

    poly_dic3 = Library(
        x + 0.5 * timestep * k2,
        param_set
    )

    k3 =  network(poly_dic3)
    poly_dic4 = Library(
        x + 1.0 * timestep * k3,
        param_set
    )

    k4 =  network(poly_dic4)

    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep




#######################################################################################################


def _combinations(param_set):  
    """ 
 
    """
    n_features = param_set.num_indp_var
    degree = param_set.poly_order
    include_interaction = param_set.include_interaction
    include_bias = param_set.include_bias
    interaction_only = param_set.interaction_only
    
    comb = combinations if interaction_only else combinations_w_r
    start = int(not include_bias)
    if not include_interaction:
        if include_bias:
            return chain(
                [()],
                chain.from_iterable(
                    combinations_w_r([j], i)
                    for i in range(1, degree + 1)
                    for j in range(n_features)
                ),
            )
        else:
            return chain.from_iterable(
                combinations_w_r([j], i)
                for i in range(1, degree + 1)
                for j in range(n_features)
            )
    return chain.from_iterable(
        comb(range(n_features), i) for i in range(start, degree + 1)
    )


##############################################################################################3

def least_square(A, B):
    """
    计算最小二乘解 C，使得 A ≈ B * C
    使用正规方程 (B^T * B) * C = B^T * A 进行求解

    参数:
    A (torch.Tensor): m x n 矩阵
    B (torch.Tensor): m x k 矩阵

    返回:
    torch.Tensor: k x n 矩阵 C
    """
    # 计算 B^T * B 和 B^T * A
    BTB = B.T @ B
    BTA = B.T @ A

    # 使用 torch.linalg.solve 解线性方程
    C = torch.linalg.solve(BTB, BTA)
    
    return C