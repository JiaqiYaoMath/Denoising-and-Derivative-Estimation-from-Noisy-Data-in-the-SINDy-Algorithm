import torch
from Functions.calculation import Derivative, Library, RK4, Numerical_Derivative, library_deriv, Second_Derivative
import torch.nn as nn
from Functions.data_set import train_test_derivative
import numpy as np
import sys, time


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#########################################################################################
class RK3_Logger:
    def __call__( self, iteration, loss_values, loss_values_fun, loss_values_der,loss_values_smooth, burnin_iterations):
        self.update_terminal(iteration, loss_values, loss_values_fun, loss_values_der, loss_values_smooth, burnin_iterations)

    def update_terminal( self, iteration, loss_values, loss_values_fun, loss_values_der, loss_values_smooth, burnin_iterations):
        """Prints and updates progress of training cycle in command line."""
        sys.stdout.write(
            f"\r RK3 Training: {iteration:>6} \
                loss_values: {loss_values[-1]:>8.2e}\
                loss_values_fun: {loss_values_fun[-1]:8.2e}\
                loss_values_der: {loss_values_der[-1]:8.2e}\
                loss_values_smooth: {loss_values_smooth[-1]:8.2e}"
        )
        
        if (iteration == burnin_iterations -1):
            sys.stdout.write("\n")
            sys.stdout.write("=================")
            sys.stdout.write("\n")

            
        sys.stdout.flush()



class Raw_Logger:
    def __call__( self, iteration, loss_values, burnin_iterations):
        self.update_terminal(iteration, loss_values, burnin_iterations)

    def update_terminal( self, iteration, loss_values, burnin_iterations):
        """Prints and updates progress of training cycle in command line."""
        sys.stdout.write(
            f"\r Normal Training: {iteration:>6} \
                loss_values: {loss_values[-1]:>8.2e}"
        )
        
        if (iteration == burnin_iterations -1):
            sys.stdout.write("\n")
            sys.stdout.write("=================")
            sys.stdout.write("\n")

            
        sys.stdout.flush()

class Sobolev_Logger:
    def __call__( self, iteration, loss_values, loss_values_fun, loss_values_der, burnin_iterations):
        self.update_terminal(iteration, loss_values, loss_values_fun, loss_values_der, burnin_iterations)

    def update_terminal( self, iteration, loss_values, loss_values_fun, loss_values_der, burnin_iterations):
        """Prints and updates progress of training cycle in command line."""
        sys.stdout.write(
            f"\r Sobolev Training: {iteration:>6} \
                loss_values: {loss_values[-1]:>8.2e}\
                loss_values_fun: {loss_values_fun[-1]:>8.2e}\
                loss_values_der: {loss_values_der[-1]:8.2e}"
        )
        
        if (iteration == burnin_iterations -1):
            sys.stdout.write("\n")
            sys.stdout.write("=================")
            sys.stdout.write("\n")

            
        sys.stdout.flush()
#######################################################################################
#######################################################################################
#######################################################################################
def Runge_Kutta(network1, data_train, target_train, step_size, param_set):
    X1 = data_train.clone().detach()
    X2 = data_train.clone().detach()
    X3 = data_train.clone().detach()
    
    coeff = param_set.time_deriv_coef
    
    X2[:,0] = X2[:,0] + step_size*coeff/2
    X3[:,0] = X3[:,0] + step_size*coeff
    
    k1 = Derivative(network1, X1, param_set)
    k2 = Derivative(network1, X2, param_set)
    k3 = Derivative(network1, X3, param_set)
        
    target1 = target_train + step_size*(k1 + 4*k2 + k3)/6
    
    return target1


######################################################################################
def RK4_train(
    network,
    optimizer,
    train_data,
    test_data,
    param_set,
    coeffs = [1.0, 0.7, 5e-2]
) -> None:

    burnin_iterations = param_set.burnin_iterations
    RK_timestep = param_set.RK_timestep
    
    a1 = coeffs[0]
    a2 = coeffs[1]
    a3 = coeffs[2]
    ###############################################################
    
    optimizer.zero_grad()
    network = network.to(device)
    network.train()
    
    train_time, train_output, train_deri = train_data.giveback()
    test_time, test_output, test_deri = test_data.giveback()
    
    #####################################
    train_time = train_time.to(device)
    train_output = train_output.to(device)
    train_deri = train_deri.to(device)
    
    test_time = test_time.to(device)
    test_output = test_output.to(device)
    test_deri = test_deri.to(device)
    #####################################
    
    loss_fn = nn.MSELoss()
    
    losslist_train_target = []
    losslist_train_fun  = []
    losslist_train_runge = []
    losslist_train_smooth = []
    losslist_train_deri = []
    
    losslist_test_fun = []
    losslist_test_deri = []

    logger = RK3_Logger()
    
    
    for iteration in range(burnin_iterations):
        
        optimizer.zero_grad()
       
        ###########################################################
        
        prediction =  network(train_time)         
        loss_train_fun = loss_fn(prediction, train_output)
        ##
        step_size = RK_timestep
        target = Runge_Kutta(network, train_time[:-1], train_output[:-1], step_size, param_set)
        loss_train_runge = loss_fn(target, train_output[1:])
        ##
        sec_deri = Second_Derivative(network, train_time, param_set)
        diff = sec_deri[1:, :] - sec_deri[:-1, :]
        loss_train_smooth = torch.mean(diff**2) 
        ##
        
        loss = a1 * loss_train_fun + a2 * loss_train_runge + a3 * loss_train_smooth
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        ##
        deri = Derivative(network, train_time, param_set)
        loss_train_deri = loss_fn(deri, train_deri)
        optimizer.zero_grad()
        ###########################################################
        
        prediction_test =  network(test_time)         
        loss_test_fun = loss_fn(prediction_test, test_output)
        optimizer.zero_grad()
        
        deri_test = Derivative(network, test_time, param_set)
        loss_test_deri = loss_fn(deri_test, test_deri)
        optimizer.zero_grad()
        
        ##########################################################
        
        losslist_train_target.append(loss.item())
        losslist_train_fun.append(loss_train_fun.item())
        losslist_train_runge.append(loss_train_runge.item())
        losslist_train_smooth.append(loss_train_smooth.item())
        losslist_train_deri.append(loss_train_deri.item())
             
        losslist_test_fun.append(loss_test_fun.item())
        losslist_test_deri.append(loss_test_deri.item())
        
        ##########################################Loggering
        
        logger(iteration, losslist_train_target, losslist_train_fun, losslist_train_runge, losslist_train_smooth, burnin_iterations)
        
        
    return losslist_train_target, losslist_train_fun, losslist_train_runge, losslist_train_smooth, losslist_train_deri, losslist_test_fun, losslist_test_deri, network


#######################################################################################
#######################################################################################
#######################################################################################

def normal_train(
    network,
    optimizer,
    train_data,
    test_data,
    param_set
) -> None:

    burnin_iterations = param_set.burnin_iterations
    ###############################################################
    
    optimizer.zero_grad()
    network = network.to(device)
    network.train()
    
    train_time, train_output, train_deri = train_data.giveback()
    test_time, test_output, test_deri = test_data.giveback()
    
    #####################################
    train_time = train_time.to(device)
    train_output = train_output.to(device)
    train_deri = train_deri.to(device)
    
    test_time = test_time.to(device)
    test_output = test_output.to(device)
    test_deri = test_deri.to(device)
    #####################################
    
    loss_fn = nn.MSELoss()
    
    losslist_train_fun  = []
    losslist_train_deri = []
    
    losslist_test_fun = []
    losslist_test_deri = []
    
    
    logger = Raw_Logger()
    
    for iteration in range(burnin_iterations):
        
        optimizer.zero_grad()
       
        ###########################################################
        
        prediction =  network(train_time)         
        loss_train_fun = loss_fn(prediction, train_output)
                
        loss = 1.0 * loss_train_fun
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        deri = Derivative(network, train_time, param_set)
        loss_train_deri = loss_fn(deri, train_deri)
        optimizer.zero_grad()
        
        ###########################################################
        
        prediction_test =  network(test_time)         
        loss_test_fun = loss_fn(prediction_test, test_output)
        optimizer.zero_grad()
        
        deri_test = Derivative(network, test_time, param_set)
        loss_test_deri = loss_fn(deri_test, test_deri)
        optimizer.zero_grad()

        # ##########################################################
        
        losslist_train_fun.append(loss_train_fun.item())
        losslist_train_deri.append(loss_train_deri.item())
        
        losslist_test_fun.append(loss_test_fun.item())
        losslist_test_deri.append(loss_test_deri.item())
        
        ##########################################Loggering
        
        logger(iteration, losslist_train_fun, burnin_iterations)
        
        
    return losslist_train_fun, losslist_train_deri, losslist_test_fun, losslist_test_deri, network

#######################################################################################
#######################################################################################
#######################################################################################
def Sobolev_train(
    network,
    optimizer,
    train_data,
    test_data,
    param_set,
    coeffs = [1.0, 0.3]
) -> None:
    
    burnin_iterations = param_set.burnin_iterations
    
    a1 = coeffs[0]
    a2 = coeffs[1]
    ###############################################################
    
    optimizer.zero_grad()
    network = network.to(device)
    network.train()
    
    train_time, train_output, train_deri = train_data.giveback()
    test_time, test_output, test_deri = test_data.giveback()
    
    #####################################
    train_time = train_time.to(device)
    train_output = train_output.to(device)
    train_deri = train_deri.to(device)
    
    test_time = test_time.to(device)
    test_output = test_output.to(device)
    test_deri = test_deri.to(device)
    #####################################
    
    loss_fn = nn.MSELoss()
    
    losslist_train_target = []
    losslist_train_fun  = []
    losslist_train_deri = []
    
    losslist_test_fun = []
    losslist_test_deri = []

    logger = Sobolev_Logger()
    
    
    for iteration in range(burnin_iterations):
        
        optimizer.zero_grad()
       
        ###########################################################
        
        prediction =  network(train_time)         
        loss_train_fun = loss_fn(prediction, train_output)

        deri = Derivative(network, train_time, param_set)
        loss_train_deri = loss_fn(deri, train_deri)
                
        loss = a1 * loss_train_fun + a2 * loss_train_deri
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        ###########################################################
        
        prediction_test =  network(test_time)         
        loss_test_fun = loss_fn(prediction_test, test_output)
        optimizer.zero_grad()
        
        deri_test = Derivative(network, test_time, param_set)
        loss_test_deri = loss_fn(deri_test, test_deri)
        optimizer.zero_grad()
        
        ##########################################################
        
        losslist_train_fun.append(loss_train_fun.item())
        losslist_train_deri.append(loss_train_deri.item())
        losslist_train_target.append(loss.item())
        
        losslist_test_fun.append(loss_test_fun.item())
        losslist_test_deri.append(loss_test_deri.item())
        
        ##########################################Loggering
        
        logger(iteration, losslist_train_target, losslist_train_fun, losslist_train_deri, burnin_iterations)
        
        
    return losslist_train_target, losslist_train_fun, losslist_train_deri, losslist_test_fun, losslist_test_deri, network

