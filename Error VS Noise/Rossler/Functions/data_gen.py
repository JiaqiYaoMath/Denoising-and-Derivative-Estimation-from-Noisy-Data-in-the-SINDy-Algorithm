#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from scipy.integrate import odeint, ode, solve_ivp
from sympy import *
import matplotlib.pyplot as plt
import random

#####################################
#####################################
#####################################

from abc import ABCMeta, abstractstaticmethod
from scipy.integrate import odeint, ode, solve_ivp
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Functions.utiles import NormalizedData, time_scaling_func


########################################
########################################
########################################
########################################
########################################
########################################
########################################


class Interface_Dynamic_System(metaclass = ABCMeta):

    @abstractstaticmethod
    def Diff_Equation(self):
        pass



class Lorenz(Interface_Dynamic_System):
        
    @staticmethod   
    def Diff_Equation(state,t):
        #lorenz system
        #print(initial_condition)
        x, y, z = state  # Unpack the state vector
                
        rho = 28.0
        sigma = 10.0
        beta = 8.0 / 3.0
        
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z



class Rosser(Interface_Dynamic_System):
        
    @staticmethod   
    def Diff_Equation(state,t):
        #lorenz system
        #print(initial_condition)
        a = 0.2
        b = 0.2
        c = 5.7
        
        x, y, z = state  # Unpack the state vector
                
        dxdt = -y - z
        dydt = x + a*y
        dzdt = b + z*(x - c)
        
        return dxdt, dydt, dzdt


class SEIR(Interface_Dynamic_System):
        
    @staticmethod   
    def Diff_Equation(state,t):
        #lorenz system
        #print(initial_condition)
        beta = 0.3  
        sigma = 1/5  
        gamma = 1/10 
        
        S,E,I,R = state  # Unpack the state vector
                
        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        
        return dSdt, dEdt, dIdt, dRdt
    
    
class Simple(Interface_Dynamic_System):
        
    @staticmethod   
    def Diff_Equation(state,t):
        #lorenz system
        #print(initial_condition)
        x, y = state  # Unpack the state vector
                

        
        return y, (x+2*y)/3



class Sincos(Interface_Dynamic_System):
        
    @staticmethod   
    def Diff_Equation(state,t):
        #lorenz system
        #print(initial_condition)
        x, y = state  # Unpack the state vector
                
        return y+0 , -x+0


    
class Osc_show(Interface_Dynamic_System):
        
    @staticmethod   
    def Diff_Equation(state,t):
        #lorenz system
        #print(initial_condition)
        x, y = state  # Unpack the state vector
                
        return - 0.5 * x+y+0 , -x - 0.5 * y+0
    

class Two_D_Oscillator(Interface_Dynamic_System):
    
    @staticmethod
    def Diff_Equation(state,t):
        
        x, y = state
        
        return -0.1 * x + 3 * y + 0, -3 * x - 0.1 * y + 0


class First_Order_Dyn_Sys(Interface_Dynamic_System):
    @staticmethod
    def Diff_Equation(state,t):
        
        x = state
        
        return -2 * x

class Three_D_Non_Linear(Interface_Dynamic_System):
    
    @staticmethod
    def Diff_Equation(state,t):
        
        x,y,z = state  # Unpack the state vector
        return y,  z, -3*x + -1*y + -2.67*z + -1*x*z 
    
    



class Fitz_Hugh_Nagumo(Interface_Dynamic_System):
    
    @staticmethod
    def Diff_Equation(state,t):
        v, w = state
        a_1:float = -0.33333
        a_2:float = 0.04
        a_3:float = -0.028
        b_1:float = 0.5
        b_2:float = 0.032
        
        return v - w  -0.33333 * v*v*v + 0.5 , 0.04 * v - 0.028 * w + 0.032




class Fitz_Hugh_Nagumo_2(Interface_Dynamic_System):
    
    @staticmethod
    def Diff_Equation(state,t):
        v, w = state
        a_1:float = -0.33333
        a_2:float = 0.04
        a_3:float = -0.028
        b_1:float = 0.5
        b_2:float = 0.032
        
        return v - w  -0.33333 * v*v*v + 0.1 , 0.1 * v - 0.1 * w




class Cubic_Damped_Oscillator(Interface_Dynamic_System):
    @staticmethod
    def Diff_Equation(state,t):
        x,y = state        
        return -0.1*x**3 + 2* y**3, -2*x**3 - 0.1* y**3 


class GlycolyticOsci_model(Interface_Dynamic_System):
    
    @staticmethod
    def Diff_Equation(x,t):
        k1,k2,k3,k4,k5,k6 = (100.0, 6.0, 16.0, 100.0, 1.28, 12.0)
        j0,k,kappa,q,K1, phi, N, A = (2.5,1.8, 13.0, 4.0, 0.52, 0.1, 1.0, 4.0)
        ##
        dx0 = j0 - (k1*x[0]*x[5])/(1.0 + (x[5]/K1)**q)

        dx1 = (2*k1*x[0]*x[5])/(1.0 + (x[5]/K1)**q) - k2*x[1]*(N-x[4]) - k6*x[1]*x[4]

        dx2 = k2*x[1]*(N-x[4]) - k3*x[2]*(A-x[5])

        dx3 = k3*x[2]*(A-x[5]) - k4*x[3]*x[4] - kappa*(x[3]- x[6])

        dx4 = k2*x[1]*(N-x[4]) - k4*x[3]*x[4] - k6*x[1]*x[4]

        dx5 = -(2*k1*x[0]*x[5])/(1.0 + (x[5]/K1)**q) + 2*k3*x[2]*(A-x[5]) - k5*x[5]

        dx6 = phi*kappa*(x[3]-x[6]) - k*x[6]

        return np.array([dx0,dx1,dx2,dx3,dx4,dx5,dx6])


##########################
##########################

class generator():
    
    def __init__(self,param_set):
        
        
        self.t = param_set.ts
        self.initial_condition = param_set.initial_condition
        self.scaling_factor = param_set.fun_scaling_factor
        self.func_name = param_set.system

    
    def function_data(self):
        
        try:
            
            if self.func_name == "lorenz":
                dyn_sys_obj = Lorenz()
                
            if self.func_name == 'rosser':
                dyn_sys_obj = Rosser()
            
            if self.func_name == 'seir':
                dyn_sys_obj = SEIR()
                
            if self.func_name == "2_D_Oscilator":
                dyn_sys_obj = Two_D_Oscillator()
            

            if self.func_name == "sincos":
                
                dyn_sys_obj = Sincos()
            
            if self.func_name == 'osc_show':
                
                dyn_sys_obj = Osc_show()
                
            if self.func_name == "Fitz-Hugh Nagumo":
                
                #dyn_sys_obj = Fitz_Hugh_Nagumo()
                dyn_sys_obj = Fitz_Hugh_Nagumo_2()
                
            if self.func_name == "Cubic Damped Oscillator":
                
                dyn_sys_obj = Cubic_Damped_Oscillator()
                
            if self.func_name == "GlycolyticOsci_model":
            
                dyn_sys_obj = GlycolyticOsci_model()
            if self.func_name == "First Order Dynamic System":
                dyn_sys_obj = First_Order_Dyn_Sys()
                
            if self.func_name == "3_D_Non_Linear":
                
                dyn_sys_obj = Three_D_Non_Linear()

            if self.func_name == "simple":
                
                print('!')
                dyn_sys_obj = Simple()
                
            return dyn_sys_obj    
            
        except AssertionError as _e:
                print(_e)
    
                

    def data_prepration(self, dyn_sys_obj):
        
        u_original = odeint(dyn_sys_obj.Diff_Equation, self.initial_condition, self.t)
        u_scaled = torch.from_numpy(self.scaling_factor*u_original)
        
  
        t_scaled = time_scaling_func(self.t)
        constant = torch.ones_like( u_scaled , dtype=torch.float)
        t_scaled = torch.hstack((t_scaled, constant))
        
        return (t_scaled, u_scaled)
    
    
    def run(self):
            
        dyn_sys_obj = self.function_data()
        
        (t_scaled, u_scaled) = self.data_prepration(dyn_sys_obj)

        return (t_scaled, u_scaled)
    





####################################
####################################
####################################

####################################
####################################
####################################

####################################
####################################




####################################
####################################
####################################

####################################
####################################
####################################

####################################
####################################







###################
###################
###################
    

###################
###################
################### Ashwin section

#import list_functions


class rational_A:
    
    def __init__(self, t : torch.tensor, 
                 list_initial_conditions, scaling_factor, function_name) :
        
        #self.func = getattr(self, function_name)
        self.func = list_functions.__dict__[foo]


#####################
