#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def select_elements(sequence, p):
    # 生成与 sequence 同样大小的随机数数组
    random_values = np.random.rand(len(sequence))
    
    # 找出选中的元素和未选中的元素的索引
    selected_indices = np.where(random_values < p)[0]
    unselected_indices = np.where(random_values >= p)[0]
    
    return selected_indices, unselected_indices


class DataSetFod(Dataset):

    """
    Discription:
        a class that inherits from Datset torch module and prepare the
        data set for us
    attributes:
        time: [t_min:step size :t_max]
        output: dynamic system outputs, e.g. x_1, x_2, etc.
        initial_conds: initial conditions
        length: length of time span [t_min:step size :t_max]
        device: CPU/GPU

    only __getitem__() method is implmented which call by index
    the other can be written if it is required

    It is read-able only and gives: input and output of our DNN module

    """

    def __init__(self, time, output, deri):
        self.time = time
        self.output = output
        self.deri = deri

    def giveback(self):
        time = self.time
        output = self.output
        deri = self.deri
        
        return time, output, deri

  
    
###############################################################################

def train_test_way1( t_scaled, u_scaled, deri_data, param_set ):
    """
    Discription:
        a function that takes args and creat an instance of DataSetFod class
        to prepare the dataset

    args:
        time: [t_min:step size :t_max]
        init_conds: initial conditions
        device: CPU/GPU
        batch_size: int, e.g. 2000
        split_value: float, 0.9, splitting the data set for training and testing
        shuffle: bool, True/False, shuffling our dataset or not
    
    return:
        
        main_train_dataloader
        main_test_dataloader

    """
    

    select_prob = param_set.select_prob

    
    length_data = len(t_scaled)   
    indices = np.arange(0, length_data, dtype=int)
    train_indices, test_indices = select_elements(indices, select_prob)
    
    time_train = t_scaled[train_indices]
    time_test = t_scaled[test_indices]

    output_train = u_scaled[train_indices]
    output_test = u_scaled[test_indices]

    deri_train = deri_data[train_indices]
    deri_test = deri_data[test_indices]
    
    train_dataset = DataSetFod(time_train, output_train, deri_train)
    test_dataset = DataSetFod(time_test, output_test, deri_test)


    return train_dataset, test_dataset




#########################################################################################


def train_test_way2( t_scaled, u_scaled, deri_data, param_set ):
    """
    Discription:
        a function that takes args and creat an instance of DataSetFod class
        to prepare the dataset

    args:
        time: [t_min:step size :t_max]
        init_conds: initial conditions
        device: CPU/GPU
        batch_size: int, e.g. 2000
        split_value: float, 0.9, splitting the data set for training and testing
        shuffle: bool, True/False, shuffling our dataset or not
    
    return:
        
        main_train_dataloader
        main_test_dataloader

    """

    split_value = param_set.split_value
    
    length_data = len(t_scaled)

    

    split = int(split_value * length_data)

    time_train = t_scaled[:split]
    time_test = t_scaled[split:]
    
    
    output_train = u_scaled[:split]
    output_test = u_scaled[split:]
    
    deri_train = deri_data[:split]
    deri_test = deri_data[split:]

    train_dataset = DataSetFod(time_train, output_train, deri_train)
    test_dataset = DataSetFod(time_test, output_test, deri_test)


    return train_dataset, test_dataset


###########################################################################################3


class DataSetFod_Derivative(Dataset):

    def __init__(self, input_data, output_data, output_derivative, device):
        self.input_data = input_data
        self.output_data = output_data
        self.output_derivative = output_derivative
        self.device = device
        self.length = input_data.size()[0]
        
    def __getitem__(self, index):
        current_input = self.input_data[index]
        current_output = self.output_data[index]
        current_derivative = self.output_derivative[index]

        return current_input, current_output, current_derivative

    def __len__(self):
        return self.length

    def device_type(self):
        return self.device
    
    def giveback(self):
        inp =  self.input_data
        outp = self.output_data
        deri = self.output_derivative
        
        return inp, outp, deri


def train_test_derivative( input_data, output_data, output_derivative, param_set):

    device = param_set.device
    batch_size = param_set.batch_size
    split_value = param_set.split_value
    shuffle = param_set.shuffle
    
    
    length_data = len(input_data[0])

    main_train_dataloader = []
    main_test_dataloader = []
    data_full = []
    
    
    for i in range(len(input_data)):
        split = int(split_value * length_data)

        input_train = input_data[i][:split]
        input_test = input_data[i][split:]
        input_full = input_data[i]
        
        output_train = output_data[i][:split]
        output_test = output_data[i][split:]
        output_full = output_data[i]
        
        derivative_train = output_derivative[i][:split]
        derivative_test = output_derivative[i][split:]
        derivative_full = output_derivative[i]
        
        
        train_dataset = DataSetFod_Derivative(input_train, output_train, derivative_train, device)
        test_dataset = DataSetFod_Derivative(input_test, output_test, derivative_test, device)
        full_dataset = DataSetFod_Derivative(input_full, output_full, derivative_full, device)

        train_dataloader = DataLoader(train_dataset, batch_size, shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle)

        main_train_dataloader.append(train_dataloader)
        main_test_dataloader.append(test_dataloader)
        data_full.append(full_dataset)


    return main_train_dataloader, main_test_dataloader, data_full

######################################################################################

class DataSetFod_train2(Dataset):

    def __init__(self, input_data, output_data, output_prediction, output_derivative, output_library, device):
        self.input_data = input_data
        self.output_data = output_data
        self.output_prediction = output_prediction
        self.output_derivative = output_derivative
        self.output_library = output_library
        self.device = device
        self.length = input_data.size()[0]
        
    def __getitem__(self, index):
        current_input = self.input_data[index]
        current_output = self.output_data[index]
        current_prediction = self.output_prediction[index]
        current_derivative = self.output_derivative[index]
        current_library = self.output_library[index]

        return current_input, current_output, current_prediction, current_derivative, current_library

    def __len__(self):
        return self.length

    def device_type(self):
        return self.device
    
    def giveback(self):
        inp =  self.input_data
        outp = self.output_data
        pred = self.output_prediction
        deri = self.output_derivative
        libr = self.output_library
        
        return inp, outp, pred, deri, libr


def train_test_train2( input_data, output_data, output_prediction, output_derivative, output_library, param_set):

    device = param_set.device
    batch_size = param_set.batch_size
    split_value = param_set.split_value
    shuffle = param_set.shuffle
    
    
    length_data = len(input_data[0])

    main_train_dataloader = []
    main_test_dataloader = []
    data_full = []
    
    
    for i in range(len(input_data)):
        split = int(split_value * length_data)

        input_train = input_data[i][:split]
        input_test = input_data[i][split:]
        input_full = input_data[i]
        
        output_train = output_data[i][:split]
        output_test = output_data[i][split:]
        output_full = output_data[i]

        prediction_train = output_prediction[i][:split]
        prediction_test = output_prediction[i][split:]
        prediction_full = output_prediction[i]
        
        derivative_train = output_derivative[i][:split]
        derivative_test = output_derivative[i][split:]
        derivative_full = output_derivative[i]
        
        library_train = output_library[i][:split]
        library_test = output_library[i][split:]
        library_full = output_library[i]
        
        train_dataset = DataSetFod_train2(input_train, output_train, prediction_train, derivative_train, library_train, device)
        test_dataset = DataSetFod_train2(input_test, output_test, prediction_test, derivative_test, library_test, device)
        full_dataset = DataSetFod_train2(input_full, output_full, prediction_full, derivative_full, library_full, device)

        train_dataloader = DataLoader(train_dataset, batch_size, shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle)

        main_train_dataloader.append(train_dataloader)
        main_test_dataloader.append(test_dataloader)
        data_full.append(full_dataset)


    return main_train_dataloader, main_test_dataloader, data_full
