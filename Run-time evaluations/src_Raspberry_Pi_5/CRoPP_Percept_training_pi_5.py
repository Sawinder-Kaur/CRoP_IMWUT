import argparse  
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pyhessian import hessian # Hessian computation
from sklearn.model_selection import train_test_split
import time
import psutil
#import pynvml
import subprocess
import warnings
import os
import numpy as np
import torch
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from data_loader import Speech_Dataset
from torch.utils.data import random_split, WeightedRandomSampler, DataLoader
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score
import copy
import sys
import regularizer as reg
import os
#from model_GRU import featureExtractor, metaCLF
from BiLSTM_model import BiLSTMwithAttention,BiLSTM, CNN, CostumLoss, BiLSTM_relu
from prune_utils import Prune, apply_mask
import itertools


warnings.filterwarnings('ignore')

def import_global_weights(model, global_model, mask,device):
    global_model.to(device)
    toggled_mask = []
    for m in mask:
        ones = torch.ones_like(m, device = device)
        toggled_mask.append((ones - m).to(device))
    masked_global_model = apply_mask(global_model, toggled_mask)
    
    mydict = model.state_dict()
    layer_names = list(mydict)
    i = 0
    if "weight" in layer_names[0]: 
        w_ln = 0
        b_ln = 1
    else: 
        w_ln = 1
        b_ln = 0
    for module, global_module in zip(model.modules(),masked_global_model.modules()):
        #print(layer_names[w_ln])
        
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or (isinstance(module, nn.Linear) and module.out_features != 2):
            #print(model.state_dict()[layer_names[w_ln]])
            model.state_dict()[layer_names[w_ln]].copy_(module.weight + global_module.weight)
            i = i + 1
            w_ln = w_ln+1
            b_ln = b_ln+1
            if "bias" in layer_names[w_ln]:
                w_ln = w_ln +1
                b_ln = b_ln +1
        elif isinstance(module, nn.BatchNorm2d):
            w_ln = w_ln+5
            b_ln = b_ln+5
        elif isinstance(module, nn.GRU):
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0 + global_module.weight_ih_l0)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0 + global_module.weight_hh_l0)
            w_ln = w_ln+4
            b_ln = b_ln+4
            i = i+2       
        elif isinstance(module, nn.LSTM):
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0 + global_module.weight_ih_l0)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0 + global_module.weight_hh_l0)
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0_reverse + global_module.weight_ih_l0_reverse)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0_reverse + global_module.weight_hh_l0_reverse)
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l1 + global_module.weight_ih_l1)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l1 + global_module.weight_hh_l1)
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l1_reverse + global_module.weight_ih_l1_reverse)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l1_reverse + global_module.weight_hh_l1_reverse)
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l2 + global_module.weight_ih_l2)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l2 + global_module.weight_hh_l2)
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l2_reverse + global_module.weight_ih_l2_reverse)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l2_reverse + global_module.weight_hh_l2_reverse)
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l3 + global_module.weight_ih_l3)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l3 + global_module.weight_hh_l3)
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l3_reverse + global_module.weight_ih_l3_reverse)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l3_reverse + global_module.weight_hh_l3_reverse)
            w_ln = w_ln+4
            b_ln = b_ln+4
            #w_ln = w_ln+32
            #b_ln = b_ln+32
            i = i +16
    return model

def get_device_info():
    if device==torch.device("cuda:0"):
        print("CUDA Device Info:")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i)}")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i)}")
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"Total GPU Memory: {info.total / (1024 ** 3):.2f} GB ({info.total / (1024 ** 2):.2f} MB), Free GPU Memory: {info.free / (1024 ** 3):.2f} GB ({info.free / (1024 ** 2):.2f} MB), Used GPU Memory: {info.used / (1024 ** 3):.2f} GB ({info.used / (1024 ** 2):.2f} MB)")
    else:
        print("Using CPU")
        cpu_info = subprocess.run(['lscpu'], capture_output=True, text=True).stdout
        for line in cpu_info.split("\n"):
            if "Architecture" in line or "Model name" in line:
                print(line.strip())

def print_training_stats(_time, energy_consumption, process_memory_info,avg_percentage,mode):
    
    used_memory_gb = process_memory_info / (1024 ** 3)
    used_memory_mb = process_memory_info / (1024 ** 2)
    print(device,mode)
    print(f"Training Time: {_time} seconds")
    print(f"Used Memory: {used_memory_gb:.2f} GB ({used_memory_mb:.2f} MB)")
    print(f"Energy Consumption: {energy_consumption} Joules")
    print(f"Utlization: {avg_percentage} %")

def print_inference_stats(_time, energy_consumption, process_memory_info,mode):
    
    used_memory_gb = process_memory_info / (1024 ** 3)
    used_memory_mb = process_memory_info / (1024 ** 2)
    print(device,mode)
    print(f"Inference Time: {_time} seconds")
    print(f"Used Memory: {used_memory_gb:.2f} GB ({used_memory_mb:.2f} MB)")
    print(f"Energy Consumption: {energy_consumption:.2f} Joules")

def measure_energy_consumption(start_func, end_func):
    def decorator(func):
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            if device==torch.device("cuda:0"):
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                start_percentage= pynvml.nvmlDeviceGetUtilizationRates(handle)
                start_energy = pynvml.nvmlDeviceGetPowerUsage(handle)
            else:
                
                start_percentage = process.cpu_percent(interval=None)/psutil.cpu_count()
                cpu_power_watts=105
                
            start_time = start_func()
            result = func(*args, **kwargs)
            end_time = end_func()
            
            
            # Calculate energy consumption in Joules
            if device==torch.device("cuda:0"):
                end_energy = pynvml.nvmlDeviceGetPowerUsage(handle)
                energy_consumption = (end_energy - start_energy) * (end_time - start_time) / 1000
                end_percentage= pynvml.nvmlDeviceGetUtilizationRates(handle)
                avg_percent=(start_percentage.gpu+end_percentage.gpu)/2
                pynvml.nvmlShutdown()
            else:
                end_percentage = process.cpu_percent(interval=None)/psutil.cpu_count()
                avg_percent = (start_percentage + end_percentage) / 2
                energy_consumption = (avg_percent / 100) * cpu_power_watts * (end_time - start_time) 



            process_memory_info = process.memory_info().rss  # Resident Set Size (RSS) in bytes
            return start_time, end_time, energy_consumption, process_memory_info,avg_percent
        return wrapper
    return decorator



@measure_energy_consumption(time.time, time.time)
def train(model,best_model, train_loader, num_epochs, criterion, device,user_id, mode = 'init'):
    
    model = model.to(device)
    batch_num = 0
    best_model = copy.deepcopy(model)

    best_epoch = 0
    loss_fn = CostumLoss()
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), loss_fn.parameters()), lr=1e-5)
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            batch_num += 1
            inputs,labels = data
            inputs = torch.transpose(inputs, 1, 2)
            labels = labels.type(torch.LongTensor)
            inputs=inputs.type(torch.FloatTensor)
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.type(torch.FloatTensor)
            
            outputs = outputs.to(device)

            if mode == 'ft':
                loss = criterion(outputs,labels)
            else:
                loss = loss_fn(outputs,labels, reg.regularizer(model,'l1', device),device)
                
            loss.backward()
            optimizer.step()
        
            
    return model

def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    test_bal_acc =0
    test_f1 = 0
    for data in tensor_loader:
        inputs, labels = data
        inputs = torch.transpose(inputs, 1, 2)
        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor)

        outputs = model(inputs.float())
        outputs = outputs.type(torch.FloatTensor)
        outputs = outputs.to(device)
        
        predict_y = torch.argmax(outputs,dim=1)
        labels = labels.to(device)
        loss = criterion(outputs,labels)
        predict_y = predict_y.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
         
        test_f1 += f1_score(labels,predict_y)
        test_bal_acc += balanced_accuracy_score(labels,predict_y)
        test_acc += accuracy_score(labels,predict_y)
        test_loss += loss.item() * inputs.size(0)
        
    test_acc = test_acc/len(tensor_loader)
    test_f1 = test_f1/len(tensor_loader)
    test_bal_acc = test_bal_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)

    return float(test_acc),float(test_bal_acc),float(test_f1), float(test_loss)

def balanced_loader(Dataset,batch_size):
    #print(Dataset)
    
    if isinstance(Dataset, torch.utils.data.dataset.Subset):
        target = Dataset.dataset.label#[Dataset.indices]
        indices = Dataset.indices
        target = list(map(target.__getitem__, indices))
        print(len(target))
    else:
        target = Dataset.label
        print(len(target))
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    loader = DataLoader(
        Dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    return loader

if __name__ == '__main__':
    
    device = torch.device("cpu")
    print(device)
    get_device_info()
    
    print("Training...")
    
    user_id = '17'
    seed = 1013
    batch_size = 32
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 300
    tolerance = 0.05
    prune_amount = 0.05
    
    alpha = nn.Parameter(torch.tensor([1e-3],requires_grad=True, device=device))    
    dataset= Speech_Dataset("personalize_train",user_id)

    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - valid_size - train_size
    generator1=torch.Generator().manual_seed(seed)

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size],generator=generator1)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_size, shuffle=True)
    
    print(len(train_dataset))
    print(len(test_dataset))
    
    model = BiLSTM(device)
    global_model = BiLSTM(device)
    best_model = BiLSTM(device)
    
    model.to(device)
    global_model.to(device)
    best_model.to(device)

    model.load_state_dict(torch.load("global_model_bilstm.pt",map_location=torch.device('cpu')))
    global_model.load_state_dict(torch.load("global_model_bilstm.pt",map_location=torch.device('cpu')))
    
    trials=[]
    
    for i in range(5):
    
        train_start, train_end, train_energy, train_memory_info,avg_percent = train(model,best_model, train_loader, num_epochs, criterion, device, user_id, 'init')
        
        _,train_bal_acc,_,_=test(model, train_loader, criterion, device)
            

        prev_mask = []
        fine_mask = []
        train_accuracy_pruned = train_bal_acc
        train_accuracy_orig = train_bal_acc
        while train_accuracy_pruned > (train_accuracy_orig - tolerance) and prune_amount < 0.95:
            print("Pruning amount : {}".format( prune_amount))
            prev_model = copy.deepcopy(model)
            prev_mask = fine_mask
            model, fine_mask, struct_mask = Prune(model, train_loader, prune_amount, "MP_unstruct_LSTM", "pruned_model.pt",device)
            prune_amount += 0.05
            _,train_accuracy_pruned,_,_ = test(model,train_loader,criterion,device)
            print("Train accuracy of the finetuned model : {}".format(train_accuracy_orig))
            print("Train accuracy of the pruned model : {}".format(train_accuracy_pruned))
            model = copy.deepcopy(prev_model)

        if prev_mask != []:
            model = import_global_weights(model, global_model, prev_mask, device)     
        
        train_start2, train_end2, train_energy2, train_memory_info2,avg_percent2  = train(model,best_model, train_loader, 40, criterion, device, user_id, 'final')
        trials.append([train_end2-train_start, (train_energy+train_energy2)//2, (train_memory_info+train_memory_info2)//2,(avg_percent+avg_percent2)//2])
    
    avg=np.mean(trials,axis=0)
    print_training_stats(avg[0],avg[1],avg[2],avg[3],'Training')
    
