import torch
import torch.nn as nn
import numpy as np

def regularizer(model,reg_type, device, gamma = 0.5):
    penalty = torch.tensor(0.0).to(device)
    if reg_type == 'l0':
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                penalty += torch.norm(module.weight,0)
    if reg_type == 'l1':
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                penalty += torch.norm(module.weight,1)
    if reg_type == 'l2':
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                penalty += torch.norm(module.weight,2)
    if reg_type == 'pol':
        t = 10^-6
        sum_params = sum_total_parameters(model)
        num_params = count_total_parameters(model)
        avg_param = sum_params/num_params
        #print(avg_param)
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                penalty += ((torch.norm(module.weight,1) - t*torch.norm(module.weight - avg_param,1)))
        #print(penalty)
    return penalty

def count_total_parameters(model):
    mydict = model.state_dict()
    layer_names = list(mydict)
    total_weights = 0
    for i in layer_names:
        if "weight" in i:
            weights = np.abs((model.state_dict()[i]).detach().cpu().numpy())
            total_weights += np.sum(np.ones_like(weights))
    return total_weights

def sum_total_parameters(model):
    mydict = model.state_dict()
    layer_names = list(mydict)
    total_weights = 0
    for i in layer_names:
        if "weight" in i:
            weights = np.abs((model.state_dict()[i]).detach().cpu().numpy())
            total_weights += np.sum(weights)
    return total_weights