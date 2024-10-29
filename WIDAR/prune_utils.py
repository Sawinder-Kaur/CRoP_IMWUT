import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss
import random
from pyhessian import hessian

import torch.nn.functional as F

def count_parameters(model):
    mydict = model.state_dict()
    layer_names = list(mydict)
    total_weights = 0
    non_zero_parameters = 0
    for i in layer_names:
        #print(i)
        if "weight" in i:
            weights = np.abs((model.state_dict()[i]).detach().cpu().numpy())
            total_weights += np.sum(np.ones_like(weights))
            non_zero_parameters += np.count_nonzero(weights)
    
    #print("Total number of parameters : {}".format(int(total_weights)))
    #print("Total number of non-zero parameters : {}".format(non_zero_parameters))
    print("Fraction of paramaters which are non-zero : {}".format(non_zero_parameters/total_weights))
    return non_zero_parameters/total_weights

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
    
def count_parameters_per_layer(model):
    mydict = model.state_dict()
    layer_names = list(mydict)
    total_weights = 0
    non_zero_parameters = 0
    layer_num = 1
    for i in layer_names:
        if "weight" in i:
            #print(i)
            weights = np.abs((model.state_dict()[i]).detach().cpu().numpy())
            total_weights = np.sum(np.ones_like(weights))
            non_zero_parameters = np.count_nonzero(weights)
            print("Pruning at layer {}: {}".format(layer_num,(total_weights-non_zero_parameters)/total_weights))
            layer_num += 1

def prune_model_structured_erdos_renyi_kernel(model,prune_amount):
    
    """
    if prune_amount > 0.98: 
        print("Pruning amount too high for structured pruning")
        exit()
    print("Global Pruning Amount : {}%".format(prune_amount*100))
    """
    parameters_to_prune = []
    for module in  model.modules():
        if isinstance(module, nn.Conv2d):
            #print(module.kernel_size )
            scale = 1.0 - (module.in_channels + module.out_channels + module.kernel_size[0] + module.kernel_size[1] )/(module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1])
            #print(scale)
            parameters_to_prune.append(scale * prune_amount)
            #parameters_to_prune.append(scale * prune_amount/2.0)
        elif (isinstance(module, nn.Linear) and module.out_features == 10):
            parameters_to_prune.append(prune_amount/2.0)
        elif isinstance(module, nn.Linear) :
            scale = 1.0 - (module.in_features + module.out_features)/(module.in_features * module.out_features)
            #print(scale)
            if prune_amount < 0.98 : parameters_to_prune.append(scale * prune_amount + 0.02)
            else: parameters_to_prune.append(scale * prune_amount)
    
    return parameters_to_prune

def magnitude_prune_pytorch(model,parameters_to_prune, prev_mask = None,device = "cuda"):
    i = 0
    for module in  model.modules():
        #print(module)
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            #print(prune_amount)
            prune.ln_structured(module, name="weight", amount=parameters_to_prune[i], n=2, dim=0)
            i +=1
    new_mask = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            new_mask.append(module.weight_mask)  
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.remove(module,'weight')
    
    count_parameters(model)
    
    return new_mask

def random_prune_pytorch(model,parameters_to_prune, prev_mask = None,device = "cuda"):
    i = 0
    for module in  model.modules():
        #print(module)
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            #print(prune_amount)
            prune.random_structured(module, name="weight", amount=parameters_to_prune[i], dim=0)
            i +=1
  
    new_mask = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
            new_mask.append(module.weight_mask)  
        elif isinstance(module, nn.Linear):
            #new_mask.append(torch.transpose(module.weight_mask,0,1))  
            new_mask.append(module.weight_mask)  
  
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.remove(module,'weight')
    #prev_mask = new_mask
    count_parameters(model)
    
    return new_mask

def compute_mask(module, score, prune_ratio):
    split_val = torch.quantile(score,prune_ratio)
    #print("----------------------")
    #print(split_val)
    #print("----------------------")
    struct_mask = torch.where(score <= split_val, 0.0,1.0)
    fine_mask_l = []
    
    weight = module.weight
        
    for mask, m in zip(struct_mask, weight):
        if mask == 0: 
            fine_mask_l.append(torch.zeros_like(m))
        else:
            fine_mask_l.append(torch.ones_like(m))
    #fine_mask_l = torch.cat(fine_mask_l,1)
    fine_mask = torch.stack(fine_mask_l)
    #print(fine_mask)
    #print(module.weight)
    #print(module.weight * fine_mask)
    return fine_mask,struct_mask



def magnitude_prune_unstruct(model,loader,prune_amount, combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d)or isinstance(module, nn.Linear):
            importance_score.append(torch.abs(module.weight))
            #importance_score.append(torch.norm(module.weight,p=1,keepdim=True))
   
    #print(importance_score)
    fine_mask = []
    struct_mask = []
    masked_score = []
    score = []
    i = 0
    for imp_score in importance_score:
        if i == 0: score = torch.reshape(imp_score.data, [-1])
        else : score = torch.cat((score,torch.reshape(imp_score.data, [-1])),0)
        i += 1
    split_val = torch.quantile(score,prune_amount)
    for imp_score in importance_score:
        fine_mask.append(torch.where(imp_score <= split_val, 0.0,1.0))    
    """
    for m in fine_mask:
        print(m.size())
    """
    #torch.where(importance_score3[2] < split_val, 0.0, importance_score3[2].double())
    #model = apply_mask(model, fine_mask,struct_mask)
    #count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, masked_score


def apply_mask(model, model_mask,struct_mask = None):
    mydict = model.state_dict()
    layer_names = list(mydict)
    #print(layer_names)
    #print(model_mask)
    #print(struct_mask)
    i = 0
    if "weight" in layer_names[0]: 
        w_ln = 0
        b_ln = 1
    else: 
        w_ln = 1
        b_ln = 0
    for module in model.modules():
        #print(layer_names[w_ln])
        if "bias" in layer_names[w_ln]:
            w_ln = w_ln +1
            b_ln = b_ln +1
        else:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or (isinstance(module, nn.Linear) and module.out_features != 6):
                mask = model_mask[i]
                #print(model.state_dict()[layer_names[w_ln]])
                model.state_dict()[layer_names[w_ln]].copy_(module.weight * mask)
                i = i + 1
                w_ln = w_ln+1
                b_ln = b_ln+1
            elif isinstance(module, nn.BatchNorm2d):
                w_ln = w_ln+5
                b_ln = b_ln+5
    return model


def sum_total_parameters(model):
    mydict = model.state_dict()
    layer_names = list(mydict)
    total_weights = 0
    for i in layer_names:
        if "weight" in i:
            weights = np.abs((model.state_dict()[i]).detach().cpu().numpy())
            #print(weights)
            total_weights += np.sum(weights)
    return total_weights



def Prune(model, train_loader, prune_amount, prune_mechanism, saved_model_name, device = "cuda"):
    
    parameters_to_prune = prune_model_structured_erdos_renyi_kernel(model,prune_amount)
    #print(parameters_to_prune)
    
    if prune_mechanism == "MP_unstruct":
        print("Prune_mechanism : Magnitude Based")
        model_mask, struct_mask, masked_score = magnitude_prune_unstruct(model,None,prune_amount,device = device)
    elif prune_mechanism == "Random":
        model_mask = random_prune_pytorch(model,parameters_to_prune)
        struct_mask = None
    
            

    #exit()
    model = apply_mask(model, model_mask,struct_mask) 
    
    
    return model, model_mask, struct_mask