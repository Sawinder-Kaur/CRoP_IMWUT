import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss
import random

import torch.nn.functional as F
#from model_GRU import featureExtractor

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
        elif isinstance(module,nn.Conv1d):
            scale = 1.0 - (module.in_channels + module.out_channels + module.kernel_size[0] )/(module.in_channels * module.out_channels * module.kernel_size[0])
            #print(scale)
            parameters_to_prune.append(scale * prune_amount)
        elif (isinstance(module, nn.Linear) and module.out_features == 2):
            parameters_to_prune.append(prune_amount/2.0)
        elif isinstance(module, nn.Linear) :
            scale = 1.0 - (module.in_features + module.out_features)/(module.in_features * module.out_features)
            #print(scale)
            if prune_amount < 0.98 : parameters_to_prune.append(scale * prune_amount + 0.02)
            else: parameters_to_prune.append(scale * prune_amount)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            scale = 1.0 - (module.input_size + module.hidden_size)/(module.input_size * module.hidden_size)
            #print(scale)
            if prune_amount < 0.98 : parameters_to_prune.append(scale * prune_amount + 0.02)
            else: parameters_to_prune.append(scale * prune_amount)
    
    return parameters_to_prune


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
            new_mask.append(module.weight_mask)
        elif isinstance(module,nn.GRU):
            new_mask.append(torch.ones_like(module.weight_ih_l0))
            new_mask.append(torch.ones_like(module.weight_hh_l0)) 
        elif isinstance(module, nn.LSTM):
            new_mask.append(torch.ones_like(module.weight_ih_l0))
            new_mask.append(torch.ones_like(module.weight_hh_l0))
            new_mask.append(torch.ones_like(module.weight_ih_l0_reverse))
            new_mask.append(torch.ones_like(module.weight_hh_l0_reverse))
            new_mask.append(torch.ones_like(module.weight_ih_l1))
            new_mask.append(torch.ones_like(module.weight_hh_l1))
            new_mask.append(torch.ones_like(module.weight_ih_l1_reverse))
            new_mask.append(torch.ones_like(module.weight_hh_l1_reverse))
            new_mask.append(torch.ones_like(module.weight_ih_l2))
            new_mask.append(torch.ones_like(module.weight_hh_l2))
            new_mask.append(torch.ones_like(module.weight_ih_l2_reverse))
            new_mask.append(torch.ones_like(module.weight_hh_l2_reverse))
            new_mask.append(torch.ones_like(module.weight_ih_l3))
            new_mask.append(torch.ones_like(module.weight_hh_l3))
            new_mask.append(torch.ones_like(module.weight_ih_l3_reverse))
            new_mask.append(torch.ones_like(module.weight_hh_l3_reverse))
  
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.remove(module,'weight')
    #prev_mask = new_mask
    #count_parameters(model)
    
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



def magnitude_prune_unstruct_lstm(model,loader,prune_amount,parameters_to_prune,erk, combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Conv2d)or isinstance(module, nn.Linear) or isinstance(module,nn.Conv1d):
            importance_score.append(torch.abs(module.weight))
            #importance_score.append(torch.norm(module.weight,p=1,keepdim=True))
        elif isinstance(module,nn.GRU):
            importance_score.append(torch.abs(module.weight_ih_l0))
            importance_score.append(torch.abs(module.weight_hh_l0)) 
        elif isinstance(module, nn.LSTM):
            importance_score.append(torch.abs(module.weight_ih_l0))
            importance_score.append(torch.abs(module.weight_hh_l0))
            importance_score.append(torch.abs(module.weight_ih_l0_reverse))
            importance_score.append(torch.abs(module.weight_hh_l0_reverse))
            importance_score.append(torch.abs(module.weight_ih_l1))
            importance_score.append(torch.abs(module.weight_hh_l1))
            importance_score.append(torch.abs(module.weight_ih_l1_reverse))
            importance_score.append(torch.abs(module.weight_hh_l1_reverse))
            importance_score.append(torch.abs(module.weight_ih_l2))
            importance_score.append(torch.abs(module.weight_hh_l2))
            importance_score.append(torch.abs(module.weight_ih_l2_reverse))
            importance_score.append(torch.abs(module.weight_hh_l2_reverse))
            importance_score.append(torch.abs(module.weight_ih_l3))
            importance_score.append(torch.abs(module.weight_hh_l3))
            importance_score.append(torch.abs(module.weight_ih_l3_reverse))
            importance_score.append(torch.abs(module.weight_hh_l3_reverse))
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
    if erk:
        i = 0
        j = 0
        for module in generate_mod_list(model):
            if isinstance(module, nn.Conv2d) or isinstance(module,nn.Conv1d) or isinstance(module, nn.Linear):
                split_val = torch.quantile(score,parameters_to_prune[j])
                fine_mask.append(torch.where(importance_score[i] <= split_val,0.0,1.0))
                i+=1
                j+=1          
            elif isinstance(module, nn.GRU):
                split_val = torch.quantile(score,parameters_to_prune[j])
                j+=1
                fine_mask.append(torch.where(importance_score[i] <= split_val,0.0,1.0))
                i+=1
                fine_mask.append(torch.where(importance_score[i] <= split_val,0.0,1.0))
                i+=1
            elif isinstance(module, nn.LSTM):
                split_val = torch.quantile(score,parameters_to_prune[j])
                j+=1
                for x in range(16):
                    fine_mask.append(torch.where(importance_score[i] <= split_val,0.0,1.0))
                    i+=1
                
                     
    else:
        split_val = torch.quantile(score,prune_amount)
        
        for imp_score in importance_score:
            fine_mask.append(torch.where(imp_score <= split_val, 0.0,1.0))  
    """
    for m in fine_mask:
        print(m.size())
    """
    count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, masked_score




def apply_mask(model, model_mask,struct_mask = None):
    mydict = model.state_dict()
    layer_names = list(mydict)
    layer_names = [name for name in layer_names if (name.find('bn') == -1 and name.find('bias') == -1)]
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
    """
    w_ln = 32
    for module in model.modules():
        print(layer_names[w_ln])
        w_ln = w_ln + 1
    """ 
    
    for module in model.modules():
        #print(layer_names[w_ln])
        #print(module)
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or (isinstance(module, nn.Linear) and module.out_features != 2):
            #print(module)
            mask = model_mask[i]
            #print(i)
            #print(mask.size())
            #print(module.weight.size())
            #print(model.state_dict()[layer_names[w_ln]])
            #print(layer_names[w_ln])
            #print(module.weight)
            model.state_dict()[layer_names[w_ln]].copy_(module.weight * mask)
            i = i + 1
            w_ln = w_ln+1
            b_ln = b_ln+1
            
        elif isinstance(module, nn.GRU):
            mask1 = model_mask[i]
            mask2 = model_mask[i+1]
            
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0 * mask1)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0 * mask2)
            w_ln = w_ln+2
            b_ln = b_ln+2
            i = i+2   
        elif isinstance(module, nn.LSTM):
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0 * model_mask[i])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0 * model_mask[i+1])
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0_reverse * model_mask[i+2])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0_reverse * model_mask[i+3])
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l1 * model_mask[i+4])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l1 * model_mask[i+5])
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l1_reverse * model_mask[i+6])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l1_reverse * model_mask[i+7])
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l2 * model_mask[i+8])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l2 * model_mask[i+9])
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l2_reverse * model_mask[i+10])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l2_reverse * model_mask[i+11])
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l3 * model_mask[i+12])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l3 * model_mask[i+13])
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l3_reverse * model_mask[i+14])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l3_reverse * model_mask[i+15])
            w_ln = w_ln+2
            b_ln = b_ln+2
            
            
            #w_ln = w_ln+32
            #b_ln = b_ln+32
            i = i+16   
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


def generate_mod_list(model):
    mod_list = []
    for module in model.featureExtractor.modules():
        if isinstance(module,featureExtractor):
            continue
        mod_list.append(module)
    for module in model.decision.modules():
        if isinstance(module,nn.Sequential):
            continue
        mod_list.append(module)
    return mod_list



def Prune(model, train_loader, prune_amount, prune_mechanism, saved_model_name, device = "cuda"):
    
    parameters_to_prune = prune_model_structured_erdos_renyi_kernel(model,prune_amount)
    #print(parameters_to_prune)
    
    if prune_mechanism == "MP_unstruct_LSTM":
        print("Prune_mechanism : Magnitude Based")
        model_mask, struct_mask, masked_score = magnitude_prune_unstruct_lstm(model,None,prune_amount,parameters_to_prune, 0, device = device)
        
    elif prune_mechanism == "Random":
        model_mask = random_prune_pytorch(model,parameters_to_prune)
        struct_mask = None
    else:
        print("Other Pruning mechanism")
        exit()
    
   
    model = apply_mask(model, model_mask,struct_mask) 
   
    return model, model_mask, struct_mask