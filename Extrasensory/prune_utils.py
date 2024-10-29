import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss
import random
from extrasensory_model import featureExtractor
import torch.nn.functional as F
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
    for module in  generate_mod_list(model):
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
        elif isinstance(module, nn.GRU):
            scale = 1.0 - (module.input_size + module.hidden_size)/(module.input_size * module.hidden_size)
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
        elif isinstance(module, nn.GRU):
            i+=1
  
    new_mask = []
    for module in generate_mod_list(model): #model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
            new_mask.append(module.weight_mask)  
        elif isinstance(module, nn.Linear):
            #new_mask.append(torch.transpose(module.weight_mask,0,1))  
            new_mask.append(module.weight_mask)  
        elif isinstance(module, nn.GRU):
            #print("here")
            new_mask.append(torch.ones_like(module.weight_ih_l0))
            new_mask.append(torch.ones_like(module.weight_hh_l0))
  
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.remove(module,'weight')
    #prev_mask = new_mask
    #count_parameters(model)
    
    return new_mask

def compute_mask(module, score, prune_ratio,GRU_index=None):
    split_val = torch.quantile(score,prune_ratio)
    #print("----------------------")
    #print(split_val)
    #print("----------------------")
    struct_mask = torch.where(score <= split_val, 0.0,1.0)
    fine_mask_l = []
    if isinstance(module,nn.GRU):
        if GRU_index == 1:
            weight=module.weight_ih_l0
        else:
            weight=module.weight_hh_l0
    else:
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



def magnitude_prune_unstruct(model,loader,prune_amount,parameters_to_prune,erk, combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    model.train()
    for module in generate_mod_list(model):
        if isinstance(module, nn.Conv2d)or isinstance(module, nn.Linear) or isinstance(module,nn.Conv1d):
            importance_score.append(torch.abs(module.weight))
            #importance_score.append(torch.norm(module.weight,p=1,keepdim=True))
        elif isinstance(module,nn.GRU):
            importance_score.append(torch.abs(module.weight_ih_l0))
            importance_score.append(torch.abs(module.weight_hh_l0))   
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
                     
    else:
        split_val = torch.quantile(score,prune_amount)
        
        for imp_score in importance_score:
            fine_mask.append(torch.where(imp_score <= split_val, 0.0,1.0))  
    """
    for m in fine_mask:
        print(m.size())
    """
    #count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, masked_score



def apply_mask(model, model_mask,struct_mask = None):
    #print(model)
    mydict = model.state_dict()
    layer_names = list(mydict)
    #print(layer_names)
    layer_names = [name for name in layer_names if (name.find('bn') == -1 and name.find('bias') == -1)]
    #print(layer_names)
    mods=[]
    i = 0
    if "weight" in layer_names[0]: 
        w_ln = 0
        b_ln = 1
    else: 
        w_ln = 1
        b_ln = 0
    for module in generate_mod_list(model):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or (isinstance(module, nn.Linear) and module.out_features != 2):
            mask = model_mask[i]
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


def struct_act_penalty(activation,device = "cuda"):
    penalty = torch.tensor(0.0).to(device)
    
    for layer_act in activation:
        if len(layer_act.size()) == 4:
            act = layer_act.sum(dim=0)
            for i in act:
                penalty += torch.mean(torch.abs(i))
            #print(penalty)
        elif len(layer_act.size()) == 2:
            layer_act = torch.abs(layer_act)
            penalty += layer_act.sum(dim=0).sum()
            #print(penalty)
            
    return penalty


def Prune(model, train_loader, prune_amount, prune_mechanism, saved_model_name,erk, device = "cuda"):
    #print("PARAM COUNT BEFORE PRUNE - ",count_parameters(model)) 
    parameters_to_prune = prune_model_structured_erdos_renyi_kernel(model,prune_amount)
    #print(parameters_to_prune)
    
    if prune_mechanism == "MP_unstruct":
        #print("Prune_mechanism : Magnitude Based")
        model_mask, struct_mask, masked_score = magnitude_prune_unstruct(model,None,prune_amount,parameters_to_prune,erk,device = device)
    elif prune_mechanism == "Random":
        model, model_mask = random_prune_pytorch(model,parameters_to_prune,device = device)
        struct_mask = None
    else:
        print("Other Pruning Mechanism")
        exit()
    
    model = apply_mask(model, model_mask,struct_mask)
    
    return model, model_mask, struct_mask
