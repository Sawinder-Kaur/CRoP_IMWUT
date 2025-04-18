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
            new_mask.append(torch.transpose(module.weight_mask,0,1))  
  
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

def gradient_prune(model,loader,parameters_to_prune, combine = "dot", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        batch += 1
        inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor).to(device)
        loss = criterion(outputs, targets)
        loss.backward()
    
        batch_score = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                b_score = module.weight * module.weight.grad 
                batch_score.append(b_score)
            elif isinstance(module, nn.Linear):
                b_score = module.weight * module.weight.grad 
                batch_score.append(b_score)
        if batch == 1:
            importance_score = batch_score
        else:
            sum_is = []
            for i,b in zip (importance_score,batch_score):
                sum_is.append(i+b)
            importance_score = sum_is
    """
    for imp_score in importance_score:
        print(imp_score.size())
    """
    i = 0
    ll_num = 0
    fine_mask = []
    struct_mask = []
    masked_score = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d): 
            #print(module.weight.size())
            kernel_score = []
            for i_score in importance_score[i]:
                if combine == "dot": kernel_score.append(torch.sum(i_score).item())#dot product of weight and gradient
                elif combine == "norm": kernel_score.append(torch.norm(i_score,2).item())    
            kernel_score = torch.tensor(kernel_score).to(device)
            #print(kernel_score)
            if bias == True: 
                kernel_score = kernel_score + (module.bias * module.bias.grad)
            #print(kernel_score)
            fine_mask_l, struct_mask_l = compute_mask(module, kernel_score, parameters_to_prune[i])
            fine_mask.append(fine_mask_l)
            struct_mask.append(struct_mask_l)
            masked_score.append(struct_mask_l * kernel_score)
            i += 1
        elif isinstance(module, nn.Linear):
            #print(module.weight.size())
            node_score = []
            if combine == "dot": node_score = torch.sum(importance_score[i],1)
            elif combine == "norm": 
                for i_score in importance_score[i]:
                    node_score.append(torch.norm(i_score,2).item())  
                node_score = torch.tensor(node_score).to(device)
                if bias == True: 
                    node_score = node_score + (module.bias * module.bias.grad)
            fine_mask_l, struct_mask_l = compute_mask(module, node_score, parameters_to_prune[i])
            fine_mask.append(fine_mask_l)
            struct_mask.append(struct_mask_l)
            masked_score.append(struct_mask_l * node_score)
            ll_num += 1
            
            i += 1
            
    """
    for m in fine_mask:
        print(m.size())
    """
    #torch.where(importance_score3[2] < split_val, 0.0, importance_score3[2].double())
    #model = apply_mask(model, fine_mask,struct_mask)
    count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, masked_score

def hybrid_AP_GP(model,loader,parameters_to_prune, combine = "dot", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        batch += 1
        inputs, targets = inputs.to(device), targets.to(device)
        outputs,imdt = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
    
        batch_score = []
        i = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                b_score = imdt[i].sum(dim = 0)
                batch_score.append(b_score)
                i += 1
            elif isinstance(module, nn.Linear):
                b_score = module.weight * module.weight.grad 
                batch_score.append(b_score)
        if batch == 1:
            importance_score = batch_score
        else:
            sum_is = []
            for i,b in zip (importance_score,batch_score):
                sum_is.append(i+b)
            importance_score = sum_is
    importance_score = [i/batch for i in importance_score]
    """
    for imp_score in importance_score:
        print(imp_score.size())
    """
    i = 0
    ll_num = 0
    fine_mask = []
    struct_mask = []
    masked_score = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d): 
            #print(module.weight.size())
            kernel_a_score = []
            for a_score in importance_score[i]:
                kernel_a_score.append(torch.mean(a_score).item())    
            kernel_a_score = torch.tensor(kernel_a_score).to(device)   
            
            kernel_score = kernel_a_score
            #print(kernel_score)
            fine_mask_l, struct_mask_l = compute_mask(module, kernel_score, parameters_to_prune[i])
            fine_mask.append(fine_mask_l)
            struct_mask.append(struct_mask_l)
            masked_score.append(struct_mask_l * kernel_score)
            i += 1
        elif isinstance(module, nn.Linear):
            #print(module.weight.size())
            node_score = []
            if combine == "dot": node_score = torch.sum(importance_score[i],1)
            elif combine == "norm": 
                for i_score in importance_score[i]:
                    node_score.append(torch.norm(i_score,2).item())  
                node_score = torch.tensor(node_score).to(device)
                if bias == True: 
                    node_score = node_score + (module.bias * module.bias.grad)
            fine_mask_l, struct_mask_l = compute_mask(module, node_score, parameters_to_prune[i])
            fine_mask.append(fine_mask_l)
            struct_mask.append(struct_mask_l)
            masked_score.append(struct_mask_l * node_score)
            ll_num += 1
            
            i += 1
            
    """
    for m in fine_mask:
        print(m.size())
    """
    #torch.where(importance_score3[2] < split_val, 0.0, importance_score3[2].double())
    #model = apply_mask(model, fine_mask,struct_mask)
    count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, masked_score


def activation_prune_soa(model,loader,parameters_to_prune, combine = "dot", prev_mask = None, bias = False,alpha = 0.9,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    activation_score = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        batch += 1
        activation_score_batch = []
        inputs, targets = inputs.to(device), targets.to(device)
        outputs,imdt_x = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        for i in imdt_x:
            if len(i.shape) == 4 or len(i.shape) == 2:
                #activation_score_batch.append(i.sum(dim = 0).sum(dim=1).sum(dim=1))
                activation_score_batch.append(i.sum(dim = 0))
        if batch == 1:
            activation_score = activation_score_batch
        else:
            sum_as = []
            for i,b in zip (activation_score,activation_score_batch):
                sum_as.append(i+b)
            activation_score = sum_as
    activation_score = [i/batch for i in activation_score]
    """
    for a_score in activation_score:    
        print(a_score.size())
    for i_score in importance_score:
        print(i_score.size())
    """
    #exit()
    """
    for imp_score in importance_score:
        print(imp_score.size())
    """
    i = 0
    ll_num = 0
    fine_mask = []
    struct_mask = []
    masked_score = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d): 
            kernel_a_score = []
            for a_score in activation_score[i]:
                kernel_a_score.append(torch.mean(a_score).item())    
            kernel_a_score = torch.tensor(kernel_a_score).to(device)   
            
            kernel_score = kernel_a_score
            #"""
            #print(kernel_score)
            #exit()
            
            fine_mask_l, struct_mask_l = compute_mask(module, kernel_score, parameters_to_prune[i])
            fine_mask.append(fine_mask_l)
            struct_mask.append(struct_mask_l)
            masked_score.append(struct_mask_l * kernel_score)
            i += 1
        elif isinstance(module, nn.Linear):
            #print(module.weight.size())
            node_score = activation_score[i]
            
            #print(node_score)
            
            fine_mask_l, struct_mask_l = compute_mask(module, node_score, parameters_to_prune[i])
            fine_mask.append(fine_mask_l)
            struct_mask.append(struct_mask_l)
            masked_score.append(struct_mask_l * node_score)
            ll_num += 1
            i += 1
            
    #exit()
    """
    for m in fine_mask:
        print(m.size())
    """
    #torch.where(importance_score3[2] < split_val, 0.0, importance_score3[2].double())
    #model = apply_mask(model, fine_mask,struct_mask)
    count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, masked_score

def hybrid_AP_MP(model,loader,parameters_to_prune, combine = "dot", prev_mask = None, bias = False,alpha = 0.9,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    activation_score = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        batch += 1
        activation_score_batch = []
        inputs, targets = inputs.to(device), targets.to(device)
        outputs,imdt_x = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        for i in imdt_x:
            if len(i.shape) == 4:
                #activation_score_batch.append(i.sum(dim = 0).sum(dim=1).sum(dim=1))
                activation_score_batch.append(i.sum(dim = 0))
        if batch == 1:
            activation_score = activation_score_batch
        else:
            sum_as = []
            for i,b in zip (activation_score,activation_score_batch):
                sum_as.append(i+b)
            activation_score = sum_as
    activation_score = [i/batch for i in activation_score]
    for module in model.modules():
        if isinstance(module, nn.Linear):
            activation_score.append(module.weight)
    """
    for a_score in activation_score:    
        print(a_score.size())
    for i_score in importance_score:
        print(i_score.size())
    """
    #exit()
    """
    for imp_score in importance_score:
        print(imp_score.size())
    """
    i = 0
    ll_num = 0
    fine_mask = []
    struct_mask = []
    masked_score = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d): 
            kernel_a_score = []
            for a_score in activation_score[i]:
                kernel_a_score.append(torch.mean(a_score).item())    
            kernel_a_score = torch.tensor(kernel_a_score).to(device)   
            
            kernel_score = kernel_a_score
            #"""
            #print(kernel_score)
            #exit()
            
            fine_mask_l, struct_mask_l = compute_mask(module, kernel_score, parameters_to_prune[i])
            fine_mask.append(fine_mask_l)
            struct_mask.append(struct_mask_l)
            masked_score.append(struct_mask_l * kernel_score)
            i += 1
        elif isinstance(module, nn.Linear):
            #print(module.weight.size())
            node_score = []
            for i_score in activation_score[i]:
                node_score.append(torch.norm(i_score,2).item())  
            node_score = torch.tensor(node_score).to(device)
            
            #print(node_score)
            
            fine_mask_l, struct_mask_l = compute_mask(module, node_score, parameters_to_prune[i])
            fine_mask.append(fine_mask_l)
            struct_mask.append(struct_mask_l)
            masked_score.append(struct_mask_l * node_score)
            ll_num += 1
            i += 1
            
    #exit()
    """
    for m in fine_mask:
        print(m.size())
    """
    #torch.where(importance_score3[2] < split_val, 0.0, importance_score3[2].double())
    #model = apply_mask(model, fine_mask,struct_mask)
    count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, masked_score


def magnitude_prune(model,loader,parameters_to_prune, combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            importance_score.append(module.weight)
        
    i = 0
    ll_num = 0
    fine_mask = []
    struct_mask = []
    masked_score = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d): 
            #print(module.weight.size())
            kernel_score = []
            for i_score in importance_score[i]:
                if combine == "dot": kernel_score.append(torch.sum(i_score).item())#dot product of weight and gradient
                elif combine == "norm": kernel_score.append(torch.norm(i_score,2).item())    
            kernel_score = torch.tensor(kernel_score).to(device)
            #print(kernel_score)
            if bias == True: 
                kernel_score = kernel_score + module.bias 
            #print(kernel_score)
            fine_mask_l, struct_mask_l = compute_mask(module, kernel_score, parameters_to_prune[i])
            fine_mask.append(fine_mask_l)
            struct_mask.append(struct_mask_l)
            masked_score.append(struct_mask_l * kernel_score)
            i += 1
        elif isinstance(module, nn.Linear):
            #print(module.weight.size())
            node_score = []
            if combine == "dot": node_score = torch.sum(importance_score[i],1)
            elif combine == "norm": 
                for i_score in importance_score[i]:
                    node_score.append(torch.norm(i_score,2).item())  
                node_score = torch.tensor(node_score).to(device)
                if bias == True: 
                    node_score = node_score + module.bias
            fine_mask_l, struct_mask_l = compute_mask(module, node_score, parameters_to_prune[i])
            fine_mask.append(fine_mask_l)
            struct_mask.append(struct_mask_l)
            masked_score.append(struct_mask_l * node_score)
            ll_num += 1
            i += 1
            
    """
    for m in fine_mask:
        print(m.size())
    """
    #torch.where(importance_score3[2] < split_val, 0.0, importance_score3[2].double())
    #model = apply_mask(model, fine_mask,struct_mask)
    count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, masked_score


def magnitude_prune_unstruct(model,loader,prune_amount, combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d)or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
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

def magnitude_prune_unstruct_gru(model,loader,prune_amount,parameters_to_prune,erk, combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
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
            elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
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
    count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, masked_score


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

def magnitude_prune_unstruct_reverse(model,loader,prune_amount, combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
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
        fine_mask.append(torch.where(imp_score <= split_val, 1.0,0.0))    
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

def gradient_prune_unstruct(model,loader,prune_amount, combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        batch += 1
        inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor).to(device)
        loss = criterion(outputs, targets)
        loss.backward()
    
        batch_score = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                b_score = torch.abs(module.weight * module.weight.grad) 
                #b_score = module.weight.grad
                batch_score.append(b_score)
        if batch == 1:
            importance_score = batch_score
        else:
            sum_is = []
            for i,b in zip (importance_score,batch_score):
                sum_is.append(i+b)
            importance_score = sum_is
    """
    i = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            importance_score[i] = torch.abs(module.weight * importance_score[i])
            i += 1
    #"""
    print(importance_score)
    fine_mask = []
    struct_mask = []
    masked_score = []
    score = []
    i = 0
    for imp_score in importance_score:
        if i == 0: score = torch.reshape(imp_score.data, (-1,))
        else : score = torch.cat((score,torch.reshape(imp_score.data, (-1,))),0)
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
    count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, masked_score

def gradient_prune_unstruct1(model,loader,prune_amount, combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        batch += 1
        inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor).to(device)
        loss = criterion(outputs, targets)
        loss.backward()
    
        batch_score = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                b_score = module.weight.grad
                #b_score = module.weight.grad
                batch_score.append(b_score)
        if batch == 1:
            importance_score = batch_score
        else:
            sum_is = []
            for i,b in zip (importance_score,batch_score):
                sum_is.append(i+b)
            importance_score = sum_is
    print(importance_score)
    i = 0        
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            importance_score[i] = torch.abs(module.weight * importance_score[i])
            #importance_score[i] = torch.abs(importance_score[i])
            i += 1
    """
    i = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            importance_score[i] = torch.abs(module.weight * importance_score[i])
            i += 1
    #"""
    print(importance_score)
    fine_mask = []
    struct_mask = []
    masked_score = []
    score = []
    i = 0
    for imp_score in importance_score:
        if i == 0: score = torch.reshape(imp_score.data, (-1,))
        else : score = torch.cat((score,torch.reshape(imp_score.data, (-1,))),0)
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
    count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, masked_score

def gradient_prune_unstruct_reverse(model,loader,prune_amount, combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        batch += 1
        inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor).to(device)
        loss = criterion(outputs, targets)
        loss.backward()
    
        batch_score = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                b_score = module.weight.grad
                #b_score = module.weight.grad
                batch_score.append(b_score)
        if batch == 1:
            importance_score = batch_score
        else:
            sum_is = []
            for i,b in zip (importance_score,batch_score):
                sum_is.append(i+b)
            importance_score = sum_is
    print(importance_score)
    i = 0        
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            importance_score[i] = torch.abs(module.weight * importance_score[i])
            #importance_score[i] = torch.abs(importance_score[i])
            i += 1
    """
    i = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            importance_score[i] = torch.abs(module.weight * importance_score[i])
            i += 1
    #"""
    print(importance_score)
    fine_mask = []
    struct_mask = []
    masked_score = []
    score = []
    i = 0
    for imp_score in importance_score:
        if i == 0: score = torch.reshape(imp_score.data, (-1,))
        else : score = torch.cat((score,torch.reshape(imp_score.data, (-1,))),0)
        i += 1
    split_val = torch.quantile(score,prune_amount)
    for imp_score in importance_score:
        fine_mask.append(torch.where(imp_score <= split_val, 1.0,0.0))    
    """
    for m in fine_mask:
        print(m.size())
    """
    #torch.where(importance_score3[2] < split_val, 0.0, importance_score3[2].double())
    #model = apply_mask(model, fine_mask,struct_mask)
    count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, masked_score


def hessian_prune_unstruct(model,loader,prune_amount, combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    
    params = []
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            params.append(module.weight)
    for batch_idx, (inputs, targets) in enumerate(loader):
        batch += 1
        inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
        #outputs = model(inputs)
        #outputs = outputs.type(torch.FloatTensor).to(device)
        #loss = criterion(outputs, targets)
        #loss.backward()
        #print(outputs)
        #print(targets)
        hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)
        print(hessian_comp)
        """
        h_input = (outputs,targets)
        hessian = torch.autograd.functional.hessian(CrossEntropyLoss,h_input)
        
        #env_grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
        loss.backward(retain_graph=True)
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                loss.backward(module.weight.grad)
        
        for i in range(env_grads.size(0)):
            for j in range(env_grads.size(1)):
                #hess_params[i, j] = torch.autograd.grad(env_grads[0][i][j], params, retain_graph=True)[0][i, j] #  <--- error here
                hess_params[i,j] = env_grads[0][i][j].backward()
        print( hess_params )        
        """
        
        batch_score = []
        #print(hessian)
        #print(hessian.size())
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                b_score = torch.abs(hessian) 
                #b_score = module.weight.grad
                batch_score.append(b_score)
        if batch == 1:
            importance_score = batch_score
        else:
            sum_is = []
            for i,b in zip (importance_score,batch_score):
                sum_is.append(i+b)
            importance_score = sum_is
    """
    i = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            importance_score[i] = torch.abs(module.weight * importance_score[i])
            i += 1
    #"""
    fine_mask = []
    struct_mask = []
    masked_score = []
    score = []
    i = 0
    for imp_score in importance_score:
        if i == 0: score = torch.reshape(imp_score.data, (-1,))
        else : torch.cat((score,torch.reshape(imp_score.data, (-1,))),0)
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
    count_parameters(model)
    #print(struct_mask)
    #print(fine_mask)
    return fine_mask, struct_mask, masked_score

def gradient_only_prune_unstruct(model,loader,prune_amount, combine = "norm", prev_mask = None, bias = False,device = "cuda"):
    
    criterion = CrossEntropyLoss()
    importance_score = []
    batch = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        batch += 1
        inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor).to(device)
        loss = criterion(outputs, targets)
        loss.backward()
    
        batch_score = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                #b_score = torch.abs(module.weight * module.weight.grad) 
                #b_score = torch.abs(module.weight.grad)
                b_score = module.weight.grad
                batch_score.append(b_score)
        if batch == 1:
            importance_score = batch_score
        else:
            sum_is = []
            for i,b in zip (importance_score,batch_score):
                sum_is.append(i+b)
            importance_score = sum_is
            
    #print(importance_score)
    i = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            importance_score[i] = torch.abs(importance_score[i])
            i +=1
    """
    i = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            importance_score[i] = torch.abs(module.weight * importance_score[i])
            i += 1
    #"""
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
    count_parameters(model)
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
            if "bias" in layer_names[w_ln]:
                w_ln = w_ln +1
                b_ln = b_ln +1
        elif isinstance(module, nn.BatchNorm2d):
            w_ln = w_ln+5
            b_ln = b_ln+5
        elif isinstance(module, nn.GRU):
            mask1 = model_mask[i]
            mask2 = model_mask[i+1]
            
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0 * mask1)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0 * mask2)
            w_ln = w_ln+4
            b_ln = b_ln+4
            i = i+2   
        elif isinstance(module, nn.LSTM):
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0 * model_mask[i])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0 * model_mask[i+1])
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0_reverse * model_mask[i+2])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0_reverse * model_mask[i+3])
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l1 * model_mask[i+4])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l1 * model_mask[i+5])
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l1_reverse * model_mask[i+6])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l1_reverse * model_mask[i+7])
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l2 * model_mask[i+8])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l2 * model_mask[i+9])
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l2_reverse * model_mask[i+10])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l2_reverse * model_mask[i+11])
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l3 * model_mask[i+12])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l3 * model_mask[i+13])
            w_ln = w_ln+4
            b_ln = b_ln+4
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l3_reverse * model_mask[i+14])    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l3_reverse * model_mask[i+15])
            w_ln = w_ln+4
            b_ln = b_ln+4
            
            
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

def struct_penalty(model,device = "cuda"):
    penalty = torch.tensor(0.0).to(device)
    for module in model.modules():
        
        if isinstance(module, nn.Conv2d):
            #print(module.weight.size())
            for w in module.weight:
                penalty += (torch.norm(w,2)+ torch.where(torch.norm(module.weight,0) == 0, 0.0, 1.0))
        elif isinstance(module, nn.Linear):
            #print(module.weight.size())
            penalty += torch.sum((torch.norm(module.weight,2,dim=1) + torch.where(torch.norm(module.weight,0,dim = 1) == 0, 0.0, 1.0)))
    #print(penalty)
    return penalty

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

def struct_penalty_detail(model,device = "cuda"):
    penalty = []
    for module in model.modules():
        
        if isinstance(module, nn.Conv2d):
            l2_penalty = []
            l0_penalty = []
            #print(module.weight.size())
            for w in module.weight:
                l2_penalty.append(torch.norm(w,2))
                l0_penalty.append(torch.where(torch.norm(module.weight,0) == 0, 0.0, 1.0))
            #print(l2_penalty)
            #print(l0_penalty)
            l2_penalty = torch.tensor(l2_penalty).to(device)
            l0_penalty = torch.tensor(l0_penalty).to(device)
            
            penalty.append(l2_penalty + l0_penalty)
        elif isinstance(module, nn.Linear):
            #print(module.weight.size())
            l2_penalty = torch.norm(module.weight,2,dim=1)
            l0_penalty = torch.where(torch.norm(module.weight,0) == 0, 0.0, 1.0)
            #print(l_penalty.size())
            penalty.append(l2_penalty+l0_penalty)
    #print(penalty)
    #penalty = torch.tensor(penalty).to('cuda')
    print(penalty)
    return penalty

def forward_hook_fn(module, input, output):
    if isinstance(module,nn.Conv2d):
        weight = module.weight * module.mask
        output = F.conv2d(
            input[0], weight, module.bias, module.stride, module.padding, module.dilation, module.groups
        )
        #print("conv hook implemeted")
    elif isinstance(module,nn.Linear):
        weight = module.weight * module.mask
        output = F.linear(input[0], weight, module.bias)
        #print("linear hook implemeted")
        
    return output

def mask_forward_only(model, model_mask):
    i = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or (isinstance(module, nn.Linear) and module.out_features != 6):
            module.register_buffer('mask', model_mask[i])
            i += 1
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or (isinstance(module, nn.Linear) and module.out_features != 6):
            module.register_forward_hook(forward_hook_fn)
            print("I am here")
            
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
    
    print("Before pruning")
    count_parameters(model)
    if prune_mechanism == "GP_struct":
        print("Prune_mechanism : Gradient Based -- L_2 norm")
        model_mask, struct_mask, masked_score  = gradient_prune(model,train_loader,parameters_to_prune, combine = "norm",device = device)
    elif prune_mechanism == "MP_struct":
        print("Prune_mechanism : Magnitude Based")
        model_mask, struct_mask, masked_score = magnitude_prune(model,None,parameters_to_prune,device = device)
    elif prune_mechanism == "MP_unstruct":
        print("Prune_mechanism : Magnitude Based")
        model_mask, struct_mask, masked_score = magnitude_prune_unstruct(model,None,prune_amount,device = device)
    elif prune_mechanism == "MP_unstruct_GRU":
        print("Prune_mechanism : Magnitude Based")
        model_mask, struct_mask, masked_score = magnitude_prune_unstruct_gru(model,None,prune_amount,parameters_to_prune, 0, device = device)
        
    elif prune_mechanism == "MP_unstruct_LSTM":
        print("Prune_mechanism : Magnitude Based")
        model_mask, struct_mask, masked_score = magnitude_prune_unstruct_lstm(model,None,prune_amount,parameters_to_prune, 0, device = device)
        
    elif prune_mechanism == "MP_unstruct_r":
        print("Prune_mechanism : Magnitude Based")
        model_mask, struct_mask, masked_score = magnitude_prune_unstruct_reverse(model,None,prune_amount,device = device)
    elif prune_mechanism == "GP_unstruct":
        print("Prune_mechanism : Gradient Based")
        model_mask, struct_mask, masked_score = gradient_prune_unstruct1(model,train_loader,prune_amount,device = device)
    elif prune_mechanism == "GP_unstruct_r":
        print("Prune_mechanism : Gradient Based")
        model_mask, struct_mask, masked_score = gradient_prune_unstruct_reverse(model,train_loader,prune_amount,device = device)
    

    #exit()
    #print(model_mask)
    model = apply_mask(model, model_mask,struct_mask) 
    
    """
    for mask in model_mask:
        print(float(torch.count_nonzero(mask))/torch.sum(torch.ones_like(mask)))
    mask_forward_only(model, model_mask)
    """
    print("After pruning")
    count_parameters(model)
    #saved_model_name = "trained_models/pruned_" + saved_model_name
    #torch.save({'state_dict': model.state_dict()}, saved_model_name)
    
    return model, model_mask, struct_mask