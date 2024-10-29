import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from data_loader import ExtraSensory_Dataset
from extrasensory_model import *
from torch.utils.data import random_split
import sys
import os
import torch.nn.utils.prune as prune
import ast
from prune_utils import Prune, apply_mask, generate_mod_list
import copy
import gc
import ast
import regularizer as reg
import warnings
import itertools

warnings.filterwarnings('ignore')
def warn(*args,**kwargs):
    pass
warnings.warn=warn
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score


class CostumLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_fn = nn.CrossEntropyLoss()
        self.gamma = nn.Parameter(torch.tensor([0.5]))

    def forward(self, out, ce_target, regularization,device):
        ce_loss = self.ce_fn(out, ce_target)
        loss = ce_loss + self.gamma.to(device) * regularization
        #print(self.gamma)
        return loss

def get_good_pruning(model, train_loader, acc_loss_tolerance, prune_method,erk, device,global_model):
    prune_amount = .05
    acc_threshhold2 = test(model,train_loader,device)[1]

    while(True):
        model_copy = copy.deepcopy(model)
        model_copy, copy_fine_mask,copy_struct_mask = Prune(model_copy, train_loader, prune_amount, prune_method,erk, device)
        train_accuracy,train_bal_acc,train_f1 = test(model_copy,train_loader,device)
        if (train_bal_acc < (acc_threshhold2-acc_loss_tolerance)):
            prune_amount -= .05
            model_final,fine_mask_final,struct_mask_final =  Prune(model, train_loader, prune_amount, prune_method,erk, device)
            ### Step 3 - Model Mixing ####
            if fine_mask_final != []:
                #print("IMPORT1")
                model_final = import_global_weights(model_final,global_model,fine_mask_final,device)
            return model_final     
        if (prune_amount + .05) > .99:
            ### Step 3 - Model Mixing ####
            if copy_fine_mask != []:
                #print("IMPORT2")
                model_final = import_global_weights(model_copy,global_model,copy_fine_mask,device)
            return model_final
        prune_amount+=.05

def train(model, train_loader,validation_loader,test_loader, num_epochs, optimizer, criterion, 
          device,user_id,data_for_graphs_path,test_context,prune_method,seed,total_rounds,round_num,
          data_to_plot,regularization_method,alpha):
    model = model.to(device)
    #optimizer = optimizer
    
    batch_num = 0
    test_accuracy,test_bal_acc,test_f1 = test(model,test_loader,device)
    train_accuracy,train_bal_acc,train_f1 = test(model,train_loader,device)
    valid_accuracy,valid_bal_acc,valid_f1 = test(model,validation_loader,device)
    
    loss_fn = CostumLoss()
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), loss_fn.parameters()), lr=1e-4)
    
    best_model = {"train_acc":train_accuracy,"valid_acc":valid_accuracy,"test_acc":test_accuracy,"valid_bal_acc":valid_bal_acc, "test_bal_acc":test_bal_acc,"model_params":copy.deepcopy(model.state_dict()),"epoch":0}
    
   
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        #print("EPOCH: ",(round_num*num_epochs)+epoch+1)
        epoch_accuracy = 0
        epoch_bal_acc = 0
        epoch_f1 = 0
        for data in train_loader:
            batch_num += 1
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            #loss = criterion(outputs,labels) #+ alpha * reg.regularizer(model,regularization_method,device)
            
            loss = loss_fn(outputs,labels, reg.regularizer(model,'l1', device),device)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).numpy()
            labels = labels.detach().cpu().numpy()
            epoch_accuracy += accuracy_score(labels,predict_y)
            epoch_bal_acc += balanced_accuracy_score(labels,predict_y)
            epoch_f1 += f1_score(labels,predict_y)

        test_accuracy,test_bal_acc,test_f1 = test(model,test_loader,device)
        valid_accuracy,valid_bal_acc,valid_f1 = test(model,validation_loader,device)
        if valid_bal_acc> best_model["valid_bal_acc"]:
            best_model = {"train_acc":epoch_accuracy,"valid_acc":valid_accuracy,"test_acc":test_accuracy,"valid_bal_acc":valid_bal_acc, "test_bal_acc":test_bal_acc,"model_params":copy.deepcopy(model.state_dict()),"epoch":(round_num*num_epochs)+epoch+1}
   

        epoch_loss = epoch_loss/len(train_loader.dataset)
        epoch_accuracy = epoch_accuracy/len(train_loader)
        epoch_bal_acc = epoch_bal_acc/len(train_loader)
        epoch_f1 = epoch_f1/len(train_loader)
        data_to_plot.append({"epoch":(round_num*num_epochs)+epoch+1,"acc":test_accuracy,"line_id":"test","f1":test_f1,"bal_acc":test_bal_acc})
        data_to_plot.append({"epoch":(round_num*num_epochs)+epoch+1,"acc":valid_accuracy,"line_id":"valid","f1":valid_f1,"bal_acc":valid_bal_acc})
        data_to_plot.append({"epoch":(round_num*num_epochs)+epoch+1,"acc":epoch_accuracy,"line_id":"train","f1":epoch_f1,"bal_acc":epoch_bal_acc})


    if (round_num+1) == total_rounds :
        data_to_plot = pd.DataFrame(data_to_plot)
        context_split = test_context.split("_")
        torch.save(data_to_plot,os.path.join(data_for_graphs_path,f'{user_id[0:2]}_{context_split[2]}_{prune_method}_{seed}_{alpha}_{regularization_method}.pt'))

    print("Available Context: {}".format(best_model["valid_bal_acc"]))
    print("Unseen Context: {}".format(best_model["test_bal_acc"]))
    
    return model, best_model, data_to_plot

def test(model, train_loader, device):
    model.eval()
    test_acc = 0
    test_bal_acc =0
    test_f1 = 0
    for data in train_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor)
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs = outputs.to(device)
        predict_y = torch.argmax(outputs,dim=1)
        labels = labels.to(device)
        predict_y = predict_y.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        test_f1 += f1_score(labels,predict_y)
        test_bal_acc += balanced_accuracy_score(labels,predict_y)
        test_acc += accuracy_score(labels,predict_y)
    
    test_acc = test_acc/len(train_loader)
    test_f1 = test_f1/len(train_loader)
    test_bal_acc = test_bal_acc/len(train_loader)
    return float(test_acc),float(test_bal_acc),float(test_f1)



def import_global_weights(model, global_model, mask,device):
    
    toggled_mask = []
    for m in mask:
        ones = torch.ones_like(m, device = device)
        toggled_mask.append(ones - m)
    masked_global_model = apply_mask(global_model, toggled_mask)

    

    mydict = model.state_dict()
    layer_names = list(mydict)
    layer_names = [name for name in layer_names if (name.find('bn') == -1 and name.find('bias') == -1)]

    if "weight" in layer_names[0]: 
        w_ln = 0
        b_ln = 1
    else: 
        w_ln = 1
        b_ln = 0
    modules_list = zip(generate_mod_list(model),generate_mod_list(masked_global_model))
    #print(list(modules_list))
    for module, global_module in zip(generate_mod_list(model),generate_mod_list(masked_global_model)):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or (isinstance(module, nn.Linear) and module.out_features != 2):
            #print(model.state_dict()[layer_names[w_ln]])
            model.state_dict()[layer_names[w_ln]].copy_(module.weight + global_module.weight)
            w_ln = w_ln+1
            b_ln = b_ln+1
        elif isinstance(module, nn.GRU):
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0 + global_module.weight_ih_l0)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0 + global_module.weight_hh_l0)
            w_ln = w_ln+2
            b_ln = b_ln+2
            
    return model


def main(test_context,seed,user_to_personalize):
    print("-------------------------------------------")
    print(user_to_personalize)
    print(test_context)
    print(seed)
    EXPERIMENT_SAVE_PATH = "./results"
    data_for_graphs_path = "./results"
    model_path = "gid_0_uid_0.pt"
    prune_method = "MP_unstruct"
    acc_loss_tolerance = 0.05
    erk = True
    regularization_method = 'l1'
    alpha = 0.5
    seed=int(seed)
    total_rounds = 1 
    train_dataset= ExtraSensory_Dataset("personalized_train",test_context,user_to_personalize)
    train_size = int(0.7 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    generator1=torch.Generator().manual_seed(seed)
    ## Validation and training user disjoint set <-- not needed if testing every epoch
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size],generator=generator1)
    test_dataset = ExtraSensory_Dataset("personalized_test",test_context,user_to_personalize)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True,drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=32, shuffle=True,drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=32, shuffle=False,drop_last=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FE=featureExtractor()    
    model = metaCLF(FE,64,2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    FE2=featureExtractor()    
    global_model = metaCLF(FE2,64,2)
    global_model.load_state_dict(torch.load(model_path))
    global_model.to(device)

    data_to_plot = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = float(1e-6))
    test_accuracy,test_bal_acc,test_f1 = test(model,test_loader,device)
    train_accuracy,train_bal_acc,train_f1 = test(model,train_loader,device)
    valid_accuracy,valid_bal_acc,valid_f1 = test(model,validation_loader,device)

    data_to_plot.append({"epoch":0,"acc":test_accuracy,"line_id":"test","f1":test_f1,"bal_acc":test_bal_acc})
    data_to_plot.append({"epoch":0,"acc":valid_accuracy,"line_id":"valid","f1":valid_f1,"bal_acc":valid_bal_acc})
    data_to_plot.append({"epoch":0,"acc":train_accuracy,"line_id":"train","f1":train_f1,"bal_acc":train_bal_acc})
    
    ### Step 1 - Train the Generic model on the personal data ####
    model, best_model, data_to_plot = train(
        model=model,
        train_loader= train_loader,
        test_loader = test_loader,
        validation_loader=validation_loader,
        num_epochs= int(600),
        criterion=criterion,
        device=device,
        user_id=user_to_personalize,
        optimizer=optimizer,
        data_for_graphs_path=data_for_graphs_path,
        test_context=test_context,
        prune_method=prune_method,
        seed=seed,
        total_rounds = total_rounds,
        round_num = 1,
        data_to_plot=data_to_plot,
        regularization_method=regularization_method,
        alpha=alpha,
        )
    
    ### Step 2 - Tolerated Prune ####    
    model = get_good_pruning(model,train_loader, acc_loss_tolerance, prune_method,erk, device,global_model)
    ### Step 4 - Train the Generic model on the personal data ####
    model, best_model, data_to_plot = train(
        model=model,
        train_loader= train_loader,
        test_loader = test_loader,
        validation_loader=validation_loader,
        num_epochs= int(1000),
        criterion=criterion,
        device=device,
        user_id=user_to_personalize,
        optimizer=optimizer,
        data_for_graphs_path=data_for_graphs_path,
        test_context=test_context,
        prune_method=prune_method,
        seed=seed,
        total_rounds = total_rounds,
        round_num = 1,
        data_to_plot=data_to_plot,
        regularization_method=regularization_method,
        alpha=alpha,
        )
    torch.save(best_model["model_params"], os.path.join("./saved_models/",f'seed_{seed}_uid_{user_to_personalize}_set_{test_context}_finetuned2.pt'))
    return


if __name__ == "__main__":
    args = sys.argv
    #main(args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8],args[9],args[10],args[11],args[12],args[13])
    main(test_context = args[1],seed = args[2],user_to_personalize = args[3])