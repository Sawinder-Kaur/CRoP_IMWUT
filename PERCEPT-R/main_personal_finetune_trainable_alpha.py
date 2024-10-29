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
from BiLSTM_model import BiLSTM, CostumLoss
from prune_utils import Prune, apply_mask
import itertools

import warnings
warnings.filterwarnings("ignore")

def import_global_weights(model, global_model, mask,device):
    global_model.to(device)
    toggled_mask = []
    for m in mask:
        ones = torch.ones_like(m, device = device)
        toggled_mask.append((ones - m).to(device))
        #print(m)
        #print(ones-m)
    masked_global_model = apply_mask(global_model, toggled_mask)
    
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
    for module, global_module in zip(model.modules(),masked_global_model.modules()):
        #print(layer_names[w_ln])
        
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or (isinstance(module, nn.Linear) and module.out_features != 2):
            #print(model.state_dict()[layer_names[w_ln]])
            model.state_dict()[layer_names[w_ln]].copy_(module.weight + global_module.weight)
            i = i + 1
            w_ln = w_ln+1
            b_ln = b_ln+1
        elif isinstance(module, nn.GRU):
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0 + global_module.weight_ih_l0)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0 + global_module.weight_hh_l0)
            w_ln = w_ln+2
            b_ln = b_ln+2
            i = i+2       
        elif isinstance(module, nn.LSTM):
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0 + global_module.weight_ih_l0)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0 + global_module.weight_hh_l0)
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0_reverse + global_module.weight_ih_l0_reverse)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0_reverse + global_module.weight_hh_l0_reverse)
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l1 + global_module.weight_ih_l1)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l1 + global_module.weight_hh_l1)
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l1_reverse + global_module.weight_ih_l1_reverse)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l1_reverse + global_module.weight_hh_l1_reverse)
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l2 + global_module.weight_ih_l2)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l2 + global_module.weight_hh_l2)
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l2_reverse + global_module.weight_ih_l2_reverse)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l2_reverse + global_module.weight_hh_l2_reverse)
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l3 + global_module.weight_ih_l3)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l3 + global_module.weight_hh_l3)
            w_ln = w_ln+2
            b_ln = b_ln+2
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l3_reverse + global_module.weight_ih_l3_reverse)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l3_reverse + global_module.weight_hh_l3_reverse)
            w_ln = w_ln+2
            b_ln = b_ln+2
            #w_ln = w_ln+32
            #b_ln = b_ln+32
            i = i +16
    return model
    


def train(model,best_model, train_loader,validation_loader,test_loader, num_epochs, criterion, device,user_id, mode = 'init'):
    model = model.to(device)
    #optimizer = optimizer
    batch_num = 0
    #alpha = 1 #5e-1 #1e-3 #1e-4
    train_losses = []
    valid_losses = []
    best_model = copy.deepcopy(model)
    _,best_valid_acc,_,_ = test(model,validation_loader,criterion,device)
    best_epoch = 0
    loss_fn = CostumLoss()
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), loss_fn.parameters()), lr=1e-5)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        print("******************Epoch : {} ********************".format(epoch))
        for data in train_loader:
            batch_num += 1
            inputs,labels = data
            #print(inputs.size())
            #inputs = torch.transpose(inputs, 1, 2)[:,:,:,:8]
            #print(inputs.size())
            #inputs = inputs.view(inputs.size(0),10, inputs.size(1), inputs.size(3))[:,:,:,:8]
            inputs = torch.transpose(inputs, 1, 2)
            inputs = inputs.to(device)
            #print(inputs.size())
            #inputs = inputs.type(torch.DoubleTensor)
            #print(inputs)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            outputs = model(inputs.float())
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            if mode == 'ft':
                loss = criterion(outputs,labels) #+ alpha * reg.regularizer(model,'l1', device)
            else:
                loss = loss_fn(outputs,labels, reg.regularizer(model,'l1', device),device)
            #print(loss)
            loss.backward()
            optimizer.step()
            #if model_mask != None: model = apply_mask(model, model_mask, struct_mask) 
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        
        #scheduler.step()
        train_accuracy,train_bal_acc,train_f1, train_loss = test(model,train_loader,criterion,device)
        print("Train accuracy : {}".format(train_bal_acc))
        valid_accuracy,valid_bal_acc,valid_f1, valid_loss = test(model,validation_loader,criterion,device)
        print("Validation accuracy : {}".format(valid_bal_acc))
        if valid_bal_acc > best_valid_acc:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_valid_acc = valid_bal_acc
        train_losses.append(train_bal_acc)
        valid_losses.append(valid_bal_acc)
    test_accuracy,test_bal_acc,test_f1, test_loss = test(best_model,test_loader,criterion,device)
    print("Test accuracy : {}".format(test_bal_acc))
    
    graph_name = "saved_models/User_"+user_id+"_"+mode+"_f1.png"
    plt.plot(np.array(train_losses), 'r')
    plt.plot(np.array(valid_losses), 'b')
    plt.axvline(x=best_epoch)
    plt.savefig(graph_name)
    plt.clf()
    
    if mode == 'final':
        return best_model
    else:
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
        #print(inputs.size())
        #inputs = torch.transpose(inputs, 1, 2)[:,:,:,:8]
        #inputs = inputs.view(inputs.size(0),10, inputs.size(1), inputs.size(3))[:,:,:,:8]
        #print(inputs.size())
        inputs = inputs.to(device)
        #print(inputs.size())
        #print(labels)
        labels = labels.type(torch.LongTensor)
        #labels_np = labels.numpy()
        outputs = model(inputs.float())
        outputs = outputs.type(torch.FloatTensor)
        outputs = outputs.to(device)
        
        predict_y = torch.argmax(outputs,dim=1)
        #predict_y_np = predict_y.numpy()
        #print(predict_y)
        labels = labels.to(device)
        #print(labels)
        #print(outputs)
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
    #print(f"validation accuracy: {float(test_acc)}, {float(test_loss)}, {float(test_bal_acc)}, {float(test_f1)}")
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
    
def main(user_id = '32', seed = 1013):    
    batch_size = 32
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)
    # user id for personalization : {106, 29, 32, 41, 45, 56, 71, 73, 82}

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    #user_id = '41'
    dataset= Speech_Dataset("personalize_train",user_id)

    #test_dataset= Speech_Dataset("personalized_test",0,user_id)

    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - valid_size - train_size
    generator1=torch.Generator().manual_seed(seed)

    #print(len(dataset))
    print(train_size)
    print(valid_size)
    print(test_size)
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size],generator=generator1)
    #"""
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=valid_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_size, shuffle=True)
    #"""
    
    
    model = BiLSTM()
    global_model = BiLSTM()
    best_model = BiLSTM()
    

    model.load_state_dict(torch.load("global_model_bilstm.pt"))
    global_model.load_state_dict(torch.load("global_model_bilstm.pt"))
    #"""
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    

    print("********* global model **********")
    train_accuracy,train_bal_acc,train_f1, train_loss = test(model,train_loader,criterion,device)
    print("Train F1 : {}".format(train_f1))
    valid_accuracy,valid_bal_acc,valid_f1, valid_loss = test(model,validation_loader,criterion,device)
    print("Validation F1 : {}".format(valid_f1))
    test_accuracy,test_bal_acc,test_f1, test_loss = test(model,test_loader,criterion,device)
    print("Test F1 : {}".format(test_f1))

    alpha = nn.Parameter(torch.tensor([1e-3],requires_grad=True, device=device))
    #optimizer = torch.optim.Adam(model.parameters(), lr = 1e-6)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), (alpha,)), lr = 1e-6)
    
    
    num_epochs = 300
   
    
    best_model = train(model,best_model, train_loader,validation_loader,test_loader, num_epochs, criterion, device, user_id, 'ft')
    #global_model = copy.deepcopy(model)
    
    torch.save(best_model.state_dict(), "saved_models_original/personal_"+str(user_id)+"_bilstm_original.pt")
    
    ### Step 1 - Train the Generic model on the personal data ####
    model.load_state_dict(torch.load("global_model_bilstm.pt"))
    best_model = train(model,best_model, train_loader,validation_loader,test_loader, num_epochs, criterion, device, user_id, 'init')
    #global_model = copy.deepcopy(model)
    
    torch.save(best_model.state_dict(), "saved_models/personal_"+str(user_id)+"_bilstm.pt")

    train_accuracy,train_bal_acc,train_f1,_ = test(best_model,train_loader,criterion,device)
    print("Balanced Train Accuracy : {}".format(train_bal_acc))

    print("********* finetuned model **********")
    train_accuracy,train_bal_acc,train_f1, train_loss = test(best_model,train_loader,criterion,device)
    print("Train accuracy : {}".format(train_bal_acc))
    valid_accuracy,valid_bal_acc,valid_f1, valid_loss = test(best_model,validation_loader,criterion,device)
    print("Validation accuracy : {}".format(valid_bal_acc))
    test_accuracy,test_bal_acc,test_f1, test_loss = test(best_model,test_loader,criterion,device)
    print("Test accuracy : {}".format(test_bal_acc))
    print("*****************************")
    
    #"""
    ### Step 2 - Tolerated Prune ####
    tolerance = 0.05
    prune_amount = 0.05
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
    
    
    ### Step 3 - Model Mixing ####
    if prev_mask != []:
        model = import_global_weights(model, global_model, prev_mask, device)
        
    torch.save(model.state_dict(), "saved_models/personal_"+str(user_id)+"_bilstm_mixed.pt")
     
    ### Step 4 - Train the Generic model on the personal data ####
    
    best_model = train(model,best_model, train_loader,validation_loader,test_loader, num_epochs, criterion, device, user_id, 'final')
    
    
    torch.save(best_model.state_dict(), "saved_models/personal_"+str(user_id)+"_bilstm_our.pt")

    train_accuracy,train_bal_acc,train_f1,_ = test(best_model,train_loader,criterion,device)
    print("Balanced Train Accuracy : {}".format(train_bal_acc))

    print("********* finetuned model **********")
    train_accuracy,train_bal_acc,train_f1, train_loss = test(best_model,train_loader,criterion,device)
    print("Train accuracy : {}".format(train_bal_acc))
    valid_accuracy,valid_bal_acc,valid_f1, valid_loss = test(best_model,validation_loader,criterion,device)
    print("Validation accuracy : {}".format(valid_bal_acc))
    test_accuracy,test_bal_acc,test_f1, test_loss = test(best_model,test_loader,criterion,device)
    print("Test accuracy : {}".format(test_bal_acc))
    print("*****************************")
    #"""
   
    
if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1],sys.argv[2])