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
#from model_GRU import featureExtractor, metaCLF

from BiLSTM_model import BiLSTM

import warnings
warnings.filterwarnings("ignore")

    

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

def gradients(model, tensor_loader, criterion, device):
    #model.eval()
    test_acc = 0
    test_loss = 0
    test_bal_acc =0
    test_f1 = 0
    criterion = nn.CrossEntropyLoss()
    grad_sum = 0
    for data in tensor_loader:
        inputs, labels = data
        batch_size = inputs.size(0)
        inputs = torch.transpose(inputs, 1, 2)
        #print(inputs.size())
        #inputs = torch.transpose(inputs, 1, 2)[:,:,:,:8]
        #inputs = inputs.view(inputs.size(0),10, inputs.size(1), inputs.size(3))[:,:,:,:8]
        inputs = inputs.to(device)
        #print(inputs.size())
        #print(labels)
        labels = labels.type(torch.LongTensor).to(device)
        #labels_np = labels.numpy()
        outputs = model(inputs.float())
        outputs = outputs.type(torch.FloatTensor)
        outputs = outputs.to(device)
        
        loss = criterion(outputs,labels) 
        loss.backward()
        
        for module in model.modules():
            if isinstance(module, nn.LSTM):
                #grad_sum += torch.sum(torch.abs(module.weight.grad)).item()
                #grad_sum += torch.sum(module.weight.grad).item()
                grad_sum += torch.sum(torch.abs(module.weight_ih_l0.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_hh_l0.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_ih_l1.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_hh_l1.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_ih_l2.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_hh_l2.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_ih_l3.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_hh_l3.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_ih_l0_reverse.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_hh_l0_reverse.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_ih_l1_reverse.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_hh_l1_reverse.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_ih_l2_reverse.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_hh_l2_reverse.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_ih_l3_reverse.grad)).item()
                grad_sum += torch.sum(torch.abs(module.weight_hh_l3_reverse.grad)).item()
            if isinstance(module, nn.Linear):
                grad_sum += torch.sum(torch.abs(module.weight.grad)).item()
                #grad_sum += torch.sum(module.weight.grad).item()
        
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
    #return float(test_acc),float(test_bal_acc),float(test_f1), float(test_loss)
    return grad_sum/batch_size

def balanced_loader(Dataset,batch_size):
    #print(Dataset)
    
    if isinstance(Dataset, torch.utils.data.dataset.Subset):
        target = Dataset.dataset.label#[Dataset.indices]
        indices = Dataset.indices
        target = list(map(target.__getitem__, indices))
        #print(len(target))
    else:
        target = Dataset.label
        #print(len(target))
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    loader = DataLoader(
        Dataset, batch_size=batch_size, sampler=sampler)
    return loader
    
def main(user_id = '32', seed = 1013, model_name = "global"):    
    #seed = 1233
    batch_size = 2
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # user id for personalization : {106, 29, 32, 41, 45, 56, 71, 73, 82}

    #user_id = '41'
    accuracies = []
    
    dataset= Speech_Dataset("personalize_train",user_id)

    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - valid_size - train_size
    generator1=torch.Generator().manual_seed(seed)

    #print(len(dataset))
    #print(train_size)
    #print(valid_size)
    #print(test_size)
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size],generator=generator1)
    
    
    loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_size, shuffle=True)

    model = BiLSTM()
    
    
    
    #print("Balanced Accuracy : {}".format(train_bal_acc))
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("global_model_bilstm.pt"))
    model.to(device)
    _,global_bal_acc,_,_ = test(model,loader,criterion,device)
    #global_grad_sum = gradients(model,loader,criterion,device)
    model.load_state_dict(torch.load("saved_models_original/personal_"+str(user_id)+"_bilstm_original.pt"))
    model.to(device)
    _,personal_bal_acc_original,_,_ = test(model,loader,criterion,device)
    #original_grad_sum = gradients(model,loader,criterion,device)
    model.load_state_dict(torch.load("saved_models/personal_"+str(user_id)+"_bilstm.pt"))
    model.to(device)
    _,personal_bal_acc,_,_ = test(model,loader,criterion,device)
    #personal_grad_sum = gradients(model,loader,criterion,device)
    model.load_state_dict(torch.load("saved_models/personal_"+str(user_id)+"_bilstm_mixed.pt"))
    model.to(device)
    _,mixed_bal_acc,_,_ = test(model,loader,criterion,device)
    #mixed_grad_sum = gradients(model,loader,criterion,device)
    model.load_state_dict(torch.load("saved_models/personal_"+str(user_id)+"_bilstm_our.pt"))
    model.to(device)
    _,our_bal_acc,_,_ = test(model,loader,criterion,device)
    #our_grad_sum = gradients(model,loader,criterion,device)
    
    print("{},{},{},{},{},{}".format(user_id,round(global_bal_acc*100, 2),round(personal_bal_acc_original*100, 2),round(personal_bal_acc*100, 2),round(mixed_bal_acc*100, 2),round(our_bal_acc*100, 2)))
    #print("{},{},{},{},{},{}".format(user_id,round(global_grad_sum, 2),round(original_grad_sum, 2),round(personal_grad_sum, 2),round(mixed_grad_sum, 2),round(our_grad_sum, 2)))
    
    
        
    
if __name__ == "__main__":
    #print(sys.argv)
    main(sys.argv[1],sys.argv[2], sys.argv[3])