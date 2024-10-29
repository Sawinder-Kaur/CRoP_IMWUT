import numpy as np
import torch
import pandas as pd
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from data_loader import Speech_Dataset
from torch.utils.data import random_split, WeightedRandomSampler, DataLoader
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score
#from model_GRU import featureExtractor, metaCLF

from BiLSTM_model import BiLSTMwithAttention, BiLSTM, CNN, BiLSTM_relu
from estopping import EarlyStopping

import copy
import regularizer as reg

import warnings
warnings.filterwarnings("ignore")


def train(model,best_model, train_loader,validation_loader,test_loader, num_epochs, optimizer, criterion, device):
    model = model.to(device)
    optimizer = optimizer
    batch_num = 0
    best_model = copy.deepcopy(model)
    best_valid_acc = 0.0
    train_losses = []
    valid_losses = []
    alpha = 1e-4
    intermediate_model = "imdt_model.pt"
    early_stopping = EarlyStopping(patience=30, path = intermediate_model, verbose = True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        print("******************Epoch : {} ********************".format(epoch))
        for data in train_loader:
            batch_num += 1
            #print(data)
            #exit()
            inputs,labels = data
            inputs = inputs.to(device)
            inputs = torch.transpose(inputs, 1, 2)
            #print(inputs.size())
            #inputs = inputs.type(torch.DoubleTensor)
            #print(inputs)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            #print(labels)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            #print(outputs.size())
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            #print(labels.size())
            loss = criterion(outputs,labels) #+ alpha * reg.regularizer(model,'l1', device)
            #print(loss)
            loss.backward()
            optimizer.step()
            #if model_mask != None: model = apply_mask(model, model_mask, struct_mask) 
            epoch_loss += loss.item() * inputs.size(0)
            #print(labels)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        
        #scheduler.step()
        train_accuracy,train_bal_acc,train_f1, train_loss = test(model,train_loader,criterion,device)
        print("Train Balanced Accuracy : {}".format(train_bal_acc))
        valid_accuracy,valid_bal_acc,valid_f1, valid_loss = test(model,validation_loader,criterion,device)
        print("Validation Balanced Accuracy : {}".format(valid_bal_acc))
        """
        if valid_bal_acc > best_valid_acc:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_valid_acc = valid_bal_acc
        """
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            model.load_state_dict(torch.load(intermediate_model))
            test_accuracy,test_bal_acc,test_f1, test_loss = test(model,test_loader,criterion,device)
            print("Test Balanced Accuracy : {}".format(test_bal_acc))
            torch.save(model.state_dict(), "global_model.pt")
            print("Early stopping")
            break
        train_losses.append(train_bal_acc)
        valid_losses.append(valid_bal_acc)
    test_accuracy,test_bal_acc,test_f1, test_loss = test(model,test_loader,criterion,device)
    print("Test Balanced Accuracy : {}".format(test_bal_acc))
    
    plt.plot(np.array(train_losses), 'r')
    plt.plot(np.array(valid_losses), 'b')
    plt.savefig('global_acc.png')
    return model
    

def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    test_bal_acc =0
    test_f1 = 0
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        inputs = torch.transpose(inputs, 1, 2)
        labels = labels.type(torch.LongTensor)
        #labels_np = labels.numpy()
        outputs = model(inputs.float())
        outputs = outputs.type(torch.FloatTensor)
        outputs = outputs.to(device)

        predict_y = torch.argmax(outputs,dim=1)
        #predict_y_np = predict_y.numpy()
        
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
    #print(f"validation accuracy: {float(test_acc)}, {float(test_loss)}, {float(test_bal_acc)}, {float(test_f1)}")
    return float(test_acc),float(test_bal_acc),float(test_f1), float(test_loss)
    

def balanced_loader(Dataset,batch_size):
    #print(Dataset)
    
    if isinstance(Dataset, torch.utils.data.dataset.Subset):
        target = Dataset.dataset.label#[Dataset.indices]
        indices = Dataset.indices
        target = list(map(target.__getitem__, indices))
        #print(len(target))
    else:
        target = Dataset.label
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
    
seed = 1013
batch_size = 32

global_train_dataset= Speech_Dataset("global_train",0,0)
global_validate_dataset= Speech_Dataset("global_validate",0,0)
global_test_dataset= Speech_Dataset("global_test",0,0)

"""
### not balanced data loaders
train_size = int(0.7 * len(global_train_dataset))
valid_size = len(global_train_dataset) - train_size
generator1=torch.Generator().manual_seed(seed)

train_dataset, valid_dataset = random_split(global_train_dataset, [train_size, valid_size],generator=generator1)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=global_test_dataset, batch_size=batch_size, shuffle=True)
"""


### balanced data loaders
#train_data, train_target = global_train_dataset.data,global_train_dataset.label 
#print(len(global_train_dataset.label))

train_size = int(0.7 * len(global_train_dataset))
valid_size = len(global_train_dataset) - train_size
generator1=torch.Generator().manual_seed(seed)

train_dataset, valid_dataset = random_split(global_train_dataset, [train_size, valid_size],generator=generator1)

train_loader = balanced_loader(train_dataset,batch_size)
validation_loader = balanced_loader(valid_dataset,batch_size)
test_loader = balanced_loader(global_test_dataset,batch_size)

#model = BiLSTMwithAttention()
#best_model = BiLSTMwithAttention()

#model = BiLSTM()
#best_model = BiLSTM()

#model = CNN()
#best_model = CNN()


model = BiLSTM_relu()
best_model = BiLSTM_relu()


print(model)
#"""
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
best_model.to(device)

#optimizer = torch.optim.Adam(model.parameters(), lr = 1e-6)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
"""
optimizer = torch.optim.Adam([
            {'params': model.lstmlayers.parameters(), 'lr': 1e-3},
            {'params': model.attnlayer.parameters(), 'lr': 1e-2},
            {'params': model.linlayers.parameters(), 'lr': 3e-4},
            {'params': model.outputlayer.parameters(), 'lr': 3e-4}])

#"""
num_epochs = 300 #100

best_model = train(model, best_model,  train_loader,validation_loader,test_loader, num_epochs, optimizer, criterion, device)


print("-----------------------best model saved ----------------------------")
train_accuracy,train_bal_acc,train_f1, train_loss = test(best_model,train_loader,criterion,device)
print("Train Balanced Accuracy : {}".format(train_bal_acc))
valid_accuracy,valid_bal_acc,valid_f1, valid_loss = test(best_model,validation_loader,criterion,device)
print("Validation Balanced Accuracy : {}".format(valid_bal_acc))
test_accuracy,test_bal_acc,test_f1, test_loss = test(best_model,test_loader,criterion,device)
print("Test Balanced Accuracy : {}".format(test_bal_acc))



torch.save(best_model.state_dict(), "global_model_bilstm.pt")