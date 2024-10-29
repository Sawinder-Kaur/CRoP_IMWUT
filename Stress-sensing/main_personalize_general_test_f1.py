import numpy as np
import torch
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from data_loader import EEG_Dataset
from torch.utils.data import random_split, WeightedRandomSampler, DataLoader
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score
import copy
import sys
import regularizer as reg

from prune_utils import Prune, apply_mask

import warnings
warnings.filterwarnings("ignore")

class DNN(nn.Module):
    def __init__(self, num_classes):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(34, 128)  # Adjusted from 16*5*5 to 16*2*4
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
def import_global_weights(model, global_model, mask,device):
    
    toggled_mask = []
    for m in mask:
        ones = torch.ones_like(m, device = device)
        toggled_mask.append(ones - m)
        #print(m)
        #print(ones-m)
    masked_global_model = apply_mask(global_model, toggled_mask)
    
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
    for module, global_module in zip(model.modules(),masked_global_model.modules()):
        #print(layer_names[w_ln])
        
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or (isinstance(module, nn.Linear) and module.out_features != 6):
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
            
    return model

    
def train(model, train_loader,validation_loader,test_loader, num_epochs, optimizer, criterion, device,user_id, mode = 'init'):
    model = model.to(device)
    optimizer = optimizer
    batch_num = 0
    alpha = 1e-4
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        print("******************Epoch : {} ********************".format(epoch))
        for data in train_loader:
            batch_num += 1
            inputs,labels = data
            inputs = inputs.view(inputs.size(0),10, inputs.size(1), inputs.size(3))[:,:,:,:8]
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
            loss = criterion(outputs,labels) + alpha * reg.regularizer(model,'l1', device)
            #print(loss)
            loss.backward()
            optimizer.step()
            #if model_mask != None: model = apply_mask(model, model_mask, struct_mask) 
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        
        #scheduler.step()
        train_accuracy,train_bal_acc,train_f1, train_loss = test(model,train_loader,criterion,device)
        print("Balanced Train Accuracy : {}".format(train_bal_acc))
        valid_accuracy,valid_bal_acc,valid_f1, valid_loss = test(model,validation_loader,criterion,device)
        print("Balanced Validation Accuracy : {}".format(valid_bal_acc))
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
    test_accuracy,test_bal_acc,test_f1,_ = test(model,test_loader,criterion,device)
    print("Balanced Test Accuracy : {}".format(test_bal_acc))
    graph_name = "User_"+user_id+"_"+mode+".png"
    plt.plot(np.array(train_losses), 'r')
    plt.plot(np.array(valid_losses), 'b')
    plt.savefig(graph_name)
    plt.clf()
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
    
def main(user_id = '32', seed = 1013):    
    batch_size = 2
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # user id for personalization : {106, 29, 32, 41, 45, 56, 71, 73, 82}

    #user_id = '41'
    accuracies = []
    print("********* User_id : {} **********".format(user_id))
    test_phase = ['personalize_test_switch1', 'personalize_test_switch2']
    for t_phase in test_phase:
        dataset= EEG_Dataset(t_phase, user_id, "left", 'still')
        #dataset= EEG_Dataset(t_phase, user_id, "right", 'move')
        #print(len(dataset))
        loader = torch.utils.data.DataLoader(dataset,len(dataset))

        model = DNN(2)
        
        model.load_state_dict(torch.load("personal_"+str(user_id)+"_"+str(seed)+"_f1_mixed.pt"))
        
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        Result_dict = {'seed': seed, 'user_id': user_id}
        print("********* Phase : {} **********".format(t_phase))
        train_accuracy,train_bal_acc,train_f1,_ = test(model,loader,criterion,device)
        print("F1 score : {}".format(train_f1))
        accuracies.append(float('%.2f' % (train_f1)))
        
        
        model.load_state_dict(torch.load("personal_"+str(user_id)+"_"+str(seed)+"_f1_our.pt"))
        
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print("********* Phase : {} **********".format(t_phase))
        train_accuracy,train_bal_acc,train_f1,_ = test(model,loader,criterion,device)
        print("F1 score : {}".format(train_f1))
        accuracies.append(float('%.2f' % (train_f1)))
    print(accuracies)
    
if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1],sys.argv[2])