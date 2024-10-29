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
import os
import itertools

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


    
class CostumLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_fn = nn.CrossEntropyLoss()
        self.gamma = nn.Parameter(torch.tensor([1e-4]))

    def forward(self, out, ce_target, regularization,device):
        ce_loss = self.ce_fn(out, ce_target)
        loss = ce_loss + self.gamma.to(device) * regularization
        #print(self.gamma)
        return loss

    
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
            if "bias" in layer_names[w_ln]:
                w_ln = w_ln +1
                b_ln = b_ln +1
        elif isinstance(module, nn.BatchNorm2d):
            w_ln = w_ln+5
            b_ln = b_ln+5
        elif isinstance(module, nn.GRU):
            model.state_dict()[layer_names[w_ln]].copy_(module.weight_ih_l0 + global_module.weight_ih_l0)    
            model.state_dict()[layer_names[w_ln+1]].copy_(module.weight_hh_l0 + global_module.weight_hh_l0)
            w_ln = w_ln+4
            b_ln = b_ln+4
            i = i+2       
            
    return model

def sum_weights(model):
    sum_ = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            sum_ = sum_ + torch.sum(module.weight)

def train(model,best_model, train_loader,validation_loader,test_loader, num_epochs, optimizer, criterion, device,user_id, mode = 'init'):
    model = model.to(device)
    #optimizer = optimizer
    batch_num = 0
    #alpha = 1e-4
    train_losses = []
    valid_losses = []
    
    loss_fn = CostumLoss()
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), loss_fn.parameters()), lr=1e-5)
    best_model = copy.deepcopy(model)
    _,_,best_valid_f1,_ = test(model,validation_loader,criterion,device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        print("******************Epoch : {} ********************".format(epoch))
        for data in train_loader:
            batch_num += 1
            inputs,labels = data
            #inputs = inputs.view(inputs.size(0),10, inputs.size(1), inputs.size(3))[:,:,:,:8]
            inputs = inputs.to(device)
            print(inputs.size())
            #inputs = inputs.type(torch.DoubleTensor)
            #print(inputs)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            outputs = model(inputs.float())
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            #loss = criterion(outputs,labels) + alpha * reg.regularizer(model,'l1', device)
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
        print("Train f1 : {}".format(train_f1))
        valid_accuracy,valid_bal_acc,valid_f1, valid_loss = test(model,validation_loader,criterion,device)
        print("Validation f1 : {}".format(valid_f1))
        if valid_f1 > best_valid_f1:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_valid_f1 = valid_f1
        train_losses.append(train_f1)
        valid_losses.append(valid_f1)
    test_accuracy,test_bal_acc,test_f1, test_loss = test(best_model,test_loader,criterion,device)
    print("Test f1 : {}".format(test_f1))
    
    graph_name = "User_"+user_id+"_"+mode+"_f1.png"
    plt.plot(np.array(train_losses), 'r')
    plt.plot(np.array(valid_losses), 'b')
    plt.savefig(graph_name)
    plt.clf()
    return model, best_model
    

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
    
def main(user_id = '32', seed = 1310):    
    #seed = 1233
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
    dataset= EEG_Dataset("personalize_train", user_id, "left", 'still')
    #dataset= EEG_Dataset("personalize_train", user_id, "right", 'move')

    #test_dataset= Speech_Dataset("personalized_test",0,user_id)

    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - valid_size - train_size
    generator1=torch.Generator().manual_seed(seed)

    #print(len(dataset))
    #print(train_size)
    #print(valid_size)
    #print(test_size)
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size],generator=generator1)
    #"""
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=valid_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_size, shuffle=True)
    
    model = DNN(2)
    best_model = DNN(2)
    global_model = DNN(2)
    
    model.load_state_dict(torch.load("global_model_dnn.pt"))
    global_model.load_state_dict(torch.load("global_model_dnn.pt"))
    #"""
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    Result_dict = {'seed': seed, 'user_id': user_id}

    print("********* global model **********")
    train_accuracy,train_bal_acc,train_f1, train_loss = test(model,train_loader,criterion,device)
    print("Train F1 : {}".format(train_f1))
    valid_accuracy,valid_bal_acc,valid_f1, valid_loss = test(model,validation_loader,criterion,device)
    print("Validation F1 : {}".format(valid_f1))
    test_accuracy,test_bal_acc,test_f1, test_loss = test(model,test_loader,criterion,device)
    print("Test F1 : {}".format(test_f1))

    Result_dict.update({'Global Train': train_f1*100, 'Global Valid': valid_f1*100, 'Global Test': test_f1*100})
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
    num_epochs = 500

    ### Step 1 - Train the Generic model on the personal data ####
    model, best_model = train(model,best_model, train_loader,validation_loader,test_loader, num_epochs, optimizer, criterion, device, user_id, 'init')
    #global_model = copy.deepcopy(model)
    
    #torch.save(best_model.state_dict(), "personal_"+str(user_id)+"_"+str(seed)+"_f1.pt")
    torch.save(model.state_dict(), "personal_"+str(user_id)+"_"+str(seed)+"_f1.pt")

    train_accuracy,train_bal_acc,train_f1,_ = test(model,train_loader,criterion,device)
    print("Balanced Train Accuracy : {}".format(train_bal_acc))

    print("********* finetuned model **********")
    train_accuracy,train_bal_acc,train_f1, train_loss = test(model,train_loader,criterion,device)
    print("Train F1 : {}".format(train_f1))
    valid_accuracy,valid_bal_acc,valid_f1, valid_loss = test(model,validation_loader,criterion,device)
    print("Validation F1 : {}".format(valid_f1))
    test_accuracy,test_bal_acc,test_f1, test_loss = test(model,test_loader,criterion,device)
    print("Test F1 : {}".format(test_f1))
    print("*****************************")
    
    Result_dict.update({'Finetuned Train': train_f1*100, 'Finetuned Valid': valid_f1*100, 'Finetuned Test': test_f1*100})
    #"""
    
    ### Step 2 - Tolerated Prune ####
    tolerance = 0.01
    prune_amount = 0.01
    prev_mask = []
    fine_mask = []
    #train_accuracy_pruned = train_bal_acc
    #train_accuracy_orig = train_bal_acc
    train_f1_pruned = train_f1
    train_f1_orig = train_f1
    #model = copy.deepcopy(best_model).to(device)
    while train_f1_pruned > (train_f1_orig - tolerance) and prune_amount < 0.95:
        print("Pruning amount : {}".format( prune_amount))
        prev_model = copy.deepcopy(model)
        prev_mask = fine_mask
        model, fine_mask, struct_mask = Prune(model, train_loader, prune_amount, "MP_unstruct", "pruned_model.pt",device)
        prune_amount += 0.01
        _,_,train_f1_pruned,_ = test(model,train_loader,criterion,device)
        print("Train F1 of the finetuned model : {}".format(train_f1_orig))
        print("Train F1 of the pruned model : {}".format(train_f1_pruned))
        model = copy.deepcopy(prev_model)

    ### Step 3 - Model Mixing ####
    if prev_mask != []:
        model = import_global_weights(model, global_model, prev_mask, device)

    print("********* mixed model **********")
    train_accuracy,train_bal_acc,train_f1, train_loss = test(model,train_loader,criterion,device)
    print("Train F1 : {}".format(train_f1))
    valid_accuracy,valid_bal_acc,valid_f1, valid_loss = test(model,validation_loader,criterion,device)
    print("Validation F1 : {}".format(valid_f1))
    test_accuracy,test_bal_acc,test_f1, test_loss = test(model,test_loader,criterion,device)
    print("Test F1 : {}".format(test_f1))
    print("*****************************")
    torch.save(model.state_dict(), "personal_"+str(user_id)+"_"+str(seed)+"_f1_mixed.pt")

    Result_dict.update({'Mixed Train': train_f1*100, 'Mixed Valid': valid_f1*100, 'Mixed Test': test_f1*100})
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
    
    
    ### Step 4 - Train the Generic model on the personal data ####
    
    model, best_model = train(model, best_model, train_loader,validation_loader,test_loader, num_epochs, optimizer, criterion, device, user_id, 'final')

    torch.save(best_model.state_dict(), "personal_"+str(user_id)+"_"+str(seed)+"_f1_our.pt")
    print("********* final model **********")
    train_accuracy,train_bal_acc,train_f1, train_loss = test(best_model,train_loader,criterion,device)
    print("Train F1 : {}".format(train_f1))
    valid_accuracy,valid_bal_acc,valid_f1, valid_loss = test(best_model,validation_loader,criterion,device)
    print("Validation F1 : {}".format(valid_f1))
    test_accuracy,test_bal_acc,test_f1, test_loss = test(best_model,test_loader,criterion,device)
    print("Test F1 : {}".format(test_f1))
    print("*****************************")
    Result_dict.update({'Final Train': train_f1*100, 'Final Valid': valid_f1*100, 'Final Test': test_f1*100},index=[0])
    #"""
    df = DataFrame(Result_dict)
    df.to_csv('Results_batch_size_32_f1.csv', mode='a', index=False, header=False)
    
if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1],sys.argv[2])