import numpy as np
import torch
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from data_loader import WIDAR_Dataset
from widar_model import *
from torch.utils.data import random_split
import sys
import os
import copy
import regularizer as reg

from prune_utils import Prune, apply_mask
import itertools

#best_model = {"train_acc":0.0,"valid_acc":0.0,"test_acc":0.0,"model_params":[],"epoch":-1}

def train(model, best_model, train_loader,validation_loader,test_loader, num_epochs, optimizer, criterion, device,user_id,global_id,model_save_path,data_for_graphs_path,graphs_path,best_models_path,training_set,data_to_plot = [], round_num = 0, total_rounds = 1, model_mask = None, struct_mask = None, regularizer_name = None, alpha = 0.01 ):
    model = model.to(device)
    #optimizer = optimizer
    #best_model = {"train_acc":0.0,"valid_acc":0.0,"test_acc":0.0,"model_params":[],"epoch":-1}
    batch_num = 0
    #torch.save(model.state_dict(), os.path.join(model_save_path,f"epoch_BASE.pt"))
    test_accuracy = test(model,test_loader,criterion,device)
    train_accuracy = test(model,train_loader,criterion,device)
    valid_accuracy = test(model,validation_loader,criterion,device)
    
    loss_fn = CostumLoss()
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), loss_fn.parameters()), lr=1e-6)
    #global best_model 
    best_model = {"train_acc":train_accuracy,"valid_acc":valid_accuracy,"test_acc":test_accuracy,"model_params":copy.deepcopy(model.state_dict()),"epoch":-1}
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
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
            if regularizer_name == 'None':
                loss = criterion(outputs,labels)
            else:
                loss = loss_fn(outputs,labels, reg.regularizer(model,'l1', device),device)
            """
            if regularizer_name != None:
                loss = loss + alpha * reg.regularizer(model,regularizer_name, device)
            """
            loss.backward()
            optimizer.step()
            #if model_mask != None: model = apply_mask(model, model_mask, struct_mask) 
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        
        #scheduler.step()
        epoch_loss = epoch_loss/len(train_loader.dataset)
        epoch_accuracy = epoch_accuracy/len(train_loader)
        valid_accuracy = test(model,validation_loader,criterion,device)
        test_accuracy = test(model,test_loader,criterion,device)
        print("Epoch : {}".format(epoch))
        print("Train accuracy : {}".format(epoch_accuracy))
        print("Validation accuracy : {}".format(valid_accuracy))
        print("Test accuracy : {}".format(test_accuracy))
        data_to_plot.append({"epoch":(round_num*num_epochs)+epoch+1,"acc":test_accuracy,"line_id":"test"})
        data_to_plot.append({"epoch":(round_num*num_epochs)+epoch+1,"acc":valid_accuracy,"line_id":"valid"})
        data_to_plot.append({"epoch":(round_num*num_epochs)+epoch+1,"acc":epoch_accuracy,"line_id":"train"})
        #### IMPORTANT SAVE STUFF WITH THE TEST/TRAIN LABEL BECAUSE WE NEED MORE THAN JUST GID/UID NOW, each of the 9 models is personalized in two different ways
        #if epoch % 5 == 0:
        #    torch.save(model.state_dict(), os.path.join(model_save_path,f"epoch_{epoch}.pt"))
        #if valid_accuracy> best_model["valid_acc"]:
        if valid_accuracy> best_model["valid_acc"]:
            best_model = {"train_acc":epoch_accuracy,"valid_acc":valid_accuracy,"test_acc":test_accuracy,"model_params":copy.deepcopy(model.state_dict()),"epoch":(round_num*num_epochs) + epoch}
    
    data_to_plot = pd.DataFrame(data_to_plot)
    #if (round_num + 1) == total_rounds : 
    best_epoch = best_model['epoch']
    torch.save(best_model["model_params"], os.path.join(best_models_path,f'gid_{global_id}_uid_{user_id}_set_{training_set}.pt'))
    torch.save(data_to_plot,os.path.join(data_for_graphs_path,f'gid_{global_id}_uid_{user_id}_set_{training_set}_data.pt'))
    ax = sns.lineplot(data=data_to_plot,x="epoch",y="acc",hue="line_id")
    plt.axvline(best_epoch+1, color='red')
    plt.savefig(os.path.join(graphs_path,f'gid_{global_id}_uid_{user_id}_set_{training_set}_graph.png'),dpi=300)
    plt.close()
    return model, best_model


def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    #print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return float(test_acc)


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
        if "bias" in layer_names[w_ln]:
            w_ln = w_ln +1
            b_ln = b_ln +1
        else:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or (isinstance(module, nn.Linear) and module.out_features != 6):
                #print(model.state_dict()[layer_names[w_ln]])
                model.state_dict()[layer_names[w_ln]].copy_(module.weight + global_module.weight)
                i = i + 1
                w_ln = w_ln+1
                b_ln = b_ln+1
            elif isinstance(module, nn.BatchNorm2d):
                w_ln = w_ln+5
                b_ln = b_ln+5
            
    return model

def main(
         EXTRACTED_PATH,SPLIT_PATH,EXPERIMENT_SAVE_PATH,
         user_id,global_id,model_path,training_set,learning_rate,epochs, total_rounds = 1,total_pruning_amount = 0.9, pruning_mechanism = "MP_unstruct", regularizer_name = None, alpha = 0.01, tolerance = 0.01):
    
    #torch.manual_seed(369284)
    #torch.manual_seed(96431)
    #torch.manual_seed(64140)
    #torch.manual_seed(660205)
    torch.manual_seed(658594)

    model_save_path = os.path.join(EXPERIMENT_SAVE_PATH,user_id,global_id)
    data_for_graphs_path = os.path.join(EXPERIMENT_SAVE_PATH,"data_and_graphs") 
    graphs_path = os.path.join(EXPERIMENT_SAVE_PATH,"data_and_graphs")
    best_models_path = os.path.join(EXPERIMENT_SAVE_PATH,"best_models")
    train_dataset=WIDAR_Dataset(SPLIT_PATH,EXTRACTED_PATH,user_id,"personal",global_id,training_set)
    train_size = int(0.7 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=64, shuffle=True)
    if training_set == "train":
        test_loader = torch.utils.data.DataLoader(dataset=WIDAR_Dataset(SPLIT_PATH,EXTRACTED_PATH,user_id,"personal",global_id,"test"), batch_size=64, shuffle=False)
    else:
        test_loader = torch.utils.data.DataLoader(dataset=WIDAR_Dataset(SPLIT_PATH,EXTRACTED_PATH,user_id,"personal",global_id,"train"), batch_size=64, shuffle=False)

    model =  Widar_LeNet(6)
    #model =  Widar_MLP(6)
    global_model = Widar_LeNet(6)
    model.load_state_dict(torch.load(model_path))
    global_model.load_state_dict(torch.load(model_path))
    train_epoch = epochs
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    global_model.to(device)
    saved_model_name = "./Pruned_model.pt"
    
    #global best_model 
    best_model = {"train_acc":0.0,"valid_acc":0.0,"test_acc":0.0,"model_params":copy.deepcopy(model.state_dict()),"epoch":0}
    
    #for buf in model.buffers():
    #    print(type(buf), buf.size())
    
    
    data_to_plot = []
    test_accuracy = test(model,test_loader,criterion,device)
    train_accuracy = test(model,train_loader,criterion,device)
    valid_accuracy = test(model,validation_loader,criterion,device)
    Result_dict = {'Gid': global_id, 'user_id': user_id, 'Prune_mechanism': pruning_mechanism, 'Regulalsarizer' : regularizer_name, 'alpha':alpha, 'tolerance' : tolerance}
    
    Result_dict.update({'Global Train': train_accuracy, 'Global Valid': valid_accuracy, 'Global Test': test_accuracy})
    
    data_to_plot.append({"epoch":0,"acc":test_accuracy,"line_id":"test"})
    data_to_plot.append({"epoch":0,"acc":train_accuracy,"line_id":"valid"})
    data_to_plot.append({"epoch":0,"acc":train_accuracy,"line_id":"train"})
    
    model_finetuned = Widar_LeNet(6)
    model_prev = Widar_LeNet(6)
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-6)
    model, best_model = train(
            model=model,
            best_model=best_model,
            train_loader= train_loader,
            test_loader = test_loader,
            validation_loader=validation_loader,
            num_epochs= 600,
            criterion=criterion,
            device=device,
            user_id=user_id,
            global_id=global_id,
            optimizer=optimizer,
            model_save_path=model_save_path,
            data_for_graphs_path=data_for_graphs_path,
            graphs_path=graphs_path,
            best_models_path=best_models_path,
            training_set=training_set,
            data_to_plot = data_to_plot,
            round_num = 0,
            total_rounds = total_rounds,
            regularizer_name = 'None', alpha = alpha
            )

    torch.save(best_model["model_params"], os.path.join(best_models_path,f'gid_{global_id}_uid_{user_id}_set_{training_set}_finetuned_ft.pt'))
    
    best_model_path_after_personalization = os.path.join(best_models_path,f'gid_{global_id}_uid_{user_id}_set_{training_set}.pt')
    
    model.load_state_dict(torch.load(best_model_path_after_personalization))

    test_accuracy = test(model,test_loader,criterion,device)
    train_accuracy = test(model,train_loader,criterion,device)
    valid_accuracy = test(model,validation_loader,criterion,device)
    
    Result_dict.update({'Personalized Train (FT)': train_accuracy, 'Personalized Valid (FT)': valid_accuracy, 'Personalized Test (FT)': test_accuracy})
    
    #### load global model again ######
    model.load_state_dict(torch.load(model_path))
    
    
    ### Step 1 - Train the Generic model on the personal data ####
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-6)
    model, best_model = train(
            model=model,
            best_model=best_model,
            train_loader= train_loader,
            test_loader = test_loader,
            validation_loader=validation_loader,
            num_epochs= 600,
            criterion=criterion,
            device=device,
            user_id=user_id,
            global_id=global_id,
            optimizer=optimizer,
            model_save_path=model_save_path,
            data_for_graphs_path=data_for_graphs_path,
            graphs_path=graphs_path,
            best_models_path=best_models_path,
            training_set=training_set,
            data_to_plot = data_to_plot,
            round_num = 0,
            total_rounds = total_rounds,
            regularizer_name = regularizer_name, alpha = alpha
            )
    
    torch.save(best_model["model_params"], os.path.join(best_models_path,f'gid_{global_id}_uid_{user_id}_set_{training_set}_finetuned1.pt'))
    
    best_model_path_after_personalization = os.path.join(best_models_path,f'gid_{global_id}_uid_{user_id}_set_{training_set}.pt')
    
    #model.load_state_dict(torch.load(best_model_path_after_personalization))
    
    test_accuracy = test(model,test_loader,criterion,device)
    train_accuracy = test(model,train_loader,criterion,device)
    valid_accuracy = test(model,validation_loader,criterion,device)
    
    Result_dict.update({'Personalized Train': train_accuracy, 'Personalized Valid': valid_accuracy, 'Personalized Test': test_accuracy})
    
    best_model = {"train_acc":0.0,"valid_acc":0.0,"test_acc":0.0,"model_params":[],"epoch":-1}
    optimizer = torch.optim.Adam(model.parameters(), lr = float(learning_rate))
    total_rounds = 1
    
    ### Step 2 - Tolerated Prune ####
    for round_num in range(total_rounds):
        
        #"""
        #prune_amount = (total_pruning_amount*(round_num+1))/total_rounds
        prune_amount = 0.05
        model_finetuned = copy.deepcopy(model)
        train_accuracy_orig = test(model,train_loader,criterion,device)
        train_accuracy_pruned = train_accuracy_orig
        
        prev_mask = []
        fine_mask = []
        while train_accuracy_pruned > (train_accuracy_orig - tolerance) and prune_amount < 0.95:
            print("Pruning amount : {}".format( prune_amount))
            prev_model = copy.deepcopy(model)
            prev_mask = fine_mask
            model, fine_mask, struct_mask = Prune(model, train_loader, prune_amount, pruning_mechanism, saved_model_name,device)
            prune_amount += 0.05
            train_accuracy_pruned = test(model,train_loader,criterion,device)
            print("Train accuracy of the finetuned model : {}".format(train_accuracy_orig))
            print("Train accuracy of the pruned model : {}".format(train_accuracy_pruned))
        model = copy.deepcopy(prev_model)
        
        ### Step 3 - Model Mixing ####
        if prev_mask != []:
            #print("here")
            model = import_global_weights(model, global_model, prev_mask, device)
        #"""
        
        torch.save(copy.deepcopy(model.state_dict()), os.path.join(best_models_path,f'gid_{global_id}_uid_{user_id}_set_{training_set}_mixed.pt'))
        test_accuracy = test(model,test_loader,criterion,device)
        train_accuracy = test(model,train_loader,criterion,device)
        valid_accuracy = test(model,validation_loader,criterion,device)
    
        best_model = {"train_acc":train_accuracy,"valid_acc":valid_accuracy,"test_acc":test_accuracy,"model_params":copy.deepcopy(model.state_dict()),"epoch":-1}
    
        if round_num == 0: Result_dict.update({'Mixed Train': train_accuracy, 'Mixed Valid': valid_accuracy, 'Mixed Test': test_accuracy})
            
        ### Step 4 - Train the Generic model on the personal data ####    
        model, best_model = train(
            model=model,
            best_model=best_model,
            train_loader= train_loader,
            test_loader = test_loader,
            validation_loader=validation_loader,
            num_epochs= int(train_epoch),
            criterion=criterion,
            device=device,
            user_id=user_id,
            global_id=global_id,
            optimizer=optimizer,
            model_save_path=model_save_path,
            data_for_graphs_path=data_for_graphs_path,
            graphs_path=graphs_path,
            best_models_path=best_models_path,
            training_set=training_set,
            data_to_plot = data_to_plot,
            round_num = round_num,
            total_rounds = total_rounds,
            regularizer_name = regularizer_name, alpha = alpha
            )
    ## here should magnitude prune then test!
    torch.save(best_model["model_params"], os.path.join(best_models_path,f'gid_{global_id}_uid_{user_id}_set_{training_set}_finetuned2.pt'))
    
    Result_dict.update({'Best Train': best_model["train_acc"], 'Best Valid': best_model["valid_acc"], 'Best Test': best_model["test_acc"], 'Epoch':best_model["epoch"]},index=[0])
    
    df = DataFrame(Result_dict)
    if training_set == "train":
        df.to_csv('Contex1_one_shot_hlr_aaai.csv', mode='a', index=False, header=False)
    elif training_set == "test":
        df.to_csv('Contex2_one_shot_hlr_aaai.csv', mode='a', index=False, header=False)
    return



if __name__ == "__main__":
    args = sys.argv
    main(args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8],args[9],int(args[10]),float(args[11]),args[12],args[13],float(args[14]), float(args[15]))
