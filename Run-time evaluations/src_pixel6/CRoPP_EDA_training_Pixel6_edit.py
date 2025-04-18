from poplib import CR
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import time
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import psutil
import os
import subprocess
import copy

class MyDataset(Dataset):
    def __init__(self,X,Y):
        self.data=X
        self.label=Y
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x=self.data[idx]
        y=self.label[idx]
        
        x = torch.tensor(x)
        x=x.to(torch.float)
        
        return x,y
    
    def get_labels(self): return self.label




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
    
class BiLSTM(nn.Module):    
        #input for batch_first = True is [batch size, sequence length, input size]
        #my input is [64, 60, 5] 
        #output for batch_first = True is [batch size, sequence length, dimensions (bidirectional or unidirectional)*hidden size]
        #so my output is [64, 60, 2*hidden_sizelstm]
        #except it's flattened first to [64, 2*hidden_sizelstm)]            
               
        def __init__(self,device):
            super(BiLSTM, self).__init__()
            self.n_features = 5
            self.n_layersfc = 4
            self.n_layerslstm = 4
            self.hidden_sizefc = 64
            self.hidden_sizelstm = 64
            self.dropout = 0.0
            self.activation_function = torch.nn.Hardswish() #
            self.device = device
            self.nGPUs = 1
            if self.nGPUs == 0:
                self.nGPUs = 1 #just to avoid divide by zero in line 74
            
            linlayers_lstm = []
        
            self.lstmlayers = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_sizelstm, num_layers=self.n_layerslstm, bidirectional=True, dropout=self.dropout, batch_first=True)

            self.insize_lstm_fc = self.hidden_sizelstm*2 #2 for bidirectional
            for i in range(self.n_layersfc):
                linlayers_lstm.append(nn.Linear(in_features= self.insize_lstm_fc, out_features=self.hidden_sizefc))
                linlayers_lstm.append(self.activation_function)
                #linlayers_lstm.append(nn.Dropout(self.dropout))
                            
                self.insize_lstm_fc = self.hidden_sizefc #the output of the second linear layer needs to match the output of the first
            linlayers_lstm.append(nn.Linear(self.hidden_sizefc, 2)) #this is the final binary classification layer
            self.fc = nn.Sequential(*linlayers_lstm).to(self.device)
            
        def __init_hidden(self, inputs):
            h0 = torch.zeros(self.n_layerslstm * 2, inputs.size(0), self.hidden_sizelstm, device=self.device)
            c0 = torch.zeros(self.n_layerslstm * 2, inputs.size(0), self.hidden_sizelstm, device=self.device)
            return h0, c0


        def forward(self, inputs):
            batch_size = inputs.size(0)
            h0, c0 = self.__init_hidden(inputs)
            #print(f"shape heading into BiLSTM: {inputs.shape}")
            self.lstmlayers.flatten_parameters()
            out, (h0, c0) = self.lstmlayers(inputs, (h0, c0))
            out = out[:, -1, :] # only want the output of the last sequence
            #print(f"shape out of BiLSTM, last sequence only: {out.shape}")
            out = out.reshape(batch_size//self.nGPUs, -1)
            #print(f"shape heading into FC layers: {out.shape}")
            preds = self.fc(out)
            #print(f"shape of preds: {preds.shape}")
            return preds
        
def get_device_info():
    if device==torch.device("cuda:0"):
        print("CUDA Device Info:")
        # print(f"CUDA Device Count: {torch.cuda.device_count()}")
        # for i in range(torch.cuda.device_count()):
        #     print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        #     print(f"Memory Allocated: {torch.cuda.memory_allocated(i)}")
        #     print(f"Memory Cached: {torch.cuda.memory_reserved(i)}")
        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(f"Total GPU Memory: {info.total / (1024 ** 3):.2f} GB ({info.total / (1024 ** 2):.2f} MB), Free GPU Memory: {info.free / (1024 ** 3):.2f} GB ({info.free / (1024 ** 2):.2f} MB), Used GPU Memory: {info.used / (1024 ** 3):.2f} GB ({info.used / (1024 ** 2):.2f} MB)")
    else:
        print("Using CPU")
        cpu_info = subprocess.run(['lscpu'], capture_output=True, text=True).stdout
        for line in cpu_info.split("\n"):
            if "Architecture" in line or "Model name" in line:
                print(line.strip())

def print_training_stats(_time, energy_consumption, process_memory_info,avg_percentage,mode):
    
    used_memory_gb = process_memory_info / (1024 ** 3)
    used_memory_mb = process_memory_info / (1024 ** 2)
    print(device,mode)
    print(f"Training Time: {_time} seconds")
    print(f"Used Memory: {used_memory_gb:.2f} GB ({used_memory_mb:.2f} MB)")
    print(f"Energy Consumption: {energy_consumption} Joules")
    print(f"Utlization: {avg_percentage} %")

def print_inference_stats(_time, energy_consumption, process_memory_info,mode):
    
    used_memory_gb = process_memory_info / (1024 ** 3)
    used_memory_mb = process_memory_info / (1024 ** 2)
    print(device,mode)
    print(f"Inference Time: {_time} seconds")
    print(f"Used Memory: {used_memory_gb:.2f} GB ({used_memory_mb:.2f} MB)")
    print(f"Energy Consumption: {energy_consumption:.2f} Joules")

def measure_energy_consumption(start_func, end_func):
    def decorator(func):
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            if device==torch.device("cuda:0"):
                print('Using GPU')
                # pynvml.nvmlInit()
                # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                # start_percentage= pynvml.nvmlDeviceGetUtilizationRates(handle)
                # start_energy = pynvml.nvmlDeviceGetPowerUsage(handle)
            else:
                
                start_percentage = process.cpu_percent(interval=None)/psutil.cpu_count()
                cpu_power_watts=105
                
            start_time = start_func()
            result = func(*args, **kwargs)
            end_time = end_func()
            
            
            # Calculate energy consumption in Joules
            if device==torch.device("cuda:0"):
                print('CUDA')
                # end_energy = pynvml.nvmlDeviceGetPowerUsage(handle)
                # energy_consumption = (end_energy - start_energy) * (end_time - start_time) / 1000
                # end_percentage= pynvml.nvmlDeviceGetUtilizationRates(handle)
                # avg_percent=(start_percentage.gpu+end_percentage.gpu)/2
                # pynvml.nvmlShutdown()
            else:
                end_percentage = process.cpu_percent(interval=None)/psutil.cpu_count()
                avg_percent = (start_percentage + end_percentage) / 2
                energy_consumption = (avg_percent / 100) * cpu_power_watts * (end_time - start_time) 



            process_memory_info = process.memory_info().rss  # Resident Set Size (RSS) in bytes
            return start_time, end_time, energy_consumption, process_memory_info,avg_percent
        return wrapper
    return decorator


@measure_energy_consumption(time.time, time.time)
def train(model,best_model, train_loader, num_epochs, criterion, device,user_id, mode = 'init'):
    
    model = model.to(device)
    batch_num = 0
    best_model = copy.deepcopy(model)

    best_epoch = 0
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            batch_num += 1
            inputs,labels = data

            labels = labels.type(torch.LongTensor)
            inputs=inputs.type(torch.FloatTensor)
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.type(torch.FloatTensor)
            
            outputs = outputs.to(device)

            if mode == 'ft':
                loss = criterion(outputs,labels)
            else:
                loss = criterion(outputs,labels)
                
            loss.backward()
            optimizer.step()
        
            
    return model

@measure_energy_consumption(time.time, time.time)
def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    test_bal_acc =0
    test_f1 = 0
    for data in tensor_loader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor)

        outputs = model(inputs.float())
        outputs = outputs.type(torch.FloatTensor)
        outputs = outputs.to(device)
        
        predict_y = torch.argmax(outputs,dim=1)
        labels = labels.to(device)
        loss = criterion(outputs,labels)
        predict_y = predict_y.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
         
        # test_f1 += f1_score(labels,predict_y)
        # test_bal_acc += balanced_accuracy_score(labels,predict_y)
        # test_acc += accuracy_score(labels,predict_y)
        # test_loss += loss.item() * inputs.size(0)
        
    test_acc = test_acc/len(tensor_loader)
    test_f1 = test_f1/len(tensor_loader)
    test_bal_acc = test_bal_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)

    return float(test_acc),float(test_bal_acc),float(test_f1), float(test_loss)

device = torch.device("cpu")
get_device_info()
best_model = DNN(2)
model=DNN(2)
user_id = '17'
batch_size = 32
criterion = nn.CrossEntropyLoss()
num_epochs = 300

input_tensor = torch.randn(512, 34, dtype=torch.float32)
labels = torch.randint(0, 2, (512,), dtype=torch.long)
train_dataset=MyDataset(input_tensor,labels)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# infer_start, infer_end, infer_energy, infer_memory_info,infer_percent = test(model,train_loader,criterion,device)

# print_training_stats(infer_end-infer_start, infer_energy, infer_memory_info,infer_percent,'Generic Inference')

trials=[]

for i in range(5):
    train_start, train_end, train_energy, train_memory_info,avg_percent   = train(model,best_model, train_loader, num_epochs, criterion, device, user_id, 'init')

    infer_start, infer_end, infer_energy, infer_memory_info,infer_percent = test(model,train_loader,criterion,device)

    train_start2, train_end2, train_energy2, train_memory_info2,avg_percent2  = train(model,best_model, train_loader, 5, criterion, device, user_id, 'final')

    trials.append([train_end2-train_start, (train_energy+train_energy2)//2, (train_memory_info+train_memory_info2)//2,(avg_percent+avg_percent2)//2])

avg=np.mean(trials,axis=0)

print_training_stats(avg[0],avg[1],avg[2],avg[3],'Training')

# infer_start, infer_end, infer_energy, infer_memory_info,infer_percent = test(model,train_loader,criterion,device)

# print_training_stats(infer_end-infer_start, infer_energy, infer_memory_info,infer_percent,'Finetuned model Inference')



