# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 09:39:03 2021

@author: nrbenway
"""
import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Linear, ReLU
from math import floor
from torch.autograd import Variable
import torch.nn.functional as F

#def select(classifierType):
    
""" note: the sigmoid layer is in the criterion function in the main program
torch.nn.CrossEntropyLoss or nn.BCEWithLogitsLoss
For BCEWithLogitsLoss...
This loss combines a Sigmoid layer and the nn.NLLLoss in one single class. 
This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, 
by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability."""

"""note: running in dataparallel will run the forward function once on each GPU, meaning if print statements are turned on it will 
print the same line multiple times for each gpu. Also, any time you reshape involving BATCHSIZE, you will likely need
to write self.BATCHSIZE//self.nGPUs"""


class BiLSTMwithAttention(nn.Module): #multihead attention
    #input for batch_first = True is [batch size, sequence length, input size]
    #my input is [64, 60, 5] 
    #output for batch_first = True is [batch size, sequence length, dimensions (bidirectional or unidirectional)*hidden size]
    #so my output is [64, 60, 2*hidden_sizelstm]
    #except it's flattened first to [64, 2*hidden_sizelstm)]            
                   
    def __init__(self):
        super(BiLSTMwithAttention, self).__init__()
        #self.params = params
        self.multi = False #self.params['multi']
        self.BATCHSIZE = 32 #self.params['BATCHSIZE']
        self.n_features = 5 #self.params['n_features']
        self.seq_length = 60 #self.params['seq_length']
        self.n_layersfc = 3 #self.params['n_layersfc']
        self.n_layerslstm = 3 #self.params['n_layerslstm']
        self.hidden_sizefc = 32 #self.params['hidden_sizefc']
        self.hidden_sizelstm = 32 #self.params['hidden_sizelstm']
        self.dropout = 0.0 #self.params['dropout']
        self.activation_function = torch.nn.Hardswish() #self.params['activationfunction']
        self.device = "cuda" #self.params['device']
        self.nGPUs = torch.cuda.device_count()
        if self.nGPUs == 0:
            self.nGPUs = 1 #avoid divide by 0

        self.lstmlayers = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_sizelstm,
                                  num_layers=self.n_layerslstm, bidirectional=True,
                                  dropout=self.dropout if self.n_layerslstm > 1 else 0.0, batch_first=True)

        self.attnlayer = nn.MultiheadAttention(embed_dim=self.hidden_sizelstm * 2, num_heads=8, dropout=self.dropout,batch_first = True)

        linlayers = []
        self.insize_lstm_fc = self.hidden_sizelstm * 2 * self.seq_length  # 2 for bidirectional
        for i in range(self.n_layersfc):
            linlayers.append(nn.Linear(in_features=self.insize_lstm_fc, out_features=self.hidden_sizefc))
            nn.init.xavier_uniform_(linlayers[-1].weight)
            nn.init.zeros_(linlayers[-1].bias)
            linlayers.append(self.activation_function)
            linlayers.append(nn.Dropout(self.dropout))
            self.insize_lstm_fc = self.hidden_sizefc

        self.linlayers = nn.Sequential(*linlayers).to(self.device)

        if self.multi:
            self.outputlayer = nn.Linear(self.hidden_sizefc, 3)  # Multi-class classification layer
        else:
            self.outputlayer = nn.Linear(self.hidden_sizefc, 2)  # Binary classification layer

        nn.init.xavier_uniform_(self.outputlayer.weight)
        nn.init.zeros_(self.outputlayer.bias)

        self.outputlayer = self.outputlayer.to(self.device)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.n_layerslstm * 2, batch_size, self.hidden_sizelstm).to(self.device)
        c0 = torch.zeros(self.n_layerslstm * 2, batch_size, self.hidden_sizelstm).to(self.device)
        return h0, c0

    def forward(self, inputs):
        batch_size = inputs.size(0)  # Get the batch size from the inputs
        h0, c0 = self.init_hidden(batch_size)
        lstm_out, _ = self.lstmlayers(inputs, (h0, c0))
        #print(f"shape out of BiLSTM: {lstm_out.shape}")
        #lstm_out = lstm_out.transpose(0, 1)  # shape needed for attn - [seq_len, batch_size, hidden_dim * 2]
        #print(f"shape in to attn: {lstm_out.shape}")
        attn_out, attn_weights = self.attnlayer(lstm_out, lstm_out, lstm_out) #for self attention, 'query', 'key', and 'value' are the same
        #print(f"shape out of attn: {attn_out.shape}")
        #attn_out = attn_out.transpose(0, 1) # back to [batch_size, seq_len, hidden_dim * 2]
        #print(f"shape out of attn, transposed: {attn_out.shape}")
        attn_out = attn_out.contiguous().view(batch_size, -1)  #flatten for fc layers
        #print(f"shape in to fc layers: {attn_out.shape}")
        fc_out = self.linlayers(attn_out)
        logits = self.outputlayer(fc_out)
        
        return logits

    
class BiLSTM_relu(nn.Module):    
        #input for batch_first = True is [batch size, sequence length, input size]
        #my input is [64, 60, 5] 
        #output for batch_first = True is [batch size, sequence length, dimensions (bidirectional or unidirectional)*hidden size]
        #so my output is [64, 60, 2*hidden_sizelstm]
        #except it's flattened first to [64, 2*hidden_sizelstm)]            
               
        def __init__(self):
            super(BiLSTM_relu, self).__init__()
            self.n_features = 5
            self.n_layersfc = 4
            self.n_layerslstm = 4
            self.hidden_sizefc = 64
            self.hidden_sizelstm = 64
            self.dropout = 0.0
            self.activation_function = torch.nn.ReLU() #
            self.device = "cuda"
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
        
class CostumLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_fn = nn.CrossEntropyLoss()
        self.gamma = nn.Parameter(torch.tensor([1e-2]))

    def forward(self, out, ce_target, regularization,device):
        ce_loss = self.ce_fn(out, ce_target)
        loss = ce_loss + self.gamma.to(device) * regularization
        #print(self.gamma)
        return loss

        
        
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
        
        
class CNN(nn.Module):    
        #input for batch_first = True is [batch size, sequence length, input size]
        #my input is [64, 60, 5] 
        #output for batch_first = True is [batch size, sequence length, dimensions (bidirectional or unidirectional)*hidden size]
        #so my output is [64, 60, 2*hidden_sizelstm]
        #except it's flattened first to [64, 2*hidden_sizelstm)]            
               
        def __init__(self):
            super(CNN, self).__init__()
            self.n_features = 5
            self.n_layersfc = 2
            self.hidden_sizefc = 32
            self.dropout = 0.0
            self.activation_function = torch.nn.Hardswish() #torch.nn.ReLU()
            self.device = "cuda"
            self.nGPUs = 1
            if self.nGPUs == 0:
                self.nGPUs = 1 #just to avoid divide by zero in line 74
            
            self.conv1 = nn.Conv1d(5, 3, 5, stride=1)
            self.conv2 = nn.Conv1d(3, 3, 5, stride=1)
            self.conv3 = nn.Conv1d(3, 1, 5, stride=1)

            linlayers = []
            self.insize_fc = 52 #2 for bidirectional
            for i in range(self.n_layersfc):
                linlayers.append(nn.Linear(in_features= self.insize_fc, out_features=self.hidden_sizefc))
                linlayers.append(self.activation_function)
                #linlayers_lstm.append(nn.Dropout(self.dropout))
                            
                self.insize_fc = self.hidden_sizefc #the output of the second linear layer needs to match the output of the first
            linlayers.append(nn.Linear(self.hidden_sizefc, 2)) #this is the final binary classification layer
            self.fc = nn.Sequential(*linlayers).to(self.device)
            


        def forward(self, inputs):
            batch_size = inputs.size(0)
            #h0, c0 = self.__init_hidden(inputs)
            #print(f"shape heading into BiLSTM: {inputs.shape}")
            #self.lstmlayers.flatten_parameters()
            #out, (h0, c0) = self.lstmlayers(inputs, (h0, c0))
            out = self.conv1(inputs)
            #out = self.conv2(out)
            out = self.conv3(out)
            out = nn.Flatten()(out)
            #out = out[batch_size, -1] # only want the output of the last sequence
            #print(f"shape out of BiLSTM, last sequence only: {out.shape}")
            #out = out.reshape(batch_size//self.nGPUs, -1)
            #print(f"shape heading into FC layers: {out.shape}")
            preds = self.fc(out)
            #print(f"shape of preds: {preds.shape}")
            return preds

