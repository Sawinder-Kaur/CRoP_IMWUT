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
               
        def __init__(self):
            super(BiLSTM, self).__init__()
            self.n_features = 5
            self.n_layersfc = 4
            self.n_layerslstm = 4
            self.hidden_sizefc = 64
            self.hidden_sizelstm = 64
            self.dropout = 0.0
            self.activation_function = torch.nn.Hardswish() #
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
