import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
class Widar_LeNet(nn.Module):
    def __init__(self, num_classes):
        super(Widar_LeNet,self).__init__()
        self.encoder = nn.Sequential(
            #input size: (22,20,20)
            nn.Conv2d(22,32,6,stride=2),
            nn.ReLU(True),
            nn.Conv2d(32,64,3,stride=1),
            nn.ReLU(True),
            nn.Conv2d(64,96,3,stride=1),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(96*4*4,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = self.encoder(x)
        x = x.view(-1,96*4*4)
        #print(torch.count_nonzero(x))
        out = self.fc(x)
        return out
    
class Widar_LeNet_shot(nn.Module):
    def __init__(self, num_classes):
        super(Widar_LeNet_shot,self).__init__()
        self.encoder = nn.Sequential(
            #input size: (22,20,20)
            nn.Conv2d(22,32,6,stride=2),
            nn.ReLU(True),
            nn.Conv2d(32,64,3,stride=1),
            nn.ReLU(True),
            nn.Conv2d(64,96,3,stride=1),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(96*4*4,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = self.encoder(x)
        encoded = x
        x = x.view(-1,96*4*4)
        #print(torch.count_nonzero(x))
        out = self.fc(x)
        return out, encoded
    
class CostumLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_fn = nn.CrossEntropyLoss()
        self.gamma = nn.Parameter(torch.tensor([0.001]))

    def forward(self, out, ce_target, regularization,device):
        ce_loss = self.ce_fn(out, ce_target)
        loss = ce_loss + self.gamma.to(device) * regularization
        #print(self.gamma)
        return loss


class Widar_MLP(nn.Module):
    def __init__(self, num_classes):
        super(Widar_MLP,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(22*20*20,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = x.view(-1,22*20*20)
        x = self.fc(x)
        return x