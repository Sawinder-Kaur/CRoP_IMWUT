import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class featureExtractor(nn.Module):
    def __init__(self):
        super(featureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(1472, 512)
        self.rnn = nn.GRU(512, 256, batch_first=True)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.emb_layer = nn.Linear(128, 64)

    def forward(self, x, timestep=4): # input: batch_size x 6 x 800
        num, c, h = x.size()
        transformed_x = self.transform(x, timestep)
        transformed_x = transformed_x.reshape(num * timestep, c, h // timestep)

        out = self.conv1(transformed_x)
        out = F.max_pool1d(out, 2)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = F.max_pool1d(out, 2)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = F.max_pool1d(out, 2)
        out = self.bn3(out)
        out = F.relu(out, inplace=True)

        out = torch.flatten(out, start_dim=1)
        out = out.reshape(num, timestep, -1)
        out = self.fc1(out)
        out = F.relu(out, inplace=True)
        out, h = self.rnn(out)
        out = out[:, -1, :].reshape(num, -1)
        out = self.bn4(out)
        out = self.fc2(out)
        out = F.relu(out, inplace=True)
        out = self.emb_layer(out)
        return out

    def transform(self, x, time_step=4):
        num, c, h = x.size()
        step_size = h // time_step
        transformed_x = torch.empty(num, c, time_step, step_size)
        if (torch.cuda.is_available()):
            transformed_x = transformed_x.cuda()
        for i in range(time_step):
            transformed_x[:, :, i, :] = x[:, :, i * step_size:(i + 1) * step_size]
        transformed_x = transformed_x.transpose(1, 2)  # num, time_step, w, step_size
        return transformed_x

class metaCLF(nn.Module):
    def __init__(self, featureExtractor, emb_size, n_class):
        super(metaCLF, self).__init__()
        self.featureExtractor = featureExtractor
        self.decision = nn.Sequential(nn.Linear(emb_size, 32),
                                      nn.BatchNorm1d(32, track_running_stats=False),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(32, n_class),
                                      nn.LogSoftmax(dim=1))
    def forward(self, x): # x: batch_size x 6 x 800
        embs = self.featureExtractor(x)
        logits = self.decision(embs)
        return logits