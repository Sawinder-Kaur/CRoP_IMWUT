import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch

class Speech_Dataset(Dataset):
    def __init__(self, mode,user_to_personalize, personalize_test_phase = '0'):
        ### Modes: global_train, global_validate, personalized_train, personalized_test
        self.data = []
        self.label = []
        if mode == "global_train":
            with open("./data/global/train.pickle","rb") as f:
                pickle_data = pickle.load(f)
                for (u_data, u_label) in pickle_data:
                    self.data.append(u_data)
                    self.label.append(int(u_label))
                    
        elif mode == "global_validate":
            with open("./data/global/validate.pickle","rb") as f:
                pickle_data = pickle.load(f)
                for (u_data, u_label) in pickle_data:
                    self.data.append(u_data)
                    self.label.append(int(u_label))
        
        elif mode == "global_test":
            with open("./data/global/test.pickle","rb") as f:
                pickle_data = pickle.load(f)
                for (u_data, u_label) in pickle_data:
                    self.data.append(u_data)
                    self.label.append(int(u_label))
                    
        elif mode == "personalize_train":
            with open("./data/personal/"+str(user_to_personalize)+"/train.pickle","rb") as f:
                pickle_data = pickle.load(f)
                for (u_data, u_label) in pickle_data:
                    self.data.append(u_data)
                    self.label.append(int(u_label))
                    
        elif mode == "personalize_test":
            with open("./data/personal/"+str(user_to_personalize)+ '/'+ 'test.pickle',"rb") as f:
                pickle_data = pickle.load(f)
                for (u_data, u_label) in pickle_data:
                    self.data.append(u_data)
                    self.label.append(int(u_label))
                    
        elif mode == "personalize_test_prev":
            with open("./data/personal_prev/"+str(user_to_personalize)+ '/'+ str(personalize_test_phase) +'.pickle',"rb") as f:
                pickle_data = pickle.load(f)
                for (u_data, u_label) in pickle_data:
                    self.data.append(u_data)
                    self.label.append(int(u_label))
        
        else:
            print("error")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
