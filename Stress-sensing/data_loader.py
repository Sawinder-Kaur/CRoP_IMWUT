import os
import numpy as np
from torch.utils.data import Dataset
import pickle 
class EEG_Dataset(Dataset):
    def __init__(self, mode,user_to_personalize, context_hand = 'left', context_move = 'still'):
        ### Modes: global_train, global_validate, personalized_train, personalized_test
        self.data = []
        self.label = []
        if mode == "global_train":
            with open("./data/change_train.pickle","rb") as f:
                pickle_data = pickle.load(f)
                #print(pickle_data)
                data_keys = pickle_data.keys()
                #print(data_keys)
                #print(user_to_personalize)
                global_users = data_keys - user_to_personalize
                #print(global_users)
                labels = {'calm':0, 'stress':1}
                for user in global_users:
                    for l in labels.keys():
                        for user_data in pickle_data[user][l]:
                            self.data.append(user_data)
                            self.label.append(int(labels[l]))
                """
                for (u_data, u_label) in pickle_data:
                    self.data.append(u_data)
                    self.label.append(int(u_label))
                """
                    
        elif mode == "global_validate":
            with open("./data/change_train.pickle","rb") as f:
                pickle_data = pickle.load(f)
                #print(pickle_data)
                data_keys = pickle_data.keys()
                #print(data_keys)
                #print(user_to_personalize)
                user_ids = user_to_personalize
                #print(global_users)
                labels = {'calm':0, 'stress':1}
                for user in user_ids:
                    for l in labels.keys():
                        for user_data in pickle_data[user][l]:
                            self.data.append(user_data)
                            self.label.append(int(labels[l]))
                """
                for (u_data, u_label) in pickle_data:
                    self.data.append(u_data)
                    self.label.append(int(u_label))
                """
        
        elif mode == "global_test":
            with open("./data/change_train.pickle","rb") as f:
                pickle_data = pickle.load(f)
                #print(pickle_data)
                data_keys = pickle_data.keys()
                #print(data_keys)
                #print(user_to_personalize)
                user_ids = user_to_personalize
                #print(global_users)
                labels = {'calm':0, 'stress':1}
                for user in user_ids:
                    for l in labels.keys():
                        for user_data in pickle_data[user][l]:
                            self.data.append(user_data)
                            self.label.append(int(labels[l]))
                    
        elif mode == "personalize_train":
            with open("./data/change_val.pickle","rb") as f:
                pickle_data = pickle.load(f)
                personal_data = pickle_data[user_to_personalize]
                #print(personal_data.keys())
                hand_personal_data = personal_data[context_hand]
                #print("{} --- {}".format(context_hand, context_move))
                labels = {'calm':0, 'stress':1}
                for l in labels.keys():
                    for user_data in hand_personal_data[l][context_move]:
                        self.data.append(user_data)
                        self.label.append(int(labels[l]))
                        
        elif mode == "personalize_test_switch1":
            with open("./data/change_val.pickle","rb") as f:
                pickle_data = pickle.load(f)
                personal_data = pickle_data[user_to_personalize]
                #if context_move == 'still': context_move = 'move'
                #else: context_move = 'still'
                if context_hand == 'left': context_hand = 'right'
                else: context_hand = 'left'
                #print(personal_data.keys())
                hand_personal_data = personal_data[context_hand]
                #print("{} --- {}".format(context_hand, context_move))
                #print(hand_personal_data)
                labels = {'calm':0, 'stress':1}
                for l in labels.keys():
                    for user_data in hand_personal_data[l][context_move]:
                        self.data.append(user_data)
                        self.label.append(int(labels[l]))
        
        elif mode == "personalize_test_switch2":
            with open("./data/change_val.pickle","rb") as f:
                pickle_data = pickle.load(f)
                personal_data = pickle_data[user_to_personalize]
                if context_move == 'still': context_move = 'move'
                else: context_move = 'still'
                if context_hand == 'left': context_hand = 'right'
                else: context_hand = 'left'
                #print(personal_data.keys())
                hand_personal_data = personal_data[context_hand]
                #print("{} --- {}".format(context_hand, context_move))
                #print(hand_personal_data)
                labels = {'calm':0, 'stress':1}
                for l in labels.keys():
                    for user_data in hand_personal_data[l][context_move]:
                        self.data.append(user_data)
                        self.label.append(int(labels[l]))
        
        else:
            print("error")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


"""    
#val_lst = ['0014', 'coung', '0002', '0007', '0016', '0011']
#test_lst = ['0004', '0009', '0013', '0015', 'brian', 'jingyu', 'joshua', 'lorn', 'ruipeng', 'susan8', 'yuxuan']
global_val_list = ['0014', 'coung', '0002',]
global_test_lst = ['0007', '0008', '0009']

personalize_list = ['0001', '0002', '0003']
#test_lst = ['0001', '0002', '0003', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016']
global_data_exludes = global_val_list + global_test_lst + personalize_list
global_train_data = EEG_Dataset("global_train", global_data_exludes)
global_val_data = EEG_Dataset("global_validate", global_val_list)
global_test_data = EEG_Dataset("global_test", global_test_lst)
#print(global_data)

#personalize_train_data = EEG_Dataset("personalize_train", '0001', "left")
#personalize_test_data_s1 = EEG_Dataset("personalize_test_switch1", '0001', "left")
#personalize_test_data_s2 = EEG_Dataset("personalize_test_switch2", '0001', "left")


print(len(global_train_data))
print(len(global_val_data))
print(len(global_test_data))

for p_user in personalize_list:
    personalize_train_data = EEG_Dataset("personalize_train", p_user, "left", 'still')
    personalize_test_data_s1 = EEG_Dataset("personalize_test_switch1", p_user, "left", 'still')
    personalize_test_data_s2 = EEG_Dataset("personalize_test_switch2", p_user, "left", 'still')
    print(len(personalize_train_data))
    print(len(personalize_test_data_s1))
    print(len(personalize_test_data_s2))
    print("--------------------------------")
    personalize_train_data = EEG_Dataset("personalize_train", p_user, "right", 'still')
    personalize_test_data_s1 = EEG_Dataset("personalize_test_switch1", p_user, "right", 'still')
    personalize_test_data_s2 = EEG_Dataset("personalize_test_switch2", p_user, "right", 'still')
    print(len(personalize_train_data))
    print(len(personalize_test_data_s1))
    print(len(personalize_test_data_s2))
    print("--------------------------------")
    
    
for p_user in personalize_list:
    personalize_train_data = EEG_Dataset("personalize_train", p_user, "left", 'move')
    personalize_test_data_s1 = EEG_Dataset("personalize_test_switch1", p_user, "left", 'move')
    personalize_test_data_s2 = EEG_Dataset("personalize_test_switch2", p_user, "left", 'move')
    print(len(personalize_train_data))
    print(len(personalize_test_data_s1))
    print(len(personalize_test_data_s2))
    print("--------------------------------")
    personalize_train_data = EEG_Dataset("personalize_train", p_user, "right", 'move')
    personalize_test_data_s1 = EEG_Dataset("personalize_test_switch1", p_user, "right", 'move')
    personalize_test_data_s2 = EEG_Dataset("personalize_test_switch2", p_user, "right", 'move')
    print(len(personalize_train_data))
    print(len(personalize_test_data_s1))
    print(len(personalize_test_data_s2))
    print("--------------------------------")
"""
"""
for sample, label in global_test_data:
    print(sample)
#"""