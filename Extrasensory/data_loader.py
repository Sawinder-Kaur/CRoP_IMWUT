import os
import numpy as np
from torch.utils.data import Dataset
import pickle 
class ExtraSensory_Dataset(Dataset):
    def __init__(self, mode,testing_contexts,user_to_personalize):
        ### Modes: global_train, global_validate, personalized_train, personalized_test
        self.data = []
        self.label = []
        contexts = ["label:PHONE_IN_POCKET","label:PHONE_IN_HAND","label:PHONE_IN_BAG"]
        label_encoder = {"label:FIX_walking":0,"label:SITTING":1}
        contexts.remove(testing_contexts)
        training_contexts = contexts
        if mode == "global_train":
            with open("context1_trainData.pickle","rb") as f:
                pickle_dict = pickle.load(f)
            labels_to_take = ["label:FIX_walking","label:SITTING"]
            for user_id in pickle_dict.keys():
                user_dict = pickle_dict[user_id]
                for l in labels_to_take:
                    for datum in user_dict[l]:
                        self.data.append(datum)
                        self.label.append(label_encoder[l])

        elif mode == "global_validate":
            with open("context1_valData.pickle","rb") as f:
                pickle_dict = pickle.load(f)
            labels_to_take = ["label:FIX_walking","label:SITTING"]
            for user_id in pickle_dict.keys():
                user_dict = pickle_dict[user_id]
                for l in labels_to_take:
                    for datum in user_dict[l]:
                        self.data.append(datum)
                        self.label.append(label_encoder[l])
        elif mode == "personalized_train":
            with open("context1_testData_new.pickle","rb") as f:
                pickle_dict = pickle.load(f)
            labels_to_take = ["label:FIX_walking","label:SITTING"]
            for user_id in pickle_dict.keys():
                if user_id != user_to_personalize:
                    continue
                user_dict = pickle_dict[user_id]
                for l in labels_to_take:
                    context_dict = user_dict[l]
                    for tc in training_contexts:
                        data_to_take = context_dict[tc]
                        for datum in data_to_take:
                            self.data.append(datum)
                            self.label.append(label_encoder[l])
        elif mode == "personalized_test":
            with open("context1_testData_new.pickle","rb") as f:
                pickle_dict = pickle.load(f)
            labels_to_take = ["label:FIX_walking","label:SITTING"]
            for user_id in pickle_dict.keys():
                if user_id != user_to_personalize:
                    continue
                user_dict = pickle_dict[user_id]
                for l in labels_to_take:
                    context_dict = user_dict[l]
                    data_to_take = context_dict[testing_contexts]
                    for datum in data_to_take:
                        self.data.append(datum)
                        self.label.append(label_encoder[l])
        else:
            print("error")

        del pickle_dict
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
