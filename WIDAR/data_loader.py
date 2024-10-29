import os
import numpy as np
from torch.utils.data import Dataset

class WIDAR_Dataset(Dataset):
    def __init__(self, base_path, extracted_path, user_id, split_type="personal", split_id=None, mode="train"):
        """
        :param base_path: Root directory of the data.
        :param extracted_path: Directory containing yi's extracted data.
        :param user_id: ID of the user for which data should be loaded.
        :param split_type: Either "personal" or "global" specifying which data subset to load.
        :param split_id: ID for the global dataset split (can be 0, 1, or 2). Not needed for personal splits.
        :param mode: If "training", data will be normalized. Otherwise, it remains unnormalized.
        """
        
        # Define the path based on base_path, user_id, and split_type
        params_path = os.path.join(base_path, f"user{user_id}", "global", f"split_{split_id}")
        if split_type == "personal":
            glob_path = os.path.join(base_path, f"user{user_id}", split_type)
            data_path = os.path.join(glob_path, mode)
        elif split_type == "global":
            glob_path = os.path.join(base_path, f"user{user_id}", split_type, f"split_{split_id}")
            data_path = os.path.join(glob_path, mode)
        else:
            raise ValueError("Invalid split_type. Choose either 'personal' or 'global'.")
        
        self.data,self.gesture,self.user,self.torso,self.face,self.room = self.load_data(data_path,extracted_path)
        self.gesture[self.gesture == 8] = 4
        # Normalize data if mode is "training" # change this to include test for training ctesting zero shot non magnitude. HARPS PARMETER? LAYERWISE PRUNING
        self.data=np.moveaxis(self.data, -1,1)
        self.mean,self.std = np.load(os.path.join(params_path,"mean_std.npy"))
        self.data = (self.data - self.mean) / self.std
        # Else: the data remains unnormalized

    def load_data(self,path_to_load,extracted_path):
        with open(os.path.join(path_to_load,"data.npy"), 'rb') as f:
            mask = np.load(f,)

        with open(os.path.join(extracted_path,fr"data.npy"), 'rb') as f:
            data_full=np.load(f,)
        data_full_mask = data_full[mask]
        del data_full

        with open(os.path.join(extracted_path,fr"gesture.npy"), 'rb') as f:
            gesture_full=np.load(f,)
        gesture_full_mask = gesture_full[mask]
        del gesture_full

        with open(os.path.join(extracted_path,fr"user.npy"), 'rb') as f:
            user_full=np.load(f,)
        user_full_mask = user_full[mask]
        del user_full

        with open(os.path.join(extracted_path,fr"torso.npy"), 'rb') as f:
            torso_full=np.load(f,)
        torso_full_mask = torso_full[mask]
        del torso_full

        with open(os.path.join(extracted_path,fr"face.npy"), 'rb') as f:
            face_full=np.load(f,)
        face_full_mask = face_full[mask]
        del face_full
        with open(os.path.join(extracted_path,fr"room.npy"), 'rb') as f:
            room_full=np.load(f,)
        room_full_mask = room_full[mask]
        del room_full

        return data_full_mask,gesture_full_mask,user_full_mask,torso_full_mask,face_full_mask,room_full_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = {
        #     "data": self.data[idx],
        #     "gesture": self.gesture[idx],
        #     "user": self.user[idx],
        #     "torso": self.torso[idx],
        #     "face": self.face[idx],
        #     "room": self.room[idx]
        # }
        return self.data[idx].astype(np.float32), self.gesture[idx].astype(np.float32)
