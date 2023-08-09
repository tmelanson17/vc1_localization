import os
import pandas as pd
import numpy as np
import re
import copy
from PIL import Image

from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
import torch


# Remove "#" and bracketed names
def preprocess_column_names(names):
    processed_names = [""]*len(names)
    for i in range(len(names)):
        processed_names[i] = re.sub(
                "\[.*\]", "", 
                names[i].strip().lstrip("#")
            ).strip()
    return processed_names

TRAIN_FRAC=0.8


# from https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.minimum(t2, 1.0)
        t2 = np.maximum(t2, -1.0)
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians


'''
 Pering GT dataset format:
    poses.gt
    |
    |-- timestamp
    |
    |-- position
    | |
    | |-- x
    | |
    | |-- y
    | |
    | |-- z
    |
    |
    |-- quaternion
    | |
    | |-- w
    | |
    | |-- x
    | |
    | |-- y
    | |
    | |-- z
'''
class PeringDataset(Dataset):
    def __init__(self, pering_main_directory, robot_type, transform=None, target_transform=None):
        self.gt_data = pd.read_csv(os.path.join(pering_main_directory, robot_type, "poses.gt"))
        self.gt_data.columns = preprocess_column_names(self.gt_data.columns)
        self.img_dir = os.path.join(pering_main_directory, robot_type, "cam0", "data")
        self.gt_data["filenames"] = self.gt_data["timestamp"].apply(
                lambda timestamp: (
                    os.path.join(self.img_dir, str(timestamp).zfill(19)) + ".png"
                )
        )
        self.transform = transform
        self.target_transform = target_transform
        self.TRAIN_TEST_SPLIT = (len(self.gt_data)*4)//5


    def __len__(self):
        return len(self.gt_data)

    def file(self, idx):
        return self.gt_data["filenames"].iloc[idx]

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        cam0_path = self.gt_data["filenames"].iloc[idx]
        image = torch.stack(
                cam0_path.map(lambda fname : read_image(fname, ImageReadMode.RGB)).tolist()
            )
        angle = euler_from_quaternion(
                self.gt_data["quaternion.x"].iloc[idx],
                self.gt_data["quaternion.y"].iloc[idx],
                self.gt_data["quaternion.z"].iloc[idx],
                self.gt_data["quaternion.w"].iloc[idx]
            )
        pose = (
                self.gt_data["position.x"].iloc[idx],
                self.gt_data["position.y"].iloc[idx],
                angle[2]
            )
        
        if self.transform:
            image = self.transform(image)
        labels = torch.zeros([len(cam0_path), 3])
        for i in range(3):
            labels[:, i] = torch.tensor(pose[i].tolist())
        # TODO
        if self.target_transform:
            labels = self.target_transform(labels)
        return image, labels


if __name__ == "__main__":
    names = [" ##Comment [34n/s]     ",]
    names = preprocess_column_names(names)
    print(f"{names[0]}")
    assert(names[0] == "Comment")
    sample_dataset = PeringDataset("/home/noisebridge/development/data/datasets/pering", "deer_robot")
    img, label = sample_dataset[1]
    print(len(sample_dataset))
    angles = []
    n_images = len(sample_dataset)
    offset = 2
    import time
    t1 = time.time()
    for i in range(n_images-offset):
        img, label = sample_dataset[i:i+offset]
        angles.append(label[1, 2]) # yaw
    t2 = time.time()
    print(f"Time taken per image: {(t2-t1)/(n_images*offset)}")
