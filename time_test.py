import pandas as pd
import time
from PIL import Image
from pering_dataset.pering_dataset import PeringDataset
import torch
import numpy as np

def timeit(func):
    t1 = time.time()
    func()
    t2 = time.time()
    print(f"Time taken: {t2-t1}")

if __name__ == "__main__":
    sample_dataset = PeringDataset("/home/noisebridge/development/data/datasets/pering", "deer_robot")
    n_images=64
    t1 = time.time()
    images = [Image.open(filename) for filename in sample_dataset.gt_data["filenames"].iloc[:n_images]]
    t2 = time.time()
    print(f"Non vectorized time: {t2-t1}")

    t1 = time.time()
    images = sample_dataset.gt_data["filenames"].iloc[:n_images].map(Image.open)
    t2 = time.time()
    print(f"Vectorized time: {t2-t1}")

    n_labels=128
    data = np.random.random([n_labels, 3])
    t1 = time.time()
    torch.stack([torch.tensor(data[:, i]) for i in range(3)], dim=1)
    t2 = time.time()
    print(f"tensor + stack: {t2-t1}")

    def fill_tensor():
        t = torch.zeros([n_labels, 3])
        t[:, 0] = torch.tensor(data[:, 0])
        t[:, 1] = torch.tensor(data[:, 1])
        t[:, 2] = torch.tensor(data[:, 2])
    print("Filling tensor")
    timeit(fill_tensor)

    def fill_for_loop():
        t = torch.zeros([n_labels, 3])
        for i in range(3):
            t[:, i] = torch.tensor(data[:, i])
    timeit(fill_for_loop)
