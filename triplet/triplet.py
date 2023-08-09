import numpy as np

from torch.utils.data import Dataset
import torch

class PeringTriplet(Dataset):
    def __init__(self, pering_dataset, k=5, random_seed=None):
        self._pering = pering_dataset
        self._generator = np.random.default_rng(seed=random_seed)
        self._k = k
        self._halfN = len(self._pering)//2
        self._N = len(self._pering)

    # TODO : create more than 1-1 pairs for triplets
    def __len__(self):
        return len(self._pering)

    def __getitem__(self, idx):
        kp = self._generator.integers(low=-self._k, high=self._k)
        kn = self._generator.integers(low=self._halfN-self._k, high=self._halfN+self._k)
        fileb = self._pering.file(idx)
        filep = self._pering.file(np.mod(idx+kp, self._N))
        filen = self._pering.file(np.mod(idx+kn, self._N))
        return (fileb, filep, filen)

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from pering_dataset.pering_dataset import PeringDataset
    dataset = PeringDataset(
            "/home/noisebridge/development/data/datasets/pering",
            "deer_robot"
    )
    trip = PeringTriplet(dataset)
    print(trip[5])
    dataloader = torch.utils.data.DataLoader(trip, batch_size=16, shuffle=True)
    i=0
    for fi, fj, fk in dataloader:
        i=i+1
        if i > 10:
            break
        print(f"Base: {fi}\n Positive: {fj}\n Negative: {fk}") 
