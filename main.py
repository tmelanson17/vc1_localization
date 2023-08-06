from pering_dataset.pering_dataset import PeringDataset
import torch.utils.data

def main():
    TRAIN_PERCENTAGE=0.8
    dataset = PeringDataset(
            "/home/noisebridge/development/data/datasets/pering",
            "deer_robot"
    )
    train_test_split = torch.utils.data.random_split(
            dataset,
            [TRAIN_PERCENTAGE, 1-TRAIN_PERCENTAGE], #generator=torch.utils.Generator().manual_seed(42)
    )
    dataloader = torch.utils.data.DataLoader(train_test_split[0], batch_size=16, shuffle=True)
    


if __name__ == "__main__":
    main()
