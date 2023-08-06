import torch
from vc_models.models.vit import model_utils

class LocalizationModel(torch.nn.Module):
    def __init__(self):
        super(LocalizationModel, self).__init__()
        self.vc,self.embd_size,self.model_transforms,self.model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
        print(self.embd_size)
        for param in self.vc.parameters():
            param.requires_grad = False
        l1 = torch.nn.Linear(self.embd_size, 256)
        l2 = torch.nn.Linear(256, 64)
        l3 = torch.nn.Linear(64, 3)
        self.fc = torch.nn.Sequential(
            l1, l2, l3
        )

    def forward(self, img):
        img_transform = self.model_transforms(img)
        embed = self.vc(img_transform)
        return self.fc(embed)


def localization_loss(outputs, labels):
    # L2 between outputs and labels
    # (except theta will be cosine loss)
    pass


if __name__ == "__main__":
    model = LocalizationModel()
    import numpy as np
    model(torch.tensor(np.random.random([1, 3, 640, 480]).astype(np.float32)))

