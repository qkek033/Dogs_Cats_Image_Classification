import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import timm


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class EfficientNetB7(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        hidden_dim = 512
        self.model._fc = nn.Sequential(
            nn.Linear(self.model._fc.in_features, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class ViTBasePatch32(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch32_224",
            pretrained=False,
            num_classes=1,
        )
        hidden_dim = 512
        self.model.head = nn.Sequential(
            nn.Linear(self.model.head.in_features, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
