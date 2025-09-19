# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NumericBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ImageBranch(nn.Module):
    def __init__(self, input_channels=8):  # RGB+Depth from 2 cams = 6 channels
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 16 * 16, 128)  # after conv downsamples to ~16x16

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class FusionNet(nn.Module):
    def __init__(self, numeric_dim, output_dim=2):
        super().__init__()
        self.num_branch = NumericBranch(numeric_dim)
        self.img_branch = ImageBranch()
        self.fc1 = nn.Linear(64 + 128, 128)  # combine both branches
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, numeric, images):
        n = self.num_branch(numeric)
        i = self.img_branch(images)
        x = torch.cat([n, i], dim=1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

