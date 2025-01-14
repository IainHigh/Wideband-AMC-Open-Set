import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulationClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ModulationClassifier, self).__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 256, 128)  # Adjust 256 based on IQ input length
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
