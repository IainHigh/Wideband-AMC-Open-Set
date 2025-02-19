import torch
import torch.nn as nn
import torch.nn.functional as F

# This is the first CNN model used for testing the dataset. The model consists of 3 convolutional layers and 2 fully connected layers.


class ModulationClassifier(nn.Module):
    def __init__(self, num_classes, input_len=None):
        super(ModulationClassifier, self).__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        # Global pooling to reduce the length dimension to 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)  # Now input is 64 (one per channel)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)  # x shape becomes (batch_size, 64, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
