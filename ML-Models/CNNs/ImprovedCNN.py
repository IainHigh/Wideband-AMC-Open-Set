import torch
import torch.nn as nn
import torch.nn.functional as F

# This model was implemented to be an improvement over the simple CNN model, it has more layers and uses residual connections.


class ModulationClassifier(nn.Module):
    def __init__(self, num_classes, input_len=1024):  # Default input length = 1024
        super(ModulationClassifier, self).__init__()

        # Initial convolutional block
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        # Residual block 1
        self.res1_conv1 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.res1_bn1 = nn.BatchNorm1d(256)
        self.res1_conv2 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.res1_bn2 = nn.BatchNorm1d(256)
        self.res1_downsample = nn.Conv1d(
            128, 256, kernel_size=1, stride=1
        )  # Projection layer for residual

        # Residual block 2
        self.res2_conv1 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.res2_bn1 = nn.BatchNorm1d(512)
        self.res2_conv2 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.res2_bn2 = nn.BatchNorm1d(512)
        self.res2_downsample = nn.Conv1d(
            256, 512, kernel_size=1, stride=1
        )  # Projection layer for residual

        # Convolutional block before fully connected layers
        self.conv3 = nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(1024)

        # Calculate flattened size dynamically
        self.flattened_size = input_len * 1024  # Adjust based on convolutional layers

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Residual block 1
        res1 = self.res1_downsample(x)  # Adjust dimensions using the projection layer
        x = F.relu(self.res1_bn1(self.res1_conv1(x)))
        x = self.res1_bn2(self.res1_conv2(x))
        x += res1  # Residual connection
        x = F.relu(x)

        # Residual block 2
        res2 = self.res2_downsample(x)  # Adjust dimensions using the projection layer
        x = F.relu(self.res2_bn1(self.res2_conv1(x)))
        x = self.res2_bn2(self.res2_conv2(x))
        x += res2  # Residual connection
        x = F.relu(x)

        # Convolutional block before fully connected layers
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        # Output layer
        x = self.fc4(x)
        return x
