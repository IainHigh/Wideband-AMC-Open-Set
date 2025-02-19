import torch
import torch.nn as nn
import torch.nn.functional as F

# This CNN model is based on the paper "Deep Learning-Based Automatic Modulation Classification Using Robust CNN Architecture for Cognitive Radio Networks". We followed their architecture as specified in the paper.


class ModulationClassifier(nn.Module):
    def __init__(self, num_classes, input_len=None):
        super(ModulationClassifier, self).__init__()

        # Conv Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=8, stride=1),  # 32 filters, kernel size 1x8
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Max-pooling (1x2), stride (1,2)
        )

        # Block 2 (repeated 4 times with dynamic channels)
        self.block2_layers = nn.ModuleList(
            [
                self._create_block2(
                    input_channels=32 if i == 0 else 96, output_channels=96
                )
                for i in range(4)
            ]
        )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Pool to (1x1x96)

        # Fully Connected Layer
        self.fc = nn.Linear(96, num_classes)  # FC layer

    def _create_block2(self, input_channels, output_channels):
        """Creates a Block 2 structure with parallel convolutions and skip connection."""
        return nn.ModuleDict(
            {
                "branch1": nn.Sequential(
                    nn.Conv1d(input_channels, 32, kernel_size=1, stride=2),  # 32x(1x1)
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                ),
                "branch2": nn.Sequential(
                    nn.Conv1d(
                        input_channels, 32, kernel_size=3, stride=2, padding=1
                    ),  # 32x(1x3)
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                ),
                "branch3": nn.Sequential(
                    nn.Conv1d(input_channels, 32, kernel_size=1, stride=2),  # 32x(3x1)
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                ),
                "residual": nn.Sequential(
                    nn.Conv1d(
                        input_channels, output_channels, kernel_size=1, stride=2
                    ),  # Residual path
                    nn.BatchNorm1d(output_channels),
                ),
            }
        )

    def forward(self, x):
        # Conv Block 1
        x = self.conv_block1(x)

        # Block 2 with skip connections
        for block in self.block2_layers:
            residual = block["residual"](x)  # Residual path
            branch_outputs = [
                block["branch1"](x),
                block["branch2"](x),
                block["branch3"](x),
            ]
            concatenated = torch.cat(branch_outputs, dim=1)  # Concatenate features
            x = F.relu(concatenated + residual)  # Add residual and apply ReLU

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten for FC layer

        # Fully Connected Layer
        x = self.fc(x)
        return x
