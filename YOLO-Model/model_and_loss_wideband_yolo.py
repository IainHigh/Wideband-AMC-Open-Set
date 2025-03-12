#############################################
# model_and_loss_wideband_yolo.py
#############################################
import torch
import torch.nn as nn
import torch.nn.functional as F

from config_wideband_yolo import (
    S,
    B,
    NUM_CLASSES,
    LAMBDA_COORD,
    LAMBDA_NOOBJ,
    LAMBDA_CLASS,
)

###################################################
# Hyperparams (some can remain in config if desired)
###################################################
INIT_CHANNELS = 32  # for first conv
NUM_BLOCKS = 4      # how many repeated residual blocks
BLOCK_OUT_CH = 96   # output channels of each block
KERNEL_SIZE = 8
STRIDE = 1
POOL_STRIDE=2

class ResidualBlock(nn.Module):
    """
    Residual block with 3 branches, like your previous pipeline approach.
    branch1 => stride=2 with kernel_size=1
    branch2 => stride=2 with kernel_size=3
    branch3 => stride=2 with kernel_size=1
    Then we concat, plus a skip connection that also has stride=2
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # branch 1
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=1, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        # branch 2
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        # branch 3
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=1, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        # residual (skip)
        self.residual = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=2),
            nn.BatchNorm1d(out_ch),
        )
        self.out_ch = out_ch

    def forward(self, x):
        res = self.residual(x)  # shape => [batch, out_ch, length/2]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        concat = torch.cat([b1, b2, b3], dim=1)  # => [batch, 32+32+32=96, length/2]
        out = F.relu(concat + res)
        return out

class WidebandYoloModel(nn.Module):
    """
    Deeper YOLO-like model, using a residual-based pipeline approach
    reminiscent of the old "ModulationClassifier" design.
    1) A first conv block
    2) Several residual blocks
    3) A global pooling
    4) Final FC => [S, B*(1+1+NUM_CLASSES)]
    """
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

        # initial conv
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(2, INIT_CHANNELS, kernel_size=KERNEL_SIZE, stride=STRIDE),
            nn.BatchNorm1d(INIT_CHANNELS),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # repeated residual blocks
        blocks = []
        in_ch = INIT_CHANNELS
        for i in range(NUM_BLOCKS):
            # each block outputs 96
            blocks.append(ResidualBlock(in_ch, BLOCK_OUT_CH))
            in_ch = BLOCK_OUT_CH
        self.block2_layers = nn.ModuleList(blocks)

        # after the final block, we do a global average pool => we get shape [batch, out_ch, 1]
        # then a linear
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # final FC => S*(B*(1+1+NUM_CLASSES))
        self.output_dim = S * B * (1 + 1 + NUM_CLASSES)
        self.fc = nn.Linear(BLOCK_OUT_CH, self.output_dim)

    def forward(self, x):
        # x: [batch, 2, num_samples]
        x = self.conv_block1(x)   # => [batch, 32, length/2]

        for block in self.block2_layers:
            x = block(x)          # => eventually [batch, 96, length/(2^(NUM_BLOCKS+1))]

        x = self.global_avg_pool(x)  # => [batch, 96, 1]
        x = x.squeeze(-1)           # => [batch, 96]
        x = self.fc(x)              # => [batch, output_dim]
        # reshape => [batch, S, B*(1+1+NUM_CLASSES)]
        bsize = x.shape[0]
        x = x.view(bsize, S, -1)
        return x

class WidebandYoloLoss(nn.Module):
    """
    Same YOLO loss as before, but references the new dimensioning
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        pred: [batch, S, B*(1 + 1 + NUM_CLASSES)]
        target: [batch, S, B, (1 + 1 + NUM_CLASSES)]
        """
        batch_size = pred.shape[0]
        # reshape pred
        pred = pred.view(batch_size, pred.shape[1], B, (1 + 1 + NUM_CLASSES))

        x_pred     = pred[..., 0]
        conf_pred  = pred[..., 1]
        class_pred = pred[..., 2:]

        x_tgt      = target[..., 0]
        conf_tgt   = target[..., 1]
        class_tgt  = target[..., 2:]

        obj_mask   = (conf_tgt > 0).float()
        noobj_mask = 1.0 - obj_mask

        # coordinate MSE
        coord_loss = LAMBDA_COORD * torch.sum(obj_mask*(x_pred - x_tgt)**2)

        # iou in 1D
        iou_1d = 1.0 - torch.abs(x_pred - x_tgt)
        iou_1d = torch.clamp(iou_1d, min=0.0, max=1.0)

        # confidence
        conf_loss_obj = torch.sum(obj_mask*(conf_pred - iou_1d)**2)
        conf_loss_noobj = LAMBDA_NOOBJ * torch.sum(noobj_mask*(conf_pred**2))

        # class MSE
        class_diff = (class_pred - class_tgt)**2
        class_loss = LAMBDA_CLASS*torch.sum(obj_mask.unsqueeze(-1)*class_diff)

        total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
        return total_loss / batch_size
