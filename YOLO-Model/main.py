#############################################
# main.py
#############################################
import torch
import json
import time
import sys
import os
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset_wideband_yolo import WidebandYoloDataset
from model_and_loss_wideband_yolo import WidebandYoloModel, WidebandYoloLoss
from config_wideband_yolo import (
    NUM_CLASSES,
    CREATE_NEW_DATASET,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
)

from utils import create_dataset

with open("./configs/system_parameters.json") as f:
    system_parameters = json.load(f)

working_directory = system_parameters["Working_Directory"]
sys.path.append(working_directory)

rng_seed = system_parameters["Random_Seed"]

data_dir = system_parameters["Dataset_Directory"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(rng_seed)

if torch.cuda.is_available():
    print("\n\nCUDA is available.")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("\n\nCUDA is not available. Using CPU.")
print("\n")

def main():
    # 0) Optionally create the dataset (synthetic or otherwise)
    if CREATE_NEW_DATASET:
        create_dataset(data_dir, rng_seed)

    # 1) Build dataset
    train_dataset = WidebandYoloDataset(
        os.path.join(data_dir, "training"), transform=None
    )
    val_dataset = WidebandYoloDataset(
        os.path.join(data_dir, "validation"), transform=None
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2) Create model & loss
    num_samples = train_dataset.get_num_samples()
    model = WidebandYoloModel(num_samples).to(device)
    criterion = WidebandYoloLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3) Training loop
    for epoch in range(EPOCHS):
        ############################
        # Training
        ############################
        model.train()
        total_train_loss = 0.0

        # For metrics:
        train_obj_count = 0      # How many boxes actually had an object
        train_correct_cls = 0    # How many of those objects had correct class predicted
        train_sum_freq_err = 0.0 # Sum of freq errors for each object

        for iq_tensor, label_tensor in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{EPOCHS}"):
            iq_tensor = iq_tensor.to(device)           # shape [batch, 2, N]
            label_tensor = label_tensor.to(device)     # shape [batch, S, B, (1+1+NUM_CLASSES)]

            optimizer.zero_grad()
            pred = model(iq_tensor)
            loss = criterion(pred, label_tensor)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # ====== Compute additional training metrics ======
            # 1) reshape pred => [batch, S, B, (1+1+NUM_CLASSES)]
            bsize = pred.shape[0]
            pred_reshape = pred.view(bsize, pred.shape[1], -1, (1+1+NUM_CLASSES))
            # parse
            x_pred     = pred_reshape[..., 0]               # [bsize, S, B]
            conf_pred  = pred_reshape[..., 1]               # [bsize, S, B]
            class_pred = pred_reshape[..., 2:]              # [bsize, S, B, num_classes]

            # from label
            x_tgt    = label_tensor[..., 0]
            conf_tgt = label_tensor[..., 1]
            class_tgt= label_tensor[..., 2:]

            # object mask
            obj_mask = (conf_tgt > 0)
            # collect predicted and true data for just those objects
            # frequency error
            freq_err = torch.abs(x_pred - x_tgt)   # [bsize, S, B]

            # classification => argmax
            # predicted class (..., ) => indices
            pred_class_idx = torch.argmax(class_pred, dim=-1)  # [bsize, S, B]
            # true class => argmax of class_tgt
            true_class_idx = torch.argmax(class_tgt, dim=-1)   # [bsize, S, B]

            # sum up freq error, correct classification only where obj_mask=1
            batch_obj_count = obj_mask.sum().item()

            # gather freq error for those boxes
            # freq_err[obj_mask] => shape (#objects, )
            batch_sum_freq_err = freq_err[obj_mask].sum().item()

            # gather classification
            # compare pred_class_idx[obj_mask] with true_class_idx[obj_mask]
            # => boolean => sum
            correct_cls_mask = (pred_class_idx == true_class_idx)
            batch_correct_cls = correct_cls_mask[obj_mask].sum().item()

            train_obj_count     += batch_obj_count
            train_sum_freq_err  += batch_sum_freq_err
            train_correct_cls   += batch_correct_cls

        avg_train_loss = total_train_loss / len(train_loader)

        if train_obj_count > 0:
            train_mean_freq_err = train_sum_freq_err / train_obj_count
            train_cls_accuracy  = 100.0 * (train_correct_cls / train_obj_count)
        else:
            # no objects found => trivial metrics
            train_mean_freq_err = 0.0
            train_cls_accuracy  = 0.0

        ############################
        # Validation
        ############################
        model.eval()
        total_val_loss = 0.0

        val_obj_count = 0
        val_correct_cls = 0
        val_sum_freq_err = 0.0

        with torch.no_grad():
            for iq_tensor, label_tensor in tqdm(val_loader, desc=f"Validation epoch {epoch+1}/{EPOCHS}"):
                iq_tensor = iq_tensor.to(device)
                label_tensor = label_tensor.to(device)
                pred = model(iq_tensor)
                loss = criterion(pred, label_tensor)
                total_val_loss += loss.item()

                # metrics
                bsize = pred.shape[0]
                pred_reshape = pred.view(bsize, pred.shape[1], -1, (1+1+NUM_CLASSES))

                x_pred     = pred_reshape[..., 0]
                conf_pred  = pred_reshape[..., 1]
                class_pred = pred_reshape[..., 2:]

                x_tgt    = label_tensor[..., 0]
                conf_tgt = label_tensor[..., 1]
                class_tgt= label_tensor[..., 2:]

                obj_mask = (conf_tgt > 0)
                freq_err = torch.abs(x_pred - x_tgt)

                pred_class_idx = torch.argmax(class_pred, dim=-1)
                true_class_idx = torch.argmax(class_tgt, dim=-1)

                batch_obj_count = obj_mask.sum().item()
                batch_sum_freq_err = freq_err[obj_mask].sum().item()
                correct_cls_mask = (pred_class_idx == true_class_idx)
                batch_correct_cls = correct_cls_mask[obj_mask].sum().item()

                val_obj_count += batch_obj_count
                val_sum_freq_err += batch_sum_freq_err
                val_correct_cls += batch_correct_cls

        avg_val_loss = total_val_loss / len(val_loader)

        if val_obj_count > 0:
            val_mean_freq_err = val_sum_freq_err / val_obj_count
            val_cls_accuracy  = 100.0 * (val_correct_cls / val_obj_count)
        else:
            val_mean_freq_err = 0.0
            val_cls_accuracy  = 0.0

        # Print metrics for this epoch
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"  Train: Loss={avg_train_loss:.4f},"
              f"  MeanFreqErr={train_mean_freq_err:.4f},"
              f"  ClsAcc={train_cls_accuracy:.2f}%")
        print(f"  Valid: Loss={avg_val_loss:.4f},"
              f"  MeanFreqErr={val_mean_freq_err:.4f},"
              f"  ClsAcc={val_cls_accuracy:.2f}%")

    print("Training complete.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    time_diff = time.time() - start_time
    print(
        f"\nCode Execution took {time_diff // 3600:.0f} hours, "
        f"{(time_diff % 3600) // 60:.0f} minutes, {time_diff % 60:.0f} seconds."
    )
