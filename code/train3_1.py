import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset3_1 import SpeedDataset
from model3_1 import CNN_LSTM


# =====================
# Config
# =====================

IMAGE_ROOT = "extracted_data/narvi"
TRAIN_CSV = "data_split/train.csv"
VAL_CSV = "data_split/val.csv"

SEQ_LEN = 3
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.00001
PATIENCE = 10
SPEED_MAX = 3.0

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# run1_final.pth / run2_final.pth / run3_final.pth 
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "run3_final.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =====================
# Helpers
# =====================

def fix_speed_shape(speeds):
    speeds = speeds.float() / SPEED_MAX
    if speeds.dim() == 1:
        speeds = speeds.unsqueeze(1)
    return speeds


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for images, speeds in loader:
            images = images.to(device)
            targets = fix_speed_shape(speeds).to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)

            preds = outputs.cpu().numpy().flatten() * SPEED_MAX
            acts = speeds.cpu().numpy().flatten()

            predictions.extend(preds)
            actuals.extend(acts)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    val_loss = total_loss / len(loader.dataset)
    mae = np.mean(np.abs(predictions - actuals))

    return val_loss, mae, predictions, actuals


def save_plots(train_losses, val_losses, val_maes):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train SmoothL1 Loss")
    plt.plot(val_losses, label="Validation SmoothL1 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.savefig("train_val_loss.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(val_maes, label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.title("Validation MAE")
    plt.grid(True)
    plt.savefig("val_mae.png")
    plt.close()

    print("Saved figures:")
    print("train_val_loss.png")
    print("val_mae.png")


# =====================
# Dataset
# =====================

train_dataset = SpeedDataset(
    csv_file=TRAIN_CSV,
    image_root=IMAGE_ROOT,
    seq_len=SEQ_LEN,
    is_training=True
)

val_dataset = SpeedDataset(
    csv_file=VAL_CSV,
    image_root=IMAGE_ROOT,
    seq_len=SEQ_LEN,
    is_training=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

test_images, test_targets = next(iter(train_loader))
print("Check images shape:", test_images.shape)
print("Check targets shape:", test_targets.shape)


# =====================
# Model
# =====================

model = CNN_LSTM(hidden_size=128).to(device)
print(model)

# Freeze CNN, only train LSTM + FC
for param in model.cnn.parameters():
    param.requires_grad = False

criterion = nn.SmoothL1Loss(beta=0.05)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


# =====================
# Training
# =====================

train_losses = []
val_losses = []
val_maes = []

best_val_mae = float("inf")
patience_counter = 0

print(f"Starting training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0

    for images, speeds in train_loader:
        images = images.to(device)
        targets = fix_speed_shape(speeds).to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)

    val_loss, val_mae, preds, acts = evaluate(
        model,
        val_loader,
        criterion
    )

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_maes.append(val_mae)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f} "
        f"Val MAE: {val_mae:.6f} "
        f"LR: {current_lr:.8f}"
    )

    print("Pred min/max:", np.min(preds), np.max(preds))
    print("Actual min/max:", np.min(acts), np.max(acts))

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mae": best_val_mae,
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "speed_max": SPEED_MAX,
            "loss": "SmoothL1Loss_beta_0.05",
            "cnn_frozen": True,
        }, BEST_MODEL_PATH)

        print("Saved best model by MAE.")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break


print("Training finished.")
print("Best validation MAE:", best_val_mae)
print("Best model:", BEST_MODEL_PATH)


# =====================
# Save Figures
# =====================

save_plots(train_losses, val_losses, val_maes)