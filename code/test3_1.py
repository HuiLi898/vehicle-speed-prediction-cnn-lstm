import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset3_1 import SpeedDataset
from model3_1 import CNN_LSTM


# =====================
# Config
# =====================

IMAGE_ROOT = "extracted_data/narvi"
TEST_CSV = "data_split/test.csv"

# run1_final.pth / run2_final.pth / run3_final.pth
MODEL_PATH = "models/run3_final.pth"

SEQ_LEN = 3
BATCH_SIZE = 16
SPEED_MAX = 3.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =====================
# Dataset
# =====================

test_dataset = SpeedDataset(
    csv_file=TEST_CSV,
    image_root=IMAGE_ROOT,
    seq_len=SEQ_LEN,
    is_training=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)


# =====================
# Model
# =====================

model = CNN_LSTM(hidden_size=128).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()


# =====================
# Helpers
# =====================

def fix_speed_shape(speeds):
    speeds = speeds.float() / SPEED_MAX
    if speeds.dim() == 1:
        speeds = speeds.unsqueeze(1)
    return speeds


# =====================
# Evaluation
# =====================

criterion = nn.SmoothL1Loss(beta=0.05)

all_preds = []
all_targets = []
test_loss = 0.0

with torch.no_grad():
    for images, speeds in test_loader:
        images = images.to(device)
        targets_norm = fix_speed_shape(speeds).to(device)

        outputs = model(images)
        loss = criterion(outputs, targets_norm)

        test_loss += loss.item() * images.size(0)

        preds_original = outputs.cpu().numpy().flatten() * SPEED_MAX
        targets_original = speeds.cpu().numpy().flatten()

        all_preds.extend(preds_original)
        all_targets.extend(targets_original)


all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

test_loss = test_loss / len(test_dataset)
test_mae = np.mean(np.abs(all_preds - all_targets))
test_mse = np.mean((all_preds - all_targets) ** 2)
test_rmse = np.sqrt(test_mse)


# =====================
# Save results
# =====================

df = pd.DataFrame({
    "target": all_targets,
    "prediction": all_preds,
    "absolute_error": np.abs(all_preds - all_targets)
})

df.to_csv("test_predictions.csv", index=False)


# =====================
# Plots
# =====================

plt.figure(figsize=(12, 5))
plt.plot(all_targets[:300], label="Actual Speed", alpha=0.7)
plt.plot(all_preds[:300], label="Predicted Speed", alpha=0.7, linestyle="--")
plt.xlabel("Sample Index")
plt.ylabel("Speed")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title(f"Test Prediction vs Actual (MAE: {test_mae:.4f})")
plt.savefig("prediction_vs_actual.png")
plt.close()


plt.figure(figsize=(6, 6))
plt.scatter(all_targets, all_preds, s=10, alpha=0.5)

lims = [
    min(all_targets.min(), all_preds.min()),
    max(all_targets.max(), all_preds.max())
]
plt.plot(lims, lims, "r-", alpha=0.75)

plt.xlabel("Actual Speed")
plt.ylabel("Predicted Speed")
plt.title("Prediction Scatter Plot")
plt.savefig("prediction_scatter.png")
plt.close()


# =====================
# Print results
# =====================

print("\nSaved files:")
print("test_predictions.csv")
print("prediction_vs_actual.png")
print("prediction_scatter.png")

print("\nTest results:")
print(f"SmoothL1 Loss: {test_loss:.6f}")
print(f"MSE:          {test_mse:.6f}")
print(f"RMSE:         {test_rmse:.6f}")
print(f"MAE:          {test_mae:.6f}")