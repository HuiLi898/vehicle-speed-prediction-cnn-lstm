import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = "extracted_data/narvi/labels.csv"
OUTPUT_DIR = "data_split"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# Split by time order without shuffling to avoid time-series data leakage
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("Total:", n)
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))
