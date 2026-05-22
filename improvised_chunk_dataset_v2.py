import pandas as pd
import numpy as np
import torch
import torch_directml
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader
)
from sklearn.model_selection import train_test_split


# ============================================================
# LOAD FULL DATASET
# ============================================================

df = pd.read_csv(
    "tess_window_dataset.csv"
)

print("\n===================================")
print("FULL DATASET")
print("===================================")

print(
    df["label"].value_counts()
)


# ============================================================
# SPLIT POSITIVE / NEGATIVE
# ============================================================

positive_df = df[
    df["label"] == 1
]

negative_df = df[
    df["label"] == 0
]

num_positive = len(
    positive_df
)

print("\nPositive samples:")
print(num_positive)


# ============================================================
# RATIOS
# ============================================================

ratios = [10, 5, 3]


# ============================================================
# DEVICE
# ============================================================

device = torch_directml.device()

print("\n===================================")
print("DEVICE")
print("===================================")

print(device)


# ============================================================
# FLUX COLUMNS
# ============================================================

flux_cols = [
    col
    for col in df.columns
    if col.startswith("flux_")
]


# ============================================================
# DATASET CLASS
# ============================================================

class TESSDataset(Dataset):
    def __init__(self, dataframe):
        self.X = dataframe[
            flux_cols
        ].values.astype(
            np.float32
        )
        self.y = dataframe[
            "label"
        ].values.astype(
            np.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        flux = torch.tensor(
            self.X[idx]
        ).unsqueeze(0)
        label = torch.tensor(
            self.y[idx]
        )
        return flux, label


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


# ============================================================
# MODEL
# ============================================================

class PlanetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.global_max = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        avg_pool = self.global_avg(x)
        max_pool = self.global_max(x)
        avg_pool = avg_pool.squeeze(-1)
        max_pool = max_pool.squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.classifier(x)
        return x.squeeze(1)


# ============================================================
# CREATE AND PROCESS EACH RATIO DATASET
# ============================================================

for ratio in ratios:
    print("\n===================================")
    print(f"CREATING 1:{ratio}")
    print("===================================")

    # Sample negatives
    target_negatives = num_positive * ratio
    sampled_negative_df = negative_df.sample(
        n=target_negatives,
        random_state=42
    )

    # Combine and shuffle
    combined_df = pd.concat([positive_df, sampled_negative_df])
    combined_df = combined_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Save dataset
    output_name = f"tess_dataset_ratio_1_to_{ratio}.csv"
    combined_df.to_csv(output_name, index=False)

    print("\nSaved:", output_name)
    print("\nLabel distribution:")
    print(combined_df["label"].value_counts())
    print("\nDataset shape:", combined_df.shape)

    # ============================================================
    # TRAIN/VAL/TEST SPLIT (80/10/10)
    # ============================================================
    train_df, temp_df = train_test_split(
        combined_df,
        test_size=0.2,
        random_state=42,
        stratify=combined_df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"]
    )

    print(f"\nSplit sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ============================================================
    # DATASETS
    # ============================================================
    train_dataset = TESSDataset(train_df)
    val_dataset = TESSDataset(val_df)
    test_dataset = TESSDataset(test_df)

    # ============================================================
    # DATALOADERS
    # ============================================================
    BATCH_SIZE = 256
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ============================================================
    # MODEL INIT
    # ============================================================
    model = PlanetCNN().to(device)

    print("\n===================================")
    print("MODEL")
    print("===================================")
    print(model)

    # ============================================================
    # LOSS (with pos_weight for this ratio)
    # ============================================================
    pos_weight_value = len(train_df[train_df["label"] == 0]) / len(train_df[train_df["label"] == 1])
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ============================================================
    # OPTIMIZER
    # ============================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # ============================================================
    # TEST BATCH
    # ============================================================
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        logits = model(x_batch)

    print("\n===================================")
    print("TEST FORWARD PASS")
    print("===================================")
    print("Input shape:", x_batch.shape)
    print("Output shape:", logits.shape)
    
    print("\n" + "="*60)
    print(f"COMPLETED PROCESSING FOR RATIO 1:{ratio}")
    print("="*60)