import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch_directml

from torch.utils.data import (
    Dataset,
    DataLoader
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
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
    # FLUX COLUMNS
    # ============================================================
    flux_cols = [
        col
        for col in train_df.columns
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
    # DATASETS
    # ============================================================
    train_dataset = TESSDataset(train_df)
    val_dataset = TESSDataset(val_df)
    test_dataset = TESSDataset(test_df)

    # ============================================================
    # DATALOADERS
    # ============================================================
    BATCH_SIZE = 256

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

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
            self.dropout = nn.Dropout(0.2)
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
            out = self.dropout(out)
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
                nn.MaxPool1d(2),
                nn.Dropout(0.2)
            )
            self.layer1 = ResidualBlock(32, 64, stride=2)
            self.layer2 = ResidualBlock(64, 128, stride=2)
            self.layer3 = ResidualBlock(128, 256, stride=2)
            self.global_avg = nn.AdaptiveAvgPool1d(1)
            self.global_max = nn.AdaptiveMaxPool1d(1)
            self.classifier = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
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
    # MODEL INIT
    # ============================================================
    model = PlanetCNN().to(device)

    # ============================================================
    # CLASS WEIGHTING
    # ============================================================
    num_negative = len(train_df[train_df["label"] == 0])
    num_positive = len(train_df[train_df["label"] == 1])
    pos_weight = torch.tensor([num_negative / num_positive]).to(device)

    print("\n===================================")
    print("POSITIVE CLASS WEIGHT")
    print("===================================")
    print(pos_weight)

    # ============================================================
    # LOSS
    # ============================================================
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ============================================================
    # OPTIMIZER
    # ============================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-3
    )

    # ============================================================
    # TRAIN FUNCTION
    # ============================================================
    def train_epoch():
        model.train()
        running_loss = 0.0
        for flux, labels in train_loader:
            flux = flux.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(flux)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * flux.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    # ============================================================
    # EVALUATION
    # ============================================================
    def evaluate(loader):
        model.eval()
        running_loss = 0.0
        all_probs = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for flux, labels in loader:
                flux = flux.to(device)
                labels = labels.to(device)
                logits = model(flux)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                running_loss += loss.item() * flux.size(0)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        epoch_loss = running_loss / len(loader.dataset)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        pr_auc = average_precision_score(all_labels, all_probs)
        return {
            "loss": epoch_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc
        }

    # ============================================================
    # EARLY STOPPING
    # ============================================================
    best_val_pr_auc = 0.0
    patience = 5
    patience_counter = 0

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    EPOCHS = 50

    for epoch in range(EPOCHS):
        print("\n===================================")
        print(f"EPOCH {epoch+1}/{EPOCHS}")
        print("===================================")

        train_loss = train_epoch()
        val_metrics = evaluate(val_loader)

        print("\nTRAIN LOSS:")
        print(f"{train_loss:.6f}")

        print("\nVALIDATION:")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.6f}")

        # Early stopping
        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["pr_auc"]
            patience_counter = 0
            torch.save(model.state_dict(), f"best_planet_cnn_ratio_{ratio}.pt")
            print("\nBEST MODEL SAVED")
        else:
            patience_counter += 1
            print(f"\nNo improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("\nEARLY STOPPING")
            break

    # ============================================================
    # FINAL TEST
    # ============================================================
    print("\n===================================")
    print("FINAL TEST")
    print("===================================")

    model.load_state_dict(torch.load(f"best_planet_cnn_ratio_{ratio}.pt"))
    test_metrics = evaluate(test_loader)

    for k, v in test_metrics.items():
        print(f"{k}: {v:.6f}")

    print("\n" + "="*60)
    print(f"COMPLETED PROCESSING FOR RATIO 1:{ratio}")
    print("="*60)