# ============================================================
# LEAKAGE-FREE CHUNK SYSTEM TRAINER
# TIC-LEVEL SPLIT VERSION
# DIRECTML + WINDOWS
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

import torch
import torch.nn as nn
import torch_directml

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DATASET_CSV = (
    r"C:\Users\aasha\OneDrive\Desktop\correction\tess_chunk_system_dataset.csv"
)

BATCH_SIZE = 128

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 7

RANDOM_STATE = 42


# ============================================================
# DEVICE
# ============================================================

device = torch_directml.device()

print("\n===================================")
print("DEVICE")
print("===================================")
print(device)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n===================================")
print("LOADING DATASET")
print("===================================")

df = pd.read_csv(
    DATASET_CSV
)

print(df.shape)

print("\nLabel distribution:")
print(
    df["label"].value_counts()
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

flux_cols = [

    c for c in df.columns

    if c.startswith("flux_")
]


# ============================================================
# TIC-LEVEL SPLIT
# ============================================================

unique_tics = df[
    "tic_id"
].unique()

train_tics, temp_tics = train_test_split(

    unique_tics,

    test_size=0.25,

    random_state=RANDOM_STATE
)

val_tics, test_tics = train_test_split(

    temp_tics,

    test_size=0.5,

    random_state=RANDOM_STATE
)

train_df = df[
    df["tic_id"].isin(train_tics)
]

val_df = df[
    df["tic_id"].isin(val_tics)
]

test_df = df[
    df["tic_id"].isin(test_tics)
]


# ============================================================
# TRAIN
# ============================================================

X_train = train_df[
    flux_cols
].values.astype(np.float32)

y_train = train_df[
    "label"
].values.astype(np.float32)


# ============================================================
# VALIDATION
# ============================================================

X_val = val_df[
    flux_cols
].values.astype(np.float32)

y_val = val_df[
    "label"
].values.astype(np.float32)


# ============================================================
# TEST
# ============================================================

X_test = test_df[
    flux_cols
].values.astype(np.float32)

y_test = test_df[
    "label"
].values.astype(np.float32)


# ============================================================
# SPLIT INFO
# ============================================================

print("\n===================================")
print("TRAIN")
print("===================================")

print(X_train.shape)

print(
    pd.Series(y_train).value_counts()
)

print("\nUnique TICs:")
print(
    len(train_tics)
)


print("\n===================================")
print("VALIDATION")
print("===================================")

print(X_val.shape)

print(
    pd.Series(y_val).value_counts()
)

print("\nUnique TICs:")
print(
    len(val_tics)
)


print("\n===================================")
print("TEST")
print("===================================")

print(X_test.shape)

print(
    pd.Series(y_test).value_counts()
)

print("\nUnique TICs:")
print(
    len(test_tics)
)


# ============================================================
# DATASET
# ============================================================

class ChunkDataset(Dataset):

    def __init__(

        self,

        X,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(

        self,
        idx
    ):

        x = self.X[idx].unsqueeze(0)

        y = self.y[idx]

        return x, y


train_dataset = ChunkDataset(
    X_train,
    y_train
)

val_dataset = ChunkDataset(
    X_val,
    y_val
)

test_dataset = ChunkDataset(
    X_test,
    y_test
)


# ============================================================
# BALANCED SAMPLER
# ============================================================

class_counts = np.bincount(
    y_train.astype(int)
)

class_weights = (
    1.0 / class_counts
)

sample_weights = class_weights[
    y_train.astype(int)
]

sampler = WeightedRandomSampler(

    weights=sample_weights,

    num_samples=len(sample_weights),

    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=sampler,

    num_workers=0,

    pin_memory=False
)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=False
)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=False
)


# ============================================================
# BCE LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# MODEL
# ============================================================

class ChunkCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(16)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256 * 16,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                1
            )
        )

    def forward(

        self,
        x
    ):

        x = self.features(x)

        x = self.classifier(x)

        return x.squeeze(1)


model = ChunkCNN().to(
    device
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(

    model,
    loader
):

    model.eval()

    losses = []

    all_probs = []

    all_labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            probs = torch.sigmoid(
                logits
            )

            losses.append(
                loss.item()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    all_probs = np.array(
        all_probs
    )

    all_labels = np.array(
        all_labels
    )

    preds = (
        all_probs >= 0.5
    ).astype(int)

    precision = precision_score(

        all_labels,
        preds,

        zero_division=0
    )

    recall = recall_score(

        all_labels,
        preds,

        zero_division=0
    )

    f1 = f1_score(

        all_labels,
        preds,

        zero_division=0
    )

    pr_auc = average_precision_score(

        all_labels,
        all_probs
    )

    return {

        "loss":
            np.mean(losses),

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "pr_auc":
            pr_auc
    }


# ============================================================
# TRAINING
# ============================================================

best_pr_auc = 0

epochs_without_improvement = 0

for epoch in range(EPOCHS):

    print("\n===================================")
    print(f"EPOCH {epoch+1}/{EPOCHS}")
    print("===================================")

    model.train()

    train_losses = []

    for x, y in tqdm(train_loader):

        x = x.to(device)

        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        train_losses.append(
            loss.item()
        )

    print("\nTRAIN LOSS:")
    print(
        np.mean(train_losses)
    )

    metrics = evaluate(

        model,
        val_loader
    )

    print("\nVALIDATION:")

    for k, v in metrics.items():

        print(f"{k}: {v:.6f}")

    if metrics["pr_auc"] > best_pr_auc:

        best_pr_auc = metrics["pr_auc"]

        epochs_without_improvement = 0

        torch.save(

            model.state_dict(),

            "best_chunk_model_tic_split.pt"
        )

        print("\nBEST MODEL SAVED")

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo improvement ({epochs_without_improvement}/{PATIENCE})"
        )

    if epochs_without_improvement >= PATIENCE:

        print("\nEARLY STOPPING")

        break


# ============================================================
# FINAL TEST
# ============================================================

print("\n===================================")
print("FINAL TEST")
print("===================================")

model.load_state_dict(

    torch.load(

        "best_chunk_model_tic_split.pt",

        map_location=device
    )
)

test_metrics = evaluate(

    model,
    test_loader
)

for k, v in test_metrics.items():

    print(f"{k}: {v:.6f}")