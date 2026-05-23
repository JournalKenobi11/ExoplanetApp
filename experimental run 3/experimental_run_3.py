import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

import torch.optim as optim

import torch_directml


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = (
    "tess_candidate_dataset.csv"
)

BATCH_SIZE = 128

EPOCHS = 50

LEARNING_RATE = 1e-4

WINDOW_SIZE = 1024

PATIENCE = 5

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
    CSV_PATH
)

print(df.shape)

print("\nLabel distribution:")

print(
    df["label"]
    .value_counts()
)


# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

unique_tics = df["tic_id"].unique()

train_tics, temp_tics = train_test_split(

    unique_tics,

    test_size=0.30,

    random_state=RANDOM_STATE
)

val_tics, test_tics = train_test_split(

    temp_tics,

    test_size=0.50,

    random_state=RANDOM_STATE
)

train_df = df[
    df["tic_id"].isin(train_tics)
].reset_index(drop=True)

val_df = df[
    df["tic_id"].isin(val_tics)
].reset_index(drop=True)

test_df = df[
    df["tic_id"].isin(test_tics)
].reset_index(drop=True)


print("\n===================================")
print("TRAIN")
print("===================================")

print(train_df.shape)

print(
    train_df["label"]
    .value_counts()
)

print("\n===================================")
print("VALIDATION")
print("===================================")

print(val_df.shape)

print(
    val_df["label"]
    .value_counts()
)

print("\n===================================")
print("TEST")
print("===================================")

print(test_df.shape)

print(
    test_df["label"]
    .value_counts()
)


# ============================================================
# DATASET
# ============================================================

class CandidateDataset(Dataset):

    def __init__(

        self,
        dataframe
    ):

        self.df = dataframe

        self.flux_cols = [

            c for c in self.df.columns

            if c.startswith("flux_")
        ]

    def __len__(self):

        return len(self.df)

    def __getitem__(

        self,
        idx
    ):

        row = self.df.iloc[idx]

        flux = row[
            self.flux_cols
        ].values.astype(
            np.float32
        )

        label = np.float32(
            row["label"]
        )

        flux = torch.tensor(
            flux
        ).unsqueeze(0)

        label = torch.tensor(
            label
        )

        return flux, label


# ============================================================
# DATASETS
# ============================================================

train_dataset = CandidateDataset(
    train_df
)

val_dataset = CandidateDataset(
    val_df
)

test_dataset = CandidateDataset(
    test_df
)


# ============================================================
# BALANCED SAMPLER
# ============================================================

train_labels = (
    train_df["label"]
    .values
)

class_counts = np.bincount(
    train_labels
)

class_weights = (
    1.0 / class_counts
)

sample_weights = class_weights[
    train_labels
]

sample_weights = torch.DoubleTensor(
    sample_weights
)

sampler = WeightedRandomSampler(

    sample_weights,

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
# MODEL
# ============================================================

class PlanetCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=11,
                padding=5
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(2),


            nn.Conv1d(
                32,
                64,
                kernel_size=9,
                padding=4
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(2),


            nn.Conv1d(
                64,
                128,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(2),


            nn.Conv1d(
                128,
                256,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),


            nn.Conv1d(
                256,
                256,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(256),

            nn.ReLU()
        )

        self.global_avg_pool = (
            nn.AdaptiveAvgPool1d(1)
        )

        self.global_max_pool = (
            nn.AdaptiveMaxPool1d(1)
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                1
            )
        )

    def forward(

        self,
        x
    ):

        x = self.features(x)

        avg_pool = (
            self.global_avg_pool(x)
            .squeeze(-1)
        )

        max_pool = (
            self.global_max_pool(x)
            .squeeze(-1)
        )

        x = torch.cat(
            [avg_pool, max_pool],
            dim=1
        )

        x = self.classifier(x)

        return x.squeeze(1)


# ============================================================
# MODEL
# ============================================================

model = PlanetCNN().to(
    device
)


# ============================================================
# LOSS
# ============================================================
class FocalLoss(nn.Module):

    def __init__(

        self,

        alpha=0.75,

        gamma=2.0
    ):

        super().__init__()

        self.alpha = alpha

        self.gamma = gamma

    def forward(

        self,
        logits,
        targets
    ):

        bce = nn.functional.binary_cross_entropy_with_logits(

            logits,
            targets,

            reduction="none"
        )

        probs = torch.sigmoid(
            logits
        )

        pt = torch.where(

            targets == 1,

            probs,

            1 - probs
        )

        focal_weight = (

            self.alpha
            *
            (1 - pt).pow(self.gamma)
        )

        loss = (
            focal_weight
            *
            bce
        )

        return loss.mean()
criterion = FocalLoss(
    alpha=0.75,
    gamma=2.0
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4
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

    all_preds = []

    all_labels = []

    with torch.no_grad():

        for flux, labels in loader:

            flux = flux.to(device)

            labels = labels.to(device)

            logits = model(
                flux
            )

            loss = criterion(
                logits,
                labels
            )

            probs = torch.sigmoid(
                logits
            )

            preds = (
                probs >= 0.5
            ).float()

            losses.append(
                loss.item()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    precision = precision_score(

        all_labels,
        all_preds,
        zero_division=0
    )

    recall = recall_score(

        all_labels,
        all_preds,
        zero_division=0
    )

    f1 = f1_score(

        all_labels,
        all_preds,
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

best_pr_auc = 0.0

patience_counter = 0


for epoch in range(EPOCHS):

    print("\n===================================")
    print(f"EPOCH {epoch+1}/{EPOCHS}")
    print("===================================")

    model.train()

    train_losses = []

    for flux, labels in train_loader:

        flux = flux.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(
            flux
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

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

        best_pr_auc = (
            metrics["pr_auc"]
        )

        patience_counter = 0

        torch.save(

            model.state_dict(),

            "best_candidate_model.pt"
        )

        print("\nBEST MODEL SAVED")

    else:

        patience_counter += 1

        print(
            f"\nNo improvement "
            f"({patience_counter}/{PATIENCE})"
        )

    if patience_counter >= PATIENCE:

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

        "best_candidate_model.pt",

        map_location=device
    )
)

test_metrics = evaluate(

    model,
    test_loader
)

for k, v in test_metrics.items():

    print(f"{k}: {v:.6f}")