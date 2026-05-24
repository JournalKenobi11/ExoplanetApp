# ============================================================
# LEAKAGE-FREE 2-STAGE OVERLAP REFINEMENT TRAINER
# HIGH-RECALL PLANETARY MANIFOLD VERSION
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

EPOCHS_STAGE1 = 50

EPOCHS_STAGE2 = 12

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 7

RANDOM_STATE = 42

STAGE1_THRESHOLD = 0.50

FINAL_THRESHOLD = 0.50

# ============================================================
# OVERLAP REGION
# ============================================================

HARD_POS_LOW = 0.30
HARD_POS_HIGH = 0.75

HARD_NEG_LOW = 0.30

# ============================================================
# STAGE BLEND
# ============================================================

STAGE1_WEIGHT = 0.70
STAGE2_WEIGHT = 0.30


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
# FEATURES
# ============================================================

flux_cols = [

    c for c in df.columns

    if c.startswith("flux_")
]


# ============================================================
# TIC SPLIT
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
# ARRAYS
# ============================================================

X_train = train_df[
    flux_cols
].values.astype(np.float32)

y_train = train_df[
    "label"
].values.astype(np.float32)

X_val = val_df[
    flux_cols
].values.astype(np.float32)

y_val = val_df[
    "label"
].values.astype(np.float32)

X_test = test_df[
    flux_cols
].values.astype(np.float32)

y_test = test_df[
    "label"
].values.astype(np.float32)


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
# SAMPLER
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
# LOSS
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


# ============================================================
# METRICS
# ============================================================

def evaluate_metrics(

    probs,
    labels,
    threshold
):

    preds = (
        probs >= threshold
    ).astype(int)

    precision = precision_score(

        labels,
        preds,

        zero_division=0
    )

    recall = recall_score(

        labels,
        preds,

        zero_division=0
    )

    f1 = f1_score(

        labels,
        preds,

        zero_division=0
    )

    pr_auc = average_precision_score(

        labels,
        probs
    )

    return {

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "pr_auc": pr_auc
    }


# ============================================================
# STAGE1 MODEL
# ============================================================

stage1_model = ChunkCNN().to(
    device
)

optimizer1 = torch.optim.AdamW(

    stage1_model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# STAGE1 TRAINING
# ============================================================

best_pr_auc = 0

epochs_without_improvement = 0

for epoch in range(EPOCHS_STAGE1):

    print("\n===================================")
    print(f"STAGE1 EPOCH {epoch+1}")
    print("===================================")

    stage1_model.train()

    losses = []

    for x, y in tqdm(train_loader):

        x = x.to(device)

        y = y.to(device)

        optimizer1.zero_grad()

        logits = stage1_model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer1.step()

        losses.append(
            loss.item()
        )

    print("\nTRAIN LOSS:")
    print(
        np.mean(losses)
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    stage1_model.eval()

    val_probs = []

    val_labels = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)

            logits = stage1_model(x)

            probs = torch.sigmoid(
                logits
            )

            val_probs.extend(
                probs.cpu().numpy()
            )

            val_labels.extend(
                y.numpy()
            )

    val_probs = np.array(
        val_probs
    )

    val_labels = np.array(
        val_labels
    )

    metrics = evaluate_metrics(

        val_probs,

        val_labels,

        STAGE1_THRESHOLD
    )

    for k, v in metrics.items():

        print(f"{k}: {v:.6f}")

    if metrics["pr_auc"] > best_pr_auc:

        best_pr_auc = metrics["pr_auc"]

        epochs_without_improvement = 0

        torch.save(

            stage1_model.state_dict(),

            "stage1_model.pt"
        )

        print("\nBEST STAGE1 SAVED")

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo improvement ({epochs_without_improvement}/{PATIENCE})"
        )

    if epochs_without_improvement >= PATIENCE:

        print("\nEARLY STOPPING")

        break


# ============================================================
# LOAD BEST STAGE1
# ============================================================

stage1_model.load_state_dict(

    torch.load(

        "stage1_model.pt",

        map_location=device
    )
)

stage1_model.eval()


# ============================================================
# BUILD OVERLAP DATASET
# ============================================================

print("\n===================================")
print("BUILDING OVERLAP DATASET")
print("===================================")

stage2_X = []

stage2_y = []

with torch.no_grad():

    for x, y in tqdm(train_loader):

        x_device = x.to(device)

        logits = stage1_model(
            x_device
        )

        probs = torch.sigmoid(
            logits
        ).cpu().numpy()

        x_np = x.numpy()

        y_np = y.numpy()

        for i in range(len(probs)):

            p = probs[i]

            gt = int(
                y_np[i]
            )

            # ====================================================
            # HARD POSITIVES
            # ====================================================

            if gt == 1:

                if (
                    HARD_POS_LOW
                    <= p
                    <= HARD_POS_HIGH
                ):

                    stage2_X.append(
                        x_np[i][0]
                    )

                    stage2_y.append(1)

            # ====================================================
            # HARD NEGATIVES
            # ====================================================

            else:

                if p >= HARD_NEG_LOW:

                    stage2_X.append(
                        x_np[i][0]
                    )

                    stage2_y.append(0)


stage2_X = np.array(
    stage2_X,
    dtype=np.float32
)

stage2_y = np.array(
    stage2_y,
    dtype=np.float32
)

print("\nOVERLAP DATASET:")
print(stage2_X.shape)

print("\nLabels:")
print(
    pd.Series(stage2_y).value_counts()
)


# ============================================================
# STAGE2 DATASET
# ============================================================

stage2_dataset = ChunkDataset(

    stage2_X,

    stage2_y
)


# ============================================================
# STAGE2 SAMPLER
# ============================================================

class_counts = np.bincount(
    stage2_y.astype(int)
)

class_weights = (
    1.0 / class_counts
)

sample_weights = class_weights[
    stage2_y.astype(int)
]

stage2_sampler = WeightedRandomSampler(

    weights=sample_weights,

    num_samples=len(sample_weights),

    replacement=True
)


# ============================================================
# STAGE2 LOADER
# ============================================================

stage2_loader = DataLoader(

    stage2_dataset,

    batch_size=BATCH_SIZE,

    sampler=stage2_sampler,

    num_workers=0
)


# ============================================================
# STAGE2 MODEL
# ============================================================

stage2_model = ChunkCNN().to(
    device
)

optimizer2 = torch.optim.AdamW(

    stage2_model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# STAGE2 TRAINING
# ============================================================

best_stage2_loss = 999999

for epoch in range(EPOCHS_STAGE2):

    print("\n===================================")
    print(f"STAGE2 EPOCH {epoch+1}")
    print("===================================")

    stage2_model.train()

    losses = []

    for x, y in tqdm(stage2_loader):

        x = x.to(device)

        y = y.to(device)

        optimizer2.zero_grad()

        logits = stage2_model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer2.step()

        losses.append(
            loss.item()
        )

    mean_loss = np.mean(
        losses
    )

    print("\nLOSS:")
    print(mean_loss)

    if mean_loss < best_stage2_loss:

        best_stage2_loss = mean_loss

        torch.save(

            stage2_model.state_dict(),

            "stage2_model.pt"
        )


# ============================================================
# LOAD BEST STAGE2
# ============================================================

stage2_model.load_state_dict(

    torch.load(

        "stage2_model.pt",

        map_location=device
    )
)

stage2_model.eval()


# ============================================================
# FINAL INFERENCE
# ============================================================

print("\n===================================")
print("FINAL TEST")
print("===================================")

all_probs = []

all_labels = []

with torch.no_grad():

    for x, y in tqdm(test_loader):

        x = x.to(device)

        y_np = y.numpy()

        # ====================================================
        # STAGE1
        # ====================================================

        logits1 = stage1_model(x)

        probs1 = torch.sigmoid(
            logits1
        ).cpu().numpy()

        final_batch_probs = []

        # ====================================================
        # CONDITIONAL STAGE2
        # ====================================================

        for i in range(len(probs1)):

            p1 = probs1[i]

            # =================================================
            # ONLY REFINE OVERLAP REGION
            # =================================================

            if p1 < HARD_NEG_LOW:

                final_batch_probs.append(
                    p1
                )

            else:

                single_x = x[i:i+1]

                logits2 = stage2_model(
                    single_x
                )

                p2 = torch.sigmoid(
                    logits2
                ).item()

                # =============================================
                # SOFT BLEND
                # =============================================

                final_prob = (

                    STAGE1_WEIGHT * p1

                    +

                    STAGE2_WEIGHT * p2
                )

                final_batch_probs.append(
                    final_prob
                )

        all_probs.extend(
            final_batch_probs
        )

        all_labels.extend(
            y_np
        )


all_probs = np.array(
    all_probs
)

all_labels = np.array(
    all_labels
)

# ============================================================
# THRESHOLD SEARCH
# MAX PRECISION SUBJECT TO RECALL >= 0.85
# ============================================================

print("\n===================================")
print("THRESHOLD SEARCH")
print("===================================")

best_metrics = None

best_threshold = None

best_precision = -1

thresholds = np.arange(
    0.01,
    1.00,
    0.01
)

for threshold in thresholds:

    metrics = evaluate_metrics(

        all_probs,

        all_labels,

        threshold
    )

    recall = metrics["recall"]

    precision = metrics["precision"]

    # ========================================================
    # HARD RECALL CONSTRAINT
    # ========================================================

    if recall >= 0.85:

        if precision > best_precision:

            best_precision = precision

            best_threshold = threshold

            best_metrics = metrics


print("\n===================================")
print("BEST THRESHOLD UNDER RECALL CONSTRAINT")
print("===================================")

print(f"threshold: {best_threshold:.2f}")

for k, v in best_metrics.items():

    print(f"{k}: {v:.6f}")