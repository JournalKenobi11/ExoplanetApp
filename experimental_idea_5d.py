# ============================================================
# HARD POSITIVE BOOSTED TRAINING
# FULL WORKING GOOGLE COLAB CUDA VERSION
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# IMPORTS
# ============================================================

import random
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

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from tqdm import tqdm

# ============================================================
# SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n===================================")
print("DEVICE")
print("===================================")
print(device)

# ============================================================
# PATHS
# ============================================================

TRAIN_DATASET_CSV = (
    "/content/drive/MyDrive/train_split_hard_positive_boosted.csv"
)

FULL_DATASET_CSV = (
    "/content/drive/MyDrive/tess_chunk_system_dataset.csv"
)

MODEL_SAVE_PATH = (
    "/content/drive/MyDrive/best_chunk_model_hard_positive.pt"
)

# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 128

EPOCHS = 50

LEARNING_RATE = 3e-5

WEIGHT_DECAY = 1e-4

PATIENCE = 10

# ============================================================
# LOAD TRAIN DATA
# ============================================================

print("\n===================================")
print("LOADING BOOSTED TRAIN DATA")
print("===================================")

train_df = pd.read_csv(
    TRAIN_DATASET_CSV
)

print(train_df.shape)

print("\nTrain labels:")
print(
    train_df["label"].value_counts()
)

# ============================================================
# LOAD FULL DATASET
# ============================================================

full_df = pd.read_csv(
    FULL_DATASET_CSV
)

flux_cols = [

    c for c in full_df.columns

    if c.startswith("flux_")
]

# ============================================================
# REBUILD TIC SPLIT
# ============================================================

unique_tics = full_df[
    "tic_id"
].unique()

train_tics, temp_tics = train_test_split(

    unique_tics,

    test_size=0.25,

    random_state=SEED
)

val_tics, test_tics = train_test_split(

    temp_tics,

    test_size=0.5,

    random_state=SEED
)

val_df = full_df[
    full_df["tic_id"].isin(val_tics)
].copy()

test_df = full_df[
    full_df["tic_id"].isin(test_tics)
].copy()

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
# DATASET CLASS
# ============================================================

class ChunkDataset(Dataset):

    def __init__(self, X, y):

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

    def __getitem__(self, idx):

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

    num_workers=2,

    pin_memory=True
)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2,

    pin_memory=True
)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2,

    pin_memory=True
)

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
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x.squeeze(1)

model = ChunkCNN().to(device)

# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss()

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)

# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2
)

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

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

    all_probs = np.array(all_probs)

    all_labels = np.array(all_labels)

    pr_auc = average_precision_score(
        all_labels,
        all_probs
    )

    best_threshold = None

    best_precision = 0

    best_recall = 0

    best_f1 = 0

    for threshold in np.arange(
        0.05,
        0.96,
        0.01
    ):

        preds = (
            all_probs >= threshold
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

        if (
            precision >= 0.60
            and
            recall > best_recall
        ):

            best_threshold = threshold

            best_precision = precision

            best_recall = recall

            best_f1 = f1

    return {

        "loss":
            np.mean(losses),

        "precision":
            best_precision,

        "recall":
            best_recall,

        "f1":
            best_f1,

        "pr_auc":
            pr_auc,

        "threshold":
            best_threshold
    }

# ============================================================
# TRAINING
# ============================================================

best_recall = 0

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

    scheduler.step(
        metrics["pr_auc"]
    )

    print("\nVALIDATION:")

    for k, v in metrics.items():

        print(f"{k}: {v}")

    if (
        metrics["threshold"] is not None
        and
        metrics["recall"] > best_recall
    ):

        best_recall = metrics["recall"]

        epochs_without_improvement = 0

        save_dict = {

            "model_state_dict":
                model.state_dict(),

            "threshold":
                metrics["threshold"]
        }

        torch.save(
            save_dict,
            MODEL_SAVE_PATH
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

checkpoint = torch.load(

    MODEL_SAVE_PATH,

    map_location=device,

    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

saved_threshold = checkpoint["threshold"]

print(
    "\nUsing saved threshold:",
    saved_threshold
)

model.eval()

losses = []

all_probs = []

all_labels = []

with torch.no_grad():

    for x, y in test_loader:

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
    all_probs >= saved_threshold
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

print(
    f"loss: {np.mean(losses)}"
)

print(
    f"precision: {precision}"
)

print(
    f"recall: {recall}"
)

print(
    f"f1: {f1}"
)

print(
    f"pr_auc: {pr_auc}"
)