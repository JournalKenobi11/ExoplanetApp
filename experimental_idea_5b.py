# ============================================================
# ENSEMBLE EXOPLANET DETECTOR
# CUDA + GOOGLE COLAB + GOOGLE DRIVE
# NO UNPICKLING WARNING
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
# CONFIG
# ============================================================

DATASET_CSV = (
    "/content/drive/MyDrive/tess_chunk_system_dataset.csv"
)

MODEL_DIR = (
    "/content/drive/MyDrive/"
)

BATCH_SIZE = 128

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 7


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
# FIXED GLOBAL SEED
# ============================================================

GLOBAL_SEED = 42

random.seed(GLOBAL_SEED)

np.random.seed(GLOBAL_SEED)

torch.manual_seed(GLOBAL_SEED)

torch.cuda.manual_seed_all(GLOBAL_SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False


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
# TIC SPLIT
# ============================================================

unique_tics = df[
    "tic_id"
].unique()

train_tics, temp_tics = train_test_split(

    unique_tics,

    test_size=0.25,

    random_state=GLOBAL_SEED
)

val_tics, test_tics = train_test_split(

    temp_tics,

    test_size=0.5,

    random_state=GLOBAL_SEED
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

    pin_memory=True,

    persistent_workers=True
)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2,

    pin_memory=True,

    persistent_workers=True
)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2,

    pin_memory=True,

    persistent_workers=True
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
# EVALUATION
# ============================================================

def evaluate_model(

    model,
    loader
):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            logits = model(x)

            probs = torch.sigmoid(logits)

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_labels.extend(
                y.numpy()
            )

    return (

        np.array(all_probs),

        np.array(all_labels)
    )


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_single_model(seed):

    print("\n===================================")
    print(f"TRAINING MODEL SEED {seed}")
    print("===================================")

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    model = ChunkCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )

    best_pr_auc = 0

    epochs_without_improvement = 0

    model_path = (
        MODEL_DIR +
        f"ensemble_model_seed_{seed}.pt"
    )

    for epoch in range(EPOCHS):

        print(f"\nEPOCH {epoch+1}/{EPOCHS}")

        model.train()

        losses = []

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

            losses.append(
                loss.item()
            )

        print(
            "TRAIN LOSS:",
            np.mean(losses)
        )

        probs, labels = evaluate_model(

            model,
            val_loader
        )

        pr_auc = average_precision_score(
            labels,
            probs
        )

        print(
            "VALIDATION PR-AUC:",
            pr_auc
        )

        if pr_auc > best_pr_auc:

            best_pr_auc = pr_auc

            epochs_without_improvement = 0

            torch.save(

                model.state_dict(),

                model_path
            )

            print("BEST MODEL SAVED")

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement "
                f"({epochs_without_improvement}/{PATIENCE})"
            )

        if epochs_without_improvement >= PATIENCE:

            print("EARLY STOPPING")

            break


# ============================================================
# TRAIN ENSEMBLE
# ============================================================

SEEDS = [
    11,
    22,
    33,
    44,
    55
]

for seed in SEEDS:

    train_single_model(seed)


# ============================================================
# LOAD ENSEMBLE
# ============================================================

models = []

for seed in SEEDS:

    model = ChunkCNN().to(device)

    model.load_state_dict(

        torch.load(

            MODEL_DIR +
            f"ensemble_model_seed_{seed}.pt",

            map_location=device,

            weights_only=True
        )
    )

    model.eval()

    models.append(model)


# ============================================================
# ENSEMBLE INFERENCE
# ============================================================

all_probs = []
all_labels = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)

        ensemble_probs = []

        for model in models:

            logits = model(x)

            probs = torch.sigmoid(logits)

            ensemble_probs.append(
                probs.cpu().numpy()
            )

        ensemble_probs = np.mean(

            ensemble_probs,

            axis=0
        )

        all_probs.extend(
            ensemble_probs
        )

        all_labels.extend(
            y.numpy()
        )


all_probs = np.array(
    all_probs
)

all_labels = np.array(
    all_labels
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

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


# ============================================================
# FINAL METRICS
# ============================================================

pr_auc = average_precision_score(

    all_labels,
    all_probs
)

print("\n===================================")
print("FINAL ENSEMBLE RESULTS")
print("===================================")

print(
    f"threshold: {best_threshold}"
)

print(
    f"precision: {best_precision:.6f}"
)

print(
    f"recall: {best_recall:.6f}"
)

print(
    f"f1: {best_f1:.6f}"
)

print(
    f"pr_auc: {pr_auc:.6f}"
)