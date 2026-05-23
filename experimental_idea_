# ============================================================
# HARD POSITIVE MINING
# VERIFIED TIC-SAFE VERSION
# CUDA + GOOGLE COLAB + GOOGLE DRIVE
#
# PURPOSE:
# 1. FIND HARD POSITIVE MORPHOLOGIES IN VALIDATION
# 2. FIND SIMILAR HARD POSITIVES INSIDE TRAINING
# 3. BOOST ONLY TRAINING SAMPLES
#
# THIS PRESERVES TIC-LEVEL SPLIT INTEGRITY
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

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
# DATASET
# ============================================================

DATASET_CSV = (
    "/content/drive/MyDrive/tess_chunk_system_dataset.csv"
)

df = pd.read_csv(
    DATASET_CSV
)

print("\n===================================")
print("DATASET")
print("===================================")

print(df.shape)

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

RANDOM_STATE = 42

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
].copy()

val_df = df[
    df["tic_id"].isin(val_tics)
].copy()

print("\n===================================")
print("TRAIN / VAL")
print("===================================")

print("Train:", train_df.shape)
print("Validation:", val_df.shape)

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

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = (
    "/content/drive/MyDrive/best_chunk_model_hard_fp_x2.pt"
)

model = ChunkCNN().to(device)

checkpoint = torch.load(

    MODEL_PATH,

    map_location=device,

    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

saved_threshold = checkpoint["threshold"]

print("\n===================================")
print("MODEL LOADED")
print("===================================")

print(
    "Saved threshold:",
    saved_threshold
)

model.eval()

# ============================================================
# VALIDATION POSITIVES
# ============================================================

val_positive_df = val_df[

    val_df["label"] == 1

].copy()

print("\n===================================")
print("VALIDATION POSITIVES")
print("===================================")

print(len(val_positive_df))

# ============================================================
# VALIDATION INFERENCE
# ============================================================

X_val_pos = val_positive_df[
    flux_cols
].values.astype(np.float32)

X_val_tensor = torch.tensor(

    X_val_pos,

    dtype=torch.float32

).unsqueeze(1)

all_probs = []

print("\n===================================")
print("RUNNING VALIDATION INFERENCE")
print("===================================")

with torch.no_grad():

    for i in tqdm(

        range(
            0,
            len(X_val_tensor),
            512
        )
    ):

        batch = X_val_tensor[
            i:i+512
        ].to(device)

        logits = model(batch)

        probs = torch.sigmoid(
            logits
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

val_positive_df[
    "pred_prob"
] = all_probs

# ============================================================
# HARD POSITIVES
# ============================================================

hard_positive_df = val_positive_df[

    (val_positive_df["pred_prob"] >= 0.30)

    &

    (val_positive_df["pred_prob"] <= saved_threshold)

].copy()

print("\n===================================")
print("HARD POSITIVE VALIDATION SAMPLES")
print("===================================")

print(len(hard_positive_df))

# ============================================================
# TRAINING POSITIVES
# ============================================================

train_positive_df = train_df[

    train_df["label"] == 1

].copy()

print("\n===================================")
print("TRAINING POSITIVES")
print("===================================")

print(len(train_positive_df))

# ============================================================
# FEATURE MATRICES
# ============================================================

hard_val_features = hard_positive_df[
    flux_cols
].values.astype(np.float32)

train_pos_features = train_positive_df[
    flux_cols
].values.astype(np.float32)

# ============================================================
# COSINE SIMILARITY
# ============================================================

print("\n===================================")
print("COMPUTING SIMILARITY")
print("===================================")

similarity_matrix = cosine_similarity(

    train_pos_features,

    hard_val_features
)

max_similarity = similarity_matrix.max(
    axis=1
)

train_positive_df[
    "similarity"
] = max_similarity

# ============================================================
# SELECT ANALOGOUS HARD POSITIVES
# ============================================================

SIMILARITY_THRESHOLD = 0.95

analogous_hard_positives = train_positive_df[

    train_positive_df["similarity"]
    >=
    SIMILARITY_THRESHOLD

].copy()

print("\n===================================")
print("ANALOGOUS TRAIN HARD POSITIVES")
print("===================================")

print(len(analogous_hard_positives))

# ============================================================
# BOOST
# ============================================================

BOOST_FACTOR = 2

boosted_hard_positives = pd.concat(

    [analogous_hard_positives] * BOOST_FACTOR,

    ignore_index=True
)

# ============================================================
# CREATE BOOSTED TRAIN SET
# ============================================================

boosted_train_df = pd.concat(

    [

        train_df,

        boosted_hard_positives

    ],

    ignore_index=True
)

# ============================================================
# SHUFFLE
# ============================================================

boosted_train_df = boosted_train_df.sample(

    frac=1,

    random_state=42

).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

SAVE_PATH = (
    "/content/drive/MyDrive/train_split_hard_positive_boosted.csv"
)

boosted_train_df.to_csv(

    SAVE_PATH,

    index=False
)

print("\n===================================")
print("BOOSTED TRAIN DATASET SAVED")
print("===================================")

print(
    "Original train size:",
    len(train_df)
)

print(
    "Boosted train size:",
    len(boosted_train_df)
)

print("\nOriginal labels:")
print(
    train_df["label"].value_counts()
)

print("\nBoosted labels:")
print(
    boosted_train_df["label"].value_counts()
)

print("\nSaved:")
print(SAVE_PATH)