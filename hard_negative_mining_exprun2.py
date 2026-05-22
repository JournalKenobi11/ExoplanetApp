import numpy as np
import pandas as pd
import torch


# ============================================================
# HARD NEGATIVE MINING
# ============================================================

model.eval()

negative_probs = []

negative_indices = []


# ============================================================
# GET TRAIN NEGATIVES
# ============================================================

train_negative_df = train_df[
    train_df["label"] == 0
].reset_index(drop=True)

print("\n===================================")
print("TRAIN NEGATIVES")
print("===================================")

print(train_negative_df.shape)


# ============================================================
# DATASET
# ============================================================

hard_dataset = TESSDataset(
    train_negative_df
)

hard_loader = DataLoader(

    hard_dataset,

    batch_size=256,

    shuffle=False,

    num_workers=0
)


# ============================================================
# INFERENCE
# ============================================================

current_idx = 0

with torch.no_grad():

    for flux, labels in hard_loader:

        flux = flux.to(device)

        logits = model(flux)

        probs = torch.sigmoid(
            logits
        )

        probs = probs.cpu().numpy()

        batch_size = len(probs)

        negative_probs.extend(
            probs
        )

        negative_indices.extend(

            range(

                current_idx,

                current_idx + batch_size
            )
        )

        current_idx += batch_size


negative_probs = np.array(
    negative_probs
)

negative_indices = np.array(
    negative_indices
)


# ============================================================
# HARD NEGATIVE THRESHOLD
# ============================================================

HARD_THRESHOLD = 0.70

hard_mask = (
    negative_probs >= HARD_THRESHOLD
)

hard_negative_indices = (
    negative_indices[
        hard_mask
    ]
)

hard_negative_probs = (
    negative_probs[
        hard_mask
    ]
)

print("\n===================================")
print("HARD NEGATIVES")
print("===================================")

print(
    "Count:",
    len(hard_negative_indices)
)

print(
    "Fraction:",
    len(hard_negative_indices)
    /
    len(train_negative_df)
)


# ============================================================
# HARD NEGATIVE DATAFRAME
# ============================================================

hard_negative_df = (
    train_negative_df.iloc[
        hard_negative_indices
    ].copy()
)

hard_negative_df[
    "hard_negative_prob"
] = hard_negative_probs


# ============================================================
# SORT
# ============================================================

hard_negative_df = (
    hard_negative_df.sort_values(

        "hard_negative_prob",

        ascending=False
    )
)


# ============================================================
# SAVE
# ============================================================

hard_negative_df.to_csv(

    "hard_negatives.csv",

    index=False
)

print("\n===================================")
print("SAVED")
print("===================================")

print(
    "hard_negatives.csv"
)


# ============================================================
# INSPECT TOP HARD NEGATIVES
# ============================================================

print("\n===================================")
print("TOP HARD NEGATIVES")
print("===================================")

print(

    hard_negative_df[

        [

            "tic_id",

            "sector",

            "window_id",

            "hard_negative_prob"
        ]

    ].head(20)
)