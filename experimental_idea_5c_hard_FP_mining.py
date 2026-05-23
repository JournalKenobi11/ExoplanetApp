# ============================================================
# HARD FALSE POSITIVE MINING
# FROM TRAINING NEGATIVES ONLY
# ============================================================

print("\n===================================")
print("MINING HARD FALSE POSITIVES")
print("===================================")

model.eval()

# ============================================================
# ONLY TRAINING NEGATIVES
# ============================================================

train_negative_df = train_df[
    train_df["label"] == 0
].copy()

X_neg = train_negative_df[
    flux_cols
].values.astype(np.float32)

X_neg_tensor = torch.tensor(
    X_neg,
    dtype=torch.float32
).unsqueeze(1)

all_probs = []

# ============================================================
# INFERENCE
# ============================================================

with torch.no_grad():

    for i in tqdm(

        range(
            0,
            len(X_neg_tensor),
            512
        )
    ):

        batch = X_neg_tensor[
            i:i+512
        ].to(device)

        logits = model(batch)

        probs = torch.sigmoid(
            logits
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

train_negative_df["pred_prob"] = all_probs

# ============================================================
# HARD FALSE POSITIVES
# ============================================================

hard_fp_df = train_negative_df[

    train_negative_df["pred_prob"] >= 0.80

].copy()

print("\n===================================")
print("HARD FALSE POSITIVES")
print("===================================")

print(
    "Count:",
    len(hard_fp_df)
)

# ============================================================
# SAVE HARD FPS
# ============================================================

hard_fp_df.to_csv(

    "/content/drive/MyDrive/hard_false_positives.csv",

    index=False
)

print("\nSaved:")

print(
    "/content/drive/MyDrive/hard_false_positives.csv"
)

# ============================================================
# CREATE BOOSTED TRAIN SET
# ============================================================

BOOST_FACTOR = 2

boosted_hard_fps = pd.concat(

    [hard_fp_df] * BOOST_FACTOR,

    ignore_index=True
)

boosted_train_df = pd.concat(

    [

        train_df,

        boosted_hard_fps

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
# SAVE BOOSTED DATASET
# ============================================================

boosted_train_df.to_csv(

    "/content/drive/MyDrive/train_split_hard_fp_boosted.csv",

    index=False
)

print("\n===================================")
print("BOOSTED TRAIN DATASET CREATED")
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

print(
    "/content/drive/MyDrive/train_split_hard_fp_boosted.csv"
)