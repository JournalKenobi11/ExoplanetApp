import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (

    precision_score,

    recall_score,

    f1_score
)


# ============================================================
# GET VALIDATION PROBABILITIES
# ============================================================

model.eval()

all_probs = []

all_labels = []

with torch.no_grad():

    for flux, labels in val_loader:

        flux = flux.to(device)

        logits = model(flux)

        probs = torch.sigmoid(
            logits
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
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

thresholds = np.arange(

    0.05,

    0.96,

    0.05
)

results = []


for threshold in thresholds:

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

    results.append({

        "threshold":
            threshold,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

threshold_df = pd.DataFrame(
    results
)

print("\n===================================")
print("THRESHOLD RESULTS")
print("===================================")

print(
    threshold_df
)


# ============================================================
# BEST F1
# ============================================================

best_idx = threshold_df[
    "f1"
].idxmax()

best_row = threshold_df.loc[
    best_idx
]

print("\n===================================")
print("BEST THRESHOLD")
print("===================================")

print(best_row)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(

    threshold_df["threshold"],

    threshold_df["precision"],

    label="Precision"
)

plt.plot(

    threshold_df["threshold"],

    threshold_df["recall"],

    label="Recall"
)

plt.plot(

    threshold_df["threshold"],

    threshold_df["f1"],

    label="F1"
)

plt.xlabel("Threshold")

plt.ylabel("Metric")

plt.title(
    "Threshold Optimization"
)

plt.grid(True)

plt.legend()

plt.show()