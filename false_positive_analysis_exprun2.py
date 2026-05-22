import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


# ============================================================
# VALIDATION FALSE POSITIVE ANALYSIS
# ============================================================

model.eval()

all_probs = []

all_preds = []

all_labels = []

all_flux = []


# ============================================================
# VALIDATION INFERENCE
# ============================================================

with torch.no_grad():

    for flux, labels in val_loader:

        flux_device = flux.to(device)

        logits = model(
            flux_device
        )

        probs = torch.sigmoid(
            logits
        )

        preds = (
            probs >= 0.60
        ).float()

        all_probs.extend(
            probs.cpu().numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        all_flux.extend(
            flux.numpy()
        )


# ============================================================
# ARRAYS
# ============================================================

all_probs = np.array(
    all_probs
)

all_preds = np.array(
    all_preds
)

all_labels = np.array(
    all_labels
)

all_flux = np.array(
    all_flux
)


# ============================================================
# FALSE POSITIVES
# ============================================================

false_positive_mask = (

    (all_preds == 1)

    &

    (all_labels == 0)
)

false_positive_indices = np.where(
    false_positive_mask
)[0]

print("\n===================================")
print("FALSE POSITIVES")
print("===================================")

print(
    "Count:",
    len(false_positive_indices)
)


# ============================================================
# FALSE POSITIVE DATAFRAME
# ============================================================

val_reset = val_df.reset_index(
    drop=True
)

false_positive_df = (
    val_reset.iloc[
        false_positive_indices
    ].copy()
)

false_positive_df[
    "predicted_probability"
] = all_probs[
    false_positive_indices
]


# ============================================================
# SORT BY CONFIDENCE
# ============================================================

false_positive_df = (
    false_positive_df.sort_values(

        "predicted_probability",

        ascending=False
    )
)


# ============================================================
# SAVE
# ============================================================

false_positive_df.to_csv(

    "validation_false_positives.csv",

    index=False
)

print("\n===================================")
print("SAVED")
print("===================================")

print(
    "validation_false_positives.csv"
)


# ============================================================
# TOP FALSE POSITIVES
# ============================================================

print("\n===================================")
print("TOP FALSE POSITIVES")
print("===================================")

print(

    false_positive_df[

        [

            "tic_id",

            "sector",

            "window_id",

            "predicted_probability"
        ]

    ].head(20)
)


# ============================================================
# PLOT TOP FALSE POSITIVES
# ============================================================

N_PLOTS = 12

top_indices = false_positive_indices[

    np.argsort(

        all_probs[
            false_positive_indices
        ]

    )[::-1]

][:N_PLOTS]


for i, idx in enumerate(top_indices):

    flux = all_flux[idx][0]

    prob = all_probs[idx]

    tic_id = val_reset.iloc[idx][
        "tic_id"
    ]

    sector = val_reset.iloc[idx][
        "sector"
    ]

    window_id = val_reset.iloc[idx][
        "window_id"
    ]

    plt.figure(figsize=(12,4))

    plt.plot(flux)

    plt.title(

        f"FALSE POSITIVE {i+1} | "
        f"TIC {tic_id} | "
        f"Sector {sector} | "
        f"Window {window_id} | "
        f"Prob={prob:.4f}"
    )

    plt.xlabel("Cadence")

    plt.ylabel("Normalized Flux")

    plt.grid(True)

    plt.show()