import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    "largest_chunk_statistics.csv"
)


# ============================================================
# DATASETS
# ============================================================

datasets = sorted(
    df["dataset"].unique()
)


# ============================================================
# HISTOGRAMS
# ============================================================

for dataset_name in datasets:

    subset = df[
        df["dataset"] == dataset_name
    ]

    plt.figure(figsize=(10, 5))

    plt.hist(
        subset["chunk_points"],
        bins=60
    )

    plt.title(
        f"Chunk Size Distribution: {dataset_name}"
    )

    plt.xlabel(
        "Largest Continuous Chunk Points"
    )

    plt.ylabel(
        "Sector Count"
    )

    plt.grid(True)

    plt.show()