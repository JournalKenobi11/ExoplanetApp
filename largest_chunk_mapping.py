import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from astropy.io import fits


# ============================================================
# CONFIG
# ============================================================

TESS_ARCHIVE = r"C:\Users\aasha\OneDrive\Desktop\correction\tess_archive"

DATASETS = [
    ("confirmed_planet_usable_sectors.csv", 1, "confirmed"),
    ("final_toi_false_positives.csv", 0, "toi_fp"),
    ("final_eclipsing_binaries.csv", 0, "eb"),
    ("final_random_noisy_stars.csv", 0, "random")
]

OUTPUT_CSV = "largest_chunk_statistics.csv"

GAP_THRESHOLD_DAYS = 0.02
# ~28.8 minutes


# ============================================================
# CHUNK ANALYSIS
# ============================================================

def largest_continuous_chunk(time, flux):

    valid = (
        np.isfinite(time) &
        np.isfinite(flux)
    )

    time = time[valid]
    flux = flux[valid]

    if len(time) < 2:
        return None

    order = np.argsort(time)

    time = time[order]
    flux = flux[order]

    gaps = np.diff(time)

    break_idxs = np.where(
        gaps > GAP_THRESHOLD_DAYS
    )[0]

    split_points = break_idxs + 1

    time_chunks = np.split(
        time,
        split_points
    )

    flux_chunks = np.split(
        flux,
        split_points
    )

    largest_idx = np.argmax([
        len(c) for c in time_chunks
    ])

    largest_time = time_chunks[
        largest_idx
    ]

    largest_flux = flux_chunks[
        largest_idx
    ]

    return {

        "chunk_points":
            len(largest_time),

        "chunk_duration_days":
            largest_time.max()
            -
            largest_time.min(),

        "chunk_start":
            largest_time.min(),

        "chunk_end":
            largest_time.max(),

        "largest_gap_days":
            np.max(gaps),

        "num_chunks":
            len(time_chunks)
    }


# ============================================================
# MAIN
# ============================================================

rows = []

for csv_path, label, dataset_name in DATASETS:

    print("\n===================================")
    print("PROCESSING:", dataset_name)
    print("===================================")

    df = pd.read_csv(csv_path)

    confirmed_mode = (
        "fits_file" in df.columns
    )

    grouped = df.groupby("tic_id")

    for tic_id, tic_group in tqdm(grouped):

        try:

            tic_id = int(tic_id)

            tic_dir = os.path.join(
                TESS_ARCHIVE,
                f"TIC_{tic_id}"
            )

            if not os.path.exists(tic_dir):
                continue

            # ------------------------------------------------
            # FITS FILES
            # ------------------------------------------------

            if confirmed_mode:

                fits_files = (
                    tic_group["fits_file"]
                    .dropna()
                    .unique()
                    .tolist()
                )

            else:

                fits_files = list(set([
                    f for f in os.listdir(tic_dir)
                    if f.endswith(".fits")
                ]))

            # ------------------------------------------------
            # PROCESS FITS
            # ------------------------------------------------

            for fits_name in fits_files:

                fits_path = os.path.join(
                    tic_dir,
                    fits_name
                )

                if not os.path.exists(fits_path):
                    continue

                try:

                    with fits.open(fits_path) as hdul:

                        hdr0 = hdul[0].header

                        data = hdul[1].data

                        time = data["TIME"]

                        flux = data["PDCSAP_FLUX"]

                        result = (
                            largest_continuous_chunk(
                                time,
                                flux
                            )
                        )

                        if result is None:
                            continue

                        rows.append({

                            "tic_id":
                                tic_id,

                            "sector":
                                int(hdr0["SECTOR"]),

                            "label":
                                label,

                            "dataset":
                                dataset_name,

                            "fits_file":
                                fits_name,

                            "chunk_points":
                                result["chunk_points"],

                            "chunk_duration_days":
                                result["chunk_duration_days"],

                            "chunk_start":
                                result["chunk_start"],

                            "chunk_end":
                                result["chunk_end"],

                            "largest_gap_days":
                                result["largest_gap_days"],

                            "largest_gap_hours":
                                result["largest_gap_days"] * 24,

                            "num_chunks":
                                result["num_chunks"]
                        })

                except Exception as e:

                    print(
                        "FITS ERROR:",
                        fits_name,
                        e
                    )

        except Exception as e:

            print(
                "ROW ERROR:",
                tic_id,
                e
            )


# ============================================================
# EXPORT
# ============================================================

stats_df = pd.DataFrame(rows)

stats_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n===================================")
print("DONE")
print("===================================")

print("\nSaved:")
print(OUTPUT_CSV)


# ============================================================
# SUMMARY
# ============================================================

print("\n===================================")
print("LARGEST CHUNK STATS")
print("===================================")

print("\nChunk points:")
print(
    stats_df["chunk_points"]
    .describe()
)

print("\nChunk duration days:")
print(
    stats_df["chunk_duration_days"]
    .describe()
)

print("\nLargest gap hours:")
print(
    stats_df["largest_gap_hours"]
    .describe()
)

print("\nNumber of chunks:")
print(
    stats_df["num_chunks"]
    .describe()
)