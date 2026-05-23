import os
import csv
import numpy as np
import pandas as pd

from tqdm import tqdm
from astropy.io import fits
from scipy.ndimage import median_filter


# ============================================================
# CONFIG
# ============================================================

TESS_ARCHIVE = (
    r"C:\Users\aasha\OneDrive\Desktop\correction\tess_archive"
)

PLANET_CSV = (
    "final_confirmed_planets.csv"
)

NEGATIVE_DATASETS = [

    ("final_toi_false_positives.csv", "toi_fp"),

    ("final_eclipsing_binaries.csv", "eb"),

    ("final_random_noisy_stars.csv", "random")
]

OUTPUT_CSV = (
    "tess_chunk_system_dataset.csv"
)


# ============================================================
# SCIENTIFICALLY DERIVED PARAMETERS
# ============================================================

MIN_CHUNK_POINTS = 4500

MAX_CHUNK_POINTS = 9000

POOL_SIZE = 8

POOLED_LENGTH = (
    MAX_CHUNK_POINTS // POOL_SIZE
)

LOCAL_DETREND_KERNEL = 401

GAP_THRESHOLD_DAYS = 0.02


# ============================================================
# LOAD PLANETS
# ============================================================

planet_df = pd.read_csv(
    PLANET_CSV
)

planet_df["tic_id"] = (
    planet_df["tic_id"]
    .astype(int)
)

planet_tics = set(
    planet_df["tic_id"]
)


# ============================================================
# CSV HEADER
# ============================================================

csv_header = [

    "tic_id",

    "sector",

    "chunk_id",

    "label",

    "chunk_length"
]

for i in range(POOLED_LENGTH):

    csv_header.append(
        f"flux_{i}"
    )


csv_file = open(

    OUTPUT_CSV,

    "w",

    newline=""
)

csv_writer = csv.writer(
    csv_file
)

csv_writer.writerow(
    csv_header
)


# ============================================================
# COUNTERS
# ============================================================

total_samples = 0

positive_samples = 0

negative_samples = 0


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_flux(flux):

    median_flux = np.median(
        flux
    )

    if median_flux == 0:
        return None

    flux = (
        flux / median_flux
    )

    return flux.astype(
        np.float32
    )


# ============================================================
# SPLIT CHUNKS
# ============================================================

def split_continuous_chunks(

    time,
    flux
):

    valid = (

        np.isfinite(time)

        &

        np.isfinite(flux)
    )

    time = time[valid]

    flux = flux[valid]

    if len(time) < 2:
        return []

    order = np.argsort(
        time
    )

    time = time[order]

    flux = flux[order]

    gaps = np.diff(time)

    break_idxs = np.where(

        gaps > GAP_THRESHOLD_DAYS

    )[0]

    split_points = (
        break_idxs + 1
    )

    time_chunks = np.split(

        time,
        split_points
    )

    flux_chunks = np.split(

        flux,
        split_points
    )

    chunks = []

    for t_chunk, f_chunk in zip(

        time_chunks,
        flux_chunks
    ):

        if (
            len(t_chunk)
            <
            MIN_CHUNK_POINTS
        ):
            continue

        chunks.append(
            (t_chunk, f_chunk)
        )

    return chunks


# ============================================================
# DETREND
# ============================================================

def detrend_flux(flux):

    trend = median_filter(

        flux,

        size=LOCAL_DETREND_KERNEL,

        mode="nearest"
    )

    detrended = (
        flux / trend
    )

    return detrended


# ============================================================
# CENTER CROP
# ============================================================

def center_crop_chunk(flux):

    if len(flux) <= MAX_CHUNK_POINTS:

        return flux

    center = len(flux) // 2

    half = MAX_CHUNK_POINTS // 2

    start = center - half

    end = center + half

    return flux[start:end]


# ============================================================
# PAD TO FIXED SIZE
# ============================================================

def edge_pad_chunk(flux):

    padded = np.ones(
        MAX_CHUNK_POINTS,
        dtype=np.float32
    )

    length = min(
        len(flux),
        MAX_CHUNK_POINTS
    )

    padded[:length] = flux[:length]

    return padded


# ============================================================
# POOL CHUNK
# ============================================================

def median_pool_chunk(flux):

    flux = flux.reshape(

        POOLED_LENGTH,

        POOL_SIZE
    )

    pooled = np.median(

        flux,

        axis=1
    )

    return pooled.astype(
        np.float32
    )


# ============================================================
# WRITE SAMPLE
# ============================================================

def write_sample(

    tic_id,
    sector,
    chunk_id,
    label,
    chunk_length,
    pooled_flux
):

    global total_samples
    global positive_samples
    global negative_samples

    row = [

        tic_id,

        sector,

        chunk_id,

        label,

        chunk_length
    ]

    row.extend(
        pooled_flux
    )

    csv_writer.writerow(
        row
    )

    total_samples += 1

    if label == 1:

        positive_samples += 1

    else:

        negative_samples += 1


# ============================================================
# PROCESS STAR
# ============================================================

def process_star(

    tic_id,

    fits_path,

    label
):

    try:

        with fits.open(
            fits_path
        ) as hdul:

            sector = int(

                hdul[0]
                .header["SECTOR"]
            )

            data = hdul[1].data

            time = data["TIME"]

            flux = data[
                "PDCSAP_FLUX"
            ]

        chunks = split_continuous_chunks(

            time,
            flux
        )

        for chunk_id, (
            t_chunk,
            f_chunk
        ) in enumerate(chunks):

            norm_flux = normalize_flux(
                f_chunk
            )

            if norm_flux is None:
                continue

            detrended = detrend_flux(
                norm_flux
            )

            chunk_length = len(
                detrended
            )

            cropped = center_crop_chunk(
                detrended
            )

            padded = edge_pad_chunk(
                cropped
            )

            pooled = median_pool_chunk(
                padded
            )

            write_sample(

                tic_id,
                sector,
                chunk_id,
                label,
                chunk_length,
                pooled
            )

    except Exception as e:

        print(
            "ERROR:",
            fits_path,
            e
        )


# ============================================================
# PROCESS PLANETS
# ============================================================

print("\n===================================")
print("PROCESSING PLANETS")
print("===================================")

planet_groups = planet_df.groupby(
    "tic_id"
)

for tic_id, group in tqdm(
    planet_groups
):

    tic_dir = os.path.join(

        TESS_ARCHIVE,

        f"TIC_{tic_id}"
    )

    if not os.path.exists(
        tic_dir
    ):
        continue

    fits_files = [

        f for f in os.listdir(
            tic_dir
        )

        if f.endswith(".fits")
    ]

    for fits_file in fits_files:

        fits_path = os.path.join(

            tic_dir,
            fits_file
        )

        process_star(

            tic_id,

            fits_path,

            label=1
        )


# ============================================================
# PROCESS NEGATIVES
# ============================================================

print("\n===================================")
print("PROCESSING NEGATIVES")
print("===================================")

for neg_csv, neg_name in NEGATIVE_DATASETS:

    print("\n---", neg_name, "---")

    neg_df = pd.read_csv(
        neg_csv
    )

    grouped = neg_df.groupby(
        "tic_id"
    )

    for tic_id, group in tqdm(grouped):

        tic_id = int(
            tic_id
        )

        tic_dir = os.path.join(

            TESS_ARCHIVE,

            f"TIC_{tic_id}"
        )

        if not os.path.exists(
            tic_dir
        ):
            continue

        fits_files = [

            f for f in os.listdir(
                tic_dir
            )

            if f.endswith(".fits")
        ]

        for fits_file in fits_files:

            fits_path = os.path.join(

                tic_dir,
                fits_file
            )

            process_star(

                tic_id,

                fits_path,

                label=0
            )


# ============================================================
# CLOSE
# ============================================================

csv_file.close()


# ============================================================
# FINAL STATS
# ============================================================

print("\n===================================")
print("DONE")
print("===================================")

print(
    "\nTotal samples:",
    total_samples
)

print(
    "\nPositive samples:",
    positive_samples
)

print(
    "\nNegative samples:",
    negative_samples
)

print(
    "\nSaved:",
    OUTPUT_CSV
)
