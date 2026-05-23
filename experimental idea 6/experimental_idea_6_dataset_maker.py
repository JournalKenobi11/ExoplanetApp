# ============================================================
# NORMAL MULTISCALE DATASET MAKER
# NO FOLDING
# NO CHUNK CLASSIFICATION
#
# REPRESENTATION:
# 1. LOCAL TRANSIT WINDOW
# 2. BROADER CONTEXT WINDOW
#
# THIS IS:
# candidate-centered
# multiscale
# blind-search compatible
# ============================================================

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
    "tess_multiscale_dataset.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

LOCAL_WINDOW = 1024

GLOBAL_WINDOW = 4096

GLOBAL_POOL = 4

GLOBAL_POOLED = (
    GLOBAL_WINDOW // GLOBAL_POOL
)

DETREND_KERNEL = 401

GAP_THRESHOLD = 0.02

MIN_CHUNK_POINTS = 4500

DEPTH_SIGMA = 3.0

MAX_NEGATIVE_CANDIDATES = 3


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


# ============================================================
# HEADER
# ============================================================

header = [

    "tic_id",

    "sector",

    "chunk_id",

    "candidate_id",

    "label",

    "candidate_time",

    "candidate_depth",

    "candidate_snr"
]

for i in range(LOCAL_WINDOW):

    header.append(
        f"local_{i}"
    )

for i in range(GLOBAL_POOLED):

    header.append(
        f"global_{i}"
    )


csv_file = open(

    OUTPUT_CSV,

    "w",

    newline=""
)

writer = csv.writer(
    csv_file
)

writer.writerow(
    header
)


# ============================================================
# COUNTERS
# ============================================================

total_samples = 0

positive_samples = 0

negative_samples = 0


# ============================================================
# NORMALIZE
# ============================================================

def normalize_flux(flux):

    median = np.median(
        flux
    )

    if median == 0:
        return None

    flux = flux / median

    return flux.astype(
        np.float32
    )


# ============================================================
# DETREND
# ============================================================

def detrend_flux(flux):

    trend = median_filter(

        flux,

        size=DETREND_KERNEL,

        mode="nearest"
    )

    detrended = (
        flux / trend
    )

    return detrended


# ============================================================
# SPLIT CHUNKS
# ============================================================

def split_chunks(

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

    split_idx = np.where(

        gaps > GAP_THRESHOLD

    )[0]

    split_points = (
        split_idx + 1
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

    for t, f in zip(

        time_chunks,
        flux_chunks
    ):

        if len(t) < MIN_CHUNK_POINTS:
            continue

        chunks.append((t, f))

    return chunks


# ============================================================
# DETECT CANDIDATES
# ============================================================

def detect_candidates(flux):

    detrended = detrend_flux(
        flux
    )

    residual = (
        detrended
        -
        np.median(detrended)
    )

    sigma = np.std(
        residual
    )

    threshold = (
        -DEPTH_SIGMA * sigma
    )

    candidate_idx = np.where(
        residual < threshold
    )[0]

    if len(candidate_idx) == 0:
        return []

    groups = []

    current = [
        candidate_idx[0]
    ]

    for idx in candidate_idx[1:]:

        if (
            idx
            -
            current[-1]
            <= 10
        ):

            current.append(idx)

        else:

            groups.append(current)

            current = [idx]

    groups.append(current)

    candidates = []

    for group in groups:

        center = group[
            np.argmin(
                flux[group]
            )
        ]

        depth = (
            np.median(flux)
            -
            flux[center]
        )

        snr = (
            depth
            /
            (sigma + 1e-8)
        )

        candidates.append({

            "center":
                center,

            "depth":
                depth,

            "snr":
                snr
        })

    candidates = sorted(

        candidates,

        key=lambda x: x["snr"],

        reverse=True
    )

    return candidates


# ============================================================
# PAD EXTRACTION
# ============================================================

def extract_window(

    flux,

    center,

    size
):

    half = size // 2

    start = center - half

    end = center + half

    output = np.ones(
        size,
        dtype=np.float32
    )

    src_start = max(
        0,
        start
    )

    src_end = min(
        len(flux),
        end
    )

    dst_start = (
        src_start - start
    )

    dst_end = (
        dst_start
        +
        (src_end - src_start)
    )

    output[
        dst_start:dst_end
    ] = flux[
        src_start:src_end
    ]

    return output


# ============================================================
# POOL GLOBAL
# ============================================================

def pool_global(flux):

    flux = flux.reshape(

        GLOBAL_POOLED,

        GLOBAL_POOL
    )

    pooled = np.median(

        flux,

        axis=1
    )

    return pooled.astype(
        np.float32
    )


# ============================================================
# LABEL TRANSIT
# ============================================================

def contains_transit(

    candidate_time,

    period,
    t0,
    duration
):

    n_start = int(

        np.floor(

            (
                candidate_time
                -
                duration
                -
                t0
            )
            /
            period
        )

    ) - 1

    n_end = int(

        np.ceil(

            (
                candidate_time
                +
                duration
                -
                t0
            )
            /
            period
        )

    ) + 1

    for n in range(

        n_start,
        n_end + 1
    ):

        transit_center = (
            t0
            +
            n * period
        )

        if (

            abs(
                candidate_time
                -
                transit_center
            )

            <= duration
        ):

            return True

    return False


# ============================================================
# WRITE SAMPLE
# ============================================================

def write_sample(

    tic_id,
    sector,
    chunk_id,
    candidate_id,

    label,

    candidate_time,

    depth,
    snr,

    local_view,
    global_view
):

    global total_samples
    global positive_samples
    global negative_samples

    row = [

        tic_id,

        sector,

        chunk_id,

        candidate_id,

        label,

        candidate_time,

        depth,
        snr
    ]

    row.extend(local_view)

    row.extend(global_view)

    writer.writerow(
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

    label_mode,

    period=None,
    t0=None,
    duration=None
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

        chunks = split_chunks(
            time,
            flux
        )

        for chunk_id, (
            t_chunk,
            f_chunk
        ) in enumerate(chunks):

            flux_norm = normalize_flux(
                f_chunk
            )

            if flux_norm is None:
                continue

            candidates = detect_candidates(
                flux_norm
            )

            if label_mode != "planet":

                candidates = candidates[
                    :MAX_NEGATIVE_CANDIDATES
                ]

            for candidate_id, c in enumerate(candidates):

                center = c["center"]

                local_view = extract_window(

                    flux_norm,

                    center,

                    LOCAL_WINDOW
                )

                global_view = extract_window(

                    flux_norm,

                    center,

                    GLOBAL_WINDOW
                )

                global_view = pool_global(
                    global_view
                )

                candidate_time = t_chunk[
                    center
                ]

                if label_mode == "planet":

                    label = int(

                        contains_transit(

                            candidate_time,

                            period,
                            t0,
                            duration
                        )
                    )

                else:

                    label = 0

                write_sample(

                    tic_id,
                    sector,
                    chunk_id,
                    candidate_id,

                    label,

                    candidate_time,

                    c["depth"],
                    c["snr"],

                    local_view,
                    global_view
                )

    except Exception as e:

        print(
            "ERROR:",
            fits_path,
            e
        )


# ============================================================
# PROCESS POSITIVES
# ============================================================

print("\n===================================")
print("PROCESSING POSITIVES")
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

    period = float(
        group.iloc[0]["period"]
    )

    t0 = (
        float(group.iloc[0]["t0"])
        -
        2457000
    )

    duration = (
        float(group.iloc[0]["duration"])
        / 24.0
    )

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

            "planet",

            period,
            t0,
            duration
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

                "negative"
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