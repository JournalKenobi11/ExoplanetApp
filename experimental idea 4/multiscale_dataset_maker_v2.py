# ============================================================
# MULTISCALE DATASET MAKER
# EDGE-PADDED CONTEXT VERSION
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
    "tess_multiscale_candidate_dataset.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

LOCAL_WINDOW = 1024

CONTEXT_WINDOW = 4096

POOL_FACTOR = 4

POOLED_CONTEXT_SIZE = (
    CONTEXT_WINDOW // POOL_FACTOR
)

LOCAL_DETREND_KERNEL = 401

MIN_CHUNK_POINTS = 4500

GAP_THRESHOLD_DAYS = 0.02

MAX_CANDIDATES_PER_CHUNK = 20

DEPTH_SIGMA_THRESHOLD = 3.0


# ============================================================
# LOAD PLANET INFO
# ============================================================

planet_df = pd.read_csv(
    PLANET_CSV
)

planet_df["tic_id"] = (
    planet_df["tic_id"]
    .astype(int)
)


# ============================================================
# CSV HEADER
# ============================================================

csv_header = [

    "tic_id",

    "sector",

    "chunk_id",

    "candidate_id",

    "label",

    "candidate_center_btjd",

    "candidate_depth",

    "candidate_snr"
]

for i in range(LOCAL_WINDOW):

    csv_header.append(
        f"local_flux_{i}"
    )

for i in range(POOLED_CONTEXT_SIZE):

    csv_header.append(
        f"context_flux_{i}"
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
# CANDIDATE DETECTION
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
        -DEPTH_SIGMA_THRESHOLD
        * sigma
    )

    candidate_idxs = np.where(
        residual < threshold
    )[0]

    if len(candidate_idxs) == 0:
        return []

    grouped = []

    current_group = [
        candidate_idxs[0]
    ]

    for idx in candidate_idxs[1:]:

        if (
            idx
            -
            current_group[-1]
            <= 10
        ):

            current_group.append(
                idx
            )

        else:

            grouped.append(
                current_group
            )

            current_group = [idx]

    grouped.append(
        current_group
    )

    candidates = []

    for group in grouped:

        center_idx = group[
            np.argmin(
                flux[group]
            )
        ]

        depth = (
            np.median(flux)
            -
            flux[center_idx]
        )

        snr = (
            depth
            /
            (sigma + 1e-8)
        )

        candidates.append({

            "center_idx":
                center_idx,

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

    return candidates[
        :MAX_CANDIDATES_PER_CHUNK
    ]


# ============================================================
# SAFE WINDOW EXTRACTION
# ============================================================

def extract_with_padding(

    flux,

    center_idx,

    window_size
):

    half = (
        window_size // 2
    )

    start = (
        center_idx - half
    )

    end = (
        center_idx + half
    )

    extracted = np.ones(
        window_size,
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

    extracted[
        dst_start:dst_end
    ] = flux[
        src_start:src_end
    ]

    return extracted


# ============================================================
# CONTEXT POOLING
# ============================================================

def pool_context(flux):

    flux = flux.reshape(

        POOLED_CONTEXT_SIZE,

        POOL_FACTOR
    )

    pooled = np.median(

        flux,

        axis=1
    )

    return pooled.astype(
        np.float32
    )


# ============================================================
# TRANSIT LABEL
# ============================================================

def candidate_contains_transit(

    candidate_time,

    period,
    t0_btjd,
    duration_days
):

    n_start = int(

        np.floor(

            (
                candidate_time
                -
                duration_days
                -
                t0_btjd
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
                duration_days
                -
                t0_btjd
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

            t0_btjd
            +
            n * period
        )

        if (

            abs(
                candidate_time
                -
                transit_center
            )

            <=

            duration_days
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

    candidate_center_btjd,

    candidate_depth,

    candidate_snr,

    local_flux,

    context_flux
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

        candidate_center_btjd,

        candidate_depth,

        candidate_snr
    ]

    row.extend(local_flux)

    row.extend(context_flux)

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

    label_mode,

    period=None,
    t0_btjd=None,
    duration_days=None
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

            candidates = detect_candidates(
                norm_flux
            )

            for candidate_id, candidate in enumerate(candidates):

                center_idx = candidate[
                    "center_idx"
                ]

                # ====================================
                # LOCAL
                # ====================================

                local_flux = extract_with_padding(

                    norm_flux,

                    center_idx,

                    LOCAL_WINDOW
                )

                # ====================================
                # CONTEXT
                # ====================================

                context_flux = extract_with_padding(

                    norm_flux,

                    center_idx,

                    CONTEXT_WINDOW
                )

                context_flux = pool_context(
                    context_flux
                )

                candidate_time = t_chunk[
                    center_idx
                ]

                if label_mode == "planet":

                    label = int(

                        candidate_contains_transit(

                            candidate_time,

                            period,
                            t0_btjd,
                            duration_days
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

                    candidate["depth"],

                    candidate["snr"],

                    local_flux,

                    context_flux
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

    period = float(
        group.iloc[0]["period"]
    )

    t0_btjd = (

        float(
            group.iloc[0]["t0"]
        )
        -
        2457000
    )

    duration_days = (

        float(
            group.iloc[0]["duration"]
        )
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

            label_mode="planet",

            period=period,
            t0_btjd=t0_btjd,
            duration_days=duration_days
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

                label_mode="negative"
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