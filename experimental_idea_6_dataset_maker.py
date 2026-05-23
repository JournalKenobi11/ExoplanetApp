# ============================================================
# FOLDED MULTI-VIEW DATASET MAKER
# CUDA VERSION
#
# REPRESENTATION:
# 1. GLOBAL FOLDED VIEW
# 2. LOCAL TRANSIT VIEW
# 3. ODD FOLDED VIEW
# 4. EVEN FOLDED VIEW
#
# SCIENTIFIC BASIS:
# - Astronet-style vetting
# - EB secondary eclipse detection
# - odd/even asymmetry
# - phase-aligned recurrence structure
# ============================================================

import os
import csv
import numpy as np
import pandas as pd

from tqdm import tqdm
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d


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
    "tess_folded_multiview_dataset.csv"
)


# ============================================================
# REPRESENTATION PARAMETERS
# ============================================================

GLOBAL_BINS = 2001

LOCAL_BINS = 201

ODD_EVEN_BINS = 201

LOCAL_PHASE_WIDTH = 0.04

DETREND_KERNEL = 401

MIN_POINTS = 3000


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
# CSV HEADER
# ============================================================

header = [

    "tic_id",

    "sector",

    "label",

    "period",

    "duration"
]

for i in range(GLOBAL_BINS):

    header.append(
        f"global_{i}"
    )

for i in range(LOCAL_BINS):

    header.append(
        f"local_{i}"
    )

for i in range(ODD_EVEN_BINS):

    header.append(
        f"odd_{i}"
    )

for i in range(ODD_EVEN_BINS):

    header.append(
        f"even_{i}"
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
# CLEAN
# ============================================================

def clean_lightcurve(

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

    if len(time) < MIN_POINTS:
        return None, None

    order = np.argsort(
        time
    )

    time = time[order]

    flux = flux[order]

    return time, flux


# ============================================================
# PHASE FOLD
# ============================================================

def phase_fold(

    time,
    flux,

    period,
    t0
):

    phase = (
        (
            time - t0 + 0.5 * period
        )
        %
        period
    ) / period

    phase -= 0.5

    order = np.argsort(
        phase
    )

    return phase[order], flux[order]


# ============================================================
# BIN PHASE CURVE
# ============================================================

def bin_phase_curve(

    phase,
    flux,

    bins,

    phase_min,
    phase_max
):

    edges = np.linspace(

        phase_min,
        phase_max,

        bins + 1
    )

    centers = (
        edges[:-1]
        +
        edges[1:]
    ) / 2

    digitized = np.digitize(
        phase,
        edges
    )

    binned = np.ones(
        bins,
        dtype=np.float32
    )

    for i in range(1, bins + 1):

        mask = (
            digitized == i
        )

        if np.sum(mask) > 0:

            binned[i - 1] = np.median(
                flux[mask]
            )

    return binned


# ============================================================
# ODD EVEN SPLIT
# ============================================================

def odd_even_fold(

    time,
    flux,

    period,
    t0
):

    epoch_number = np.floor(
        (time - t0) / period
    ).astype(int)

    odd_mask = (
        epoch_number % 2 == 1
    )

    even_mask = (
        epoch_number % 2 == 0
    )

    odd_phase, odd_flux = phase_fold(

        time[odd_mask],
        flux[odd_mask],

        period,
        t0
    )

    even_phase, even_flux = phase_fold(

        time[even_mask],
        flux[even_mask],

        period,
        t0
    )

    return (

        odd_phase,
        odd_flux,

        even_phase,
        even_flux
    )


# ============================================================
# WRITE SAMPLE
# ============================================================

def write_sample(

    tic_id,
    sector,

    label,

    period,
    duration,

    global_view,
    local_view,

    odd_view,
    even_view
):

    global total_samples
    global positive_samples
    global negative_samples

    row = [

        tic_id,

        sector,

        label,

        period,

        duration
    ]

    row.extend(global_view)

    row.extend(local_view)

    row.extend(odd_view)

    row.extend(even_view)

    writer.writerow(
        row
    )

    total_samples += 1

    if label == 1:

        positive_samples += 1

    else:

        negative_samples += 1


# ============================================================
# PROCESS POSITIVE
# ============================================================

def process_positive(

    tic_id,

    fits_path,

    period,
    t0,
    duration
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

        time, flux = clean_lightcurve(
            time,
            flux
        )

        if time is None:
            return

        flux = normalize_flux(
            flux
        )

        flux = detrend_flux(
            flux
        )

        phase, folded_flux = phase_fold(

            time,
            flux,

            period,
            t0
        )

        global_view = bin_phase_curve(

            phase,
            folded_flux,

            GLOBAL_BINS,

            -0.5,
            0.5
        )

        local_view = bin_phase_curve(

            phase,
            folded_flux,

            LOCAL_BINS,

            -LOCAL_PHASE_WIDTH,
            LOCAL_PHASE_WIDTH
        )

        (
            odd_phase,
            odd_flux,

            even_phase,
            even_flux

        ) = odd_even_fold(

            time,
            flux,

            period,
            t0
        )

        odd_view = bin_phase_curve(

            odd_phase,
            odd_flux,

            ODD_EVEN_BINS,

            -LOCAL_PHASE_WIDTH,
            LOCAL_PHASE_WIDTH
        )

        even_view = bin_phase_curve(

            even_phase,
            even_flux,

            ODD_EVEN_BINS,

            -LOCAL_PHASE_WIDTH,
            LOCAL_PHASE_WIDTH
        )

        write_sample(

            tic_id,
            sector,

            1,

            period,
            duration,

            global_view,
            local_view,

            odd_view,
            even_view
        )

    except Exception as e:

        print(
            "ERROR:",
            fits_path,
            e
        )


# ============================================================
# PROCESS NEGATIVE
# ============================================================

def process_negative(

    tic_id,

    fits_path
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

        time, flux = clean_lightcurve(
            time,
            flux
        )

        if time is None:
            return

        flux = normalize_flux(
            flux
        )

        flux = detrend_flux(
            flux
        )

        duration_days = (
            time[-1]
            -
            time[0]
        )

        candidate_periods = np.linspace(

            0.5,
            min(20, duration_days / 2),

            5
        )

        for period in candidate_periods:

            t0 = time[0]

            phase, folded_flux = phase_fold(

                time,
                flux,

                period,
                t0
            )

            global_view = bin_phase_curve(

                phase,
                folded_flux,

                GLOBAL_BINS,

                -0.5,
                0.5
            )

            local_view = bin_phase_curve(

                phase,
                folded_flux,

                LOCAL_BINS,

                -LOCAL_PHASE_WIDTH,
                LOCAL_PHASE_WIDTH
            )

            (
                odd_phase,
                odd_flux,

                even_phase,
                even_flux

            ) = odd_even_fold(

                time,
                flux,

                period,
                t0
            )

            odd_view = bin_phase_curve(

                odd_phase,
                odd_flux,

                ODD_EVEN_BINS,

                -LOCAL_PHASE_WIDTH,
                LOCAL_PHASE_WIDTH
            )

            even_view = bin_phase_curve(

                even_phase,
                even_flux,

                ODD_EVEN_BINS,

                -LOCAL_PHASE_WIDTH,
                LOCAL_PHASE_WIDTH
            )

            write_sample(

                tic_id,
                sector,

                0,

                period,
                0,

                global_view,
                local_view,

                odd_view,
                even_view
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

        process_positive(

            tic_id,

            fits_path,

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

            process_negative(

                tic_id,

                fits_path
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