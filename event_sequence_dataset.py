import os
import csv
import numpy as np
import pandas as pd

from tqdm import tqdm
from astropy.io import fits


# ============================================================
# CONFIG
# ============================================================

TESS_ARCHIVE = (
    r"C:\Users\aasha\OneDrive\Desktop\correction\tess_archive"
)

PLANET_CSV = (
    "final_confirmed_planets.csv"
)

CONFIRMED_SECTORS_CSV = (
    "confirmed_planet_usable_sectors.csv"
)

NEGATIVE_DATASETS = [

    ("final_toi_false_positives.csv", "toi_fp"),

    ("final_eclipsing_binaries.csv", "eb"),

    ("final_random_noisy_stars.csv", "random")
]

OUTPUT_CSV = (
    "tess_sequence_dataset.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

WINDOW_SIZE = 1024

STRIDE = 256

SEQUENCE_LENGTH = 3

CENTER_INDEX = 1

GAP_THRESHOLD_DAYS = 0.02

MIN_CHUNK_POINTS = (
    WINDOW_SIZE
    +
    (SEQUENCE_LENGTH - 1) * STRIDE
)

MAX_SECTORS_PER_TIC = 3


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

planet_map = {

    row["tic_id"]: row

    for _, row in planet_df.iterrows()
}


# ============================================================
# CSV HEADER
# ============================================================

csv_header = [

    "tic_id",

    "sector",

    "chunk_id",

    "sequence_id",

    "label",

    "window_start_btjd",

    "window_end_btjd"
]

for seq_i in range(SEQUENCE_LENGTH):

    for flux_i in range(WINDOW_SIZE):

        csv_header.append(

            f"seq{seq_i}_flux_{flux_i}"
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

    median_flux = np.median(flux)

    if median_flux == 0:
        return None

    flux = flux / median_flux

    flux = (
        flux - np.mean(flux)
    ) / (
        np.std(flux) + 1e-8
    )

    return flux.astype(np.float32)


# ============================================================
# SPLIT CONTINUOUS CHUNKS
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

    order = np.argsort(time)

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

        if len(t_chunk) < MIN_CHUNK_POINTS:
            continue

        chunks.append(
            (t_chunk, f_chunk)
        )

    return chunks


# ============================================================
# TRANSIT CHECK
# ============================================================

def full_transit_inside_window(

    window_start,
    window_end,

    period,
    t0_btjd,
    duration_days
):

    n_start = int(
        np.floor(
            (window_start - t0_btjd)
            /
            period
        )
    ) - 1

    n_end = int(
        np.ceil(
            (window_end - t0_btjd)
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

        transit_start = (
            transit_center
            -
            duration_days / 2
        )

        transit_end = (
            transit_center
            +
            duration_days / 2
        )

        if (

            window_start
            <=
            transit_start

            and

            window_end
            >=
            transit_end
        ):

            return True

    return False


# ============================================================
# WRITE SAMPLE
# ============================================================

def write_sequence(

    tic_id,
    sector,
    chunk_id,
    sequence_id,
    label,

    sequence_flux,

    window_start,
    window_end
):

    global total_samples
    global positive_samples
    global negative_samples

    row = [

        tic_id,

        sector,

        chunk_id,

        sequence_id,

        label,

        window_start,

        window_end
    ]

    for seq_flux in sequence_flux:

        row.extend(
            seq_flux
        )

    csv_writer.writerow(row)

    total_samples += 1

    if label == 1:
        positive_samples += 1
    else:
        negative_samples += 1


# ============================================================
# BUILD WINDOW LIST
# ============================================================

def build_windows(
    t_chunk,
    f_chunk
):

    windows = []

    n_windows = (

        (
            len(t_chunk)
            -
            WINDOW_SIZE
        )
        //
        STRIDE
    ) + 1

    for w in range(n_windows):

        start_idx = (
            w * STRIDE
        )

        end_idx = (
            start_idx
            +
            WINDOW_SIZE
        )

        window_flux = (
            f_chunk[
                start_idx:end_idx
            ]
        )

        window_time = (
            t_chunk[
                start_idx:end_idx
            ]
        )

        if (
            len(window_flux)
            !=
            WINDOW_SIZE
        ):
            continue

        norm_flux = normalize_flux(
            window_flux
        )

        if norm_flux is None:
            continue

        windows.append({

            "window_id":
                w,

            "flux":
                norm_flux,

            "time":
                window_time
        })

    return windows


# ============================================================
# PROCESS CONFIRMED PLANETS
# ============================================================

print("\n===================================")
print("PROCESSING CONFIRMED PLANETS")
print("===================================")

usable_df = pd.read_csv(
    CONFIRMED_SECTORS_CSV
)

usable_df["tic_id"] = (
    usable_df["tic_id"]
    .astype(int)
)

grouped = usable_df.groupby(
    "tic_id"
)

for tic_id, tic_group in tqdm(grouped):

    if tic_id not in planet_map:
        continue

    planet_info = planet_map[tic_id]

    period = float(
        planet_info["period"]
    )

    t0_btjd = (
        float(planet_info["t0"])
        - 2457000
    )

    duration_days = (
        float(
            planet_info["duration"]
        )
        / 24.0
    )

    for _, row in tic_group.iterrows():

        sector = int(
            row["sector"]
        )

        fits_file = row[
            "fits_file"
        ]

        fits_path = os.path.join(

            TESS_ARCHIVE,

            f"TIC_{tic_id}",

            fits_file
        )

        if not os.path.exists(fits_path):
            continue

        try:

            with fits.open(fits_path) as hdul:

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

                windows = build_windows(
                    t_chunk,
                    f_chunk
                )

                if (
                    len(windows)
                    <
                    SEQUENCE_LENGTH
                ):
                    continue

                for seq_start in range(

                    len(windows)
                    -
                    SEQUENCE_LENGTH
                    +
                    1
                ):

                    seq_windows = windows[

                        seq_start:
                        seq_start
                        +
                        SEQUENCE_LENGTH
                    ]

                    center_window = (
                        seq_windows[
                            CENTER_INDEX
                        ]
                    )

                    center_time = (
                        center_window[
                            "time"
                        ]
                    )

                    label = int(

                        full_transit_inside_window(

                            center_time.min(),

                            center_time.max(),

                            period,
                            t0_btjd,
                            duration_days
                        )
                    )

                    if label != 1:
                        continue

                    sequence_flux = [

                        w["flux"]

                        for w in seq_windows
                    ]

                    write_sequence(

                        tic_id,
                        sector,
                        chunk_id,
                        seq_start,
                        label,

                        sequence_flux,

                        center_time.min(),
                        center_time.max()
                    )

        except Exception as e:

            print(
                "ERROR:",
                tic_id,
                sector,
                e
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

    for tic_id, tic_group in tqdm(grouped):

        tic_id = int(tic_id)

        tic_dir = os.path.join(

            TESS_ARCHIVE,

            f"TIC_{tic_id}"
        )

        if not os.path.exists(tic_dir):
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

            try:

                with fits.open(fits_path) as hdul:

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

                    windows = build_windows(
                        t_chunk,
                        f_chunk
                    )

                    if (
                        len(windows)
                        <
                        SEQUENCE_LENGTH
                    ):
                        continue

                    for seq_start in range(

                        len(windows)
                        -
                        SEQUENCE_LENGTH
                        +
                        1
                    ):

                        seq_windows = windows[

                            seq_start:
                            seq_start
                            +
                            SEQUENCE_LENGTH
                        ]

                        center_window = (
                            seq_windows[
                                CENTER_INDEX
                            ]
                        )

                        center_time = (
                            center_window[
                                "time"
                            ]
                        )

                        sequence_flux = [

                            w["flux"]

                            for w in seq_windows
                        ]

                        write_sequence(

                            tic_id,
                            sector,
                            chunk_id,
                            seq_start,
                            0,

                            sequence_flux,

                            center_time.min(),
                            center_time.max()
                        )

            except Exception:
                pass


# ============================================================
# CLOSE
# ============================================================

csv_file.close()

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