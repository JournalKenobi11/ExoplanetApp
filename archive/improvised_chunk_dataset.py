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
    "tess_window_dataset.csv"
)

# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

GAP_THRESHOLD_DAYS = 0.02

WINDOW_SIZE = 512

STRIDE = 128

MAX_SECTORS_PER_TIC = 3

MIN_CHUNK_POINTS = WINDOW_SIZE


# ============================================================
# LOAD PLANET METADATA
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
# CSV SETUP
# ============================================================

csv_header = [

    "tic_id",
    "sector",
    "chunk_id",
    "window_id",
    "label"
]

csv_header += [

    f"flux_{i}"

    for i in range(WINDOW_SIZE)
]

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

unique_tics = set()


# ============================================================
# WRITE FUNCTION
# ============================================================

def write_window(

    tic_id,
    sector,
    chunk_id,
    window_id,
    label,
    flux
):

    global total_samples
    global positive_samples
    global negative_samples

    row = [

        tic_id,
        sector,
        chunk_id,
        window_id,
        label
    ]

    row.extend(
        flux.astype(np.float32)
    )

    csv_writer.writerow(row)

    total_samples += 1

    unique_tics.add(tic_id)

    if label == 1:
        positive_samples += 1
    else:
        negative_samples += 1


# ============================================================
# GAP SPLITTING
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

    sector_candidates = []

    # --------------------------------------------------------
    # PROCESS SECTORS
    # --------------------------------------------------------

    for _, row in tic_group.iterrows():

        sector = int(row["sector"])

        fits_file = row["fits_file"]

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

                flux = data["PDCSAP_FLUX"]

            chunks = split_continuous_chunks(
                time,
                flux
            )

            positive_windows = 0

            chunk_cache = []

            for chunk_id, (
                t_chunk,
                f_chunk
            ) in enumerate(chunks):

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

                    window_time = (
                        t_chunk[
                            start_idx:end_idx
                        ]
                    )

                    window_flux = (
                        f_chunk[
                            start_idx:end_idx
                        ]
                    )

                    if (
                        len(window_flux)
                        !=
                        WINDOW_SIZE
                    ):
                        continue

                    is_positive = (
                        full_transit_inside_window(

                            window_time.min(),
                            window_time.max(),

                            period,
                            t0_btjd,
                            duration_days
                        )
                    )

                    if is_positive:
                        positive_windows += 1

                    chunk_cache.append({

                        "chunk_id":
                            chunk_id,

                        "window_id":
                            w,

                        "flux":
                            window_flux,

                        "label":
                            int(is_positive)
                    })

            sector_candidates.append({

                "sector":
                    sector,

                "positive_windows":
                    positive_windows,

                "windows":
                    chunk_cache
            })

        except Exception as e:

            print(
                "ERROR:",
                tic_id,
                sector,
                e
            )

    # --------------------------------------------------------
    # TOP 3 SECTORS
    # --------------------------------------------------------

    sector_candidates = sorted(

        sector_candidates,

        key=lambda x: x[
            "positive_windows"
        ],

        reverse=True
    )

    sector_candidates = (
        sector_candidates[
            :MAX_SECTORS_PER_TIC
        ]
    )

    # --------------------------------------------------------
    # STORE POSITIVE WINDOWS
    # --------------------------------------------------------

    for sector_info in sector_candidates:

        sector = sector_info["sector"]

        for win in sector_info["windows"]:

            if win["label"] != 1:
                continue

            flux = normalize_flux(
                win["flux"]
            )

            if flux is None:
                continue

            write_window(

                tic_id,
                sector,
                win["chunk_id"],
                win["window_id"],
                1,
                flux
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

        sector_candidates = []

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

                largest_chunk = max([

                    len(c[0])

                    for c in chunks

                ], default=0)

                sector_candidates.append({

                    "sector":
                        sector,

                    "largest_chunk":
                        largest_chunk,

                    "chunks":
                        chunks
                })

            except Exception:
                pass

        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        sector_candidates = sorted(

            sector_candidates,

            key=lambda x: x[
                "largest_chunk"
            ],

            reverse=True
        )

        sector_candidates = (
            sector_candidates[
                :MAX_SECTORS_PER_TIC
            ]
        )

        # ----------------------------------------------------
        # WINDOWS
        # ----------------------------------------------------

        for sector_info in sector_candidates:

            sector = sector_info[
                "sector"
            ]

            chunks = sector_info[
                "chunks"
            ]

            for chunk_id, (
                t_chunk,
                f_chunk
            ) in enumerate(chunks):

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

                    if (
                        len(window_flux)
                        !=
                        WINDOW_SIZE
                    ):
                        continue

                    flux = normalize_flux(
                        window_flux
                    )

                    if flux is None:
                        continue

                    write_window(

                        tic_id,
                        sector,
                        chunk_id,
                        w,
                        0,
                        flux
                    )


# ============================================================
# CLOSE CSV
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
    "\nUnique TIC IDs:",
    len(unique_tics)
)

print(
    "\nSaved:",
    OUTPUT_CSV
)