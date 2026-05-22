import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from astropy.io import fits
from scipy.interpolate import interp1d


# ============================================================
# CONFIG
# ============================================================

TESS_ARCHIVE = r"C:\Users\aasha\OneDrive\Desktop\correction\tess_archive"

DATASETS = [
    ("confirmed_planet_usable_sectors.csv", 1),
    ("final_toi_false_positives.csv", 0),
    ("final_eclipsing_binaries.csv", 0),
    ("final_random_noisy_stars.csv", 0)
]

OUTPUT_CSV = "tess_dataset_8192.csv"

TARGET_LENGTH = 8192

MAX_SECTORS_PER_TIC = 3

MIN_POINTS = 1000

MIN_USABLE_FRACTION = 0.85

MAX_GAP_FRACTION = 0.10


# ============================================================
# OUTLIER REMOVAL
# ============================================================

def remove_outliers(flux, sigma=5):

    median = np.median(flux)

    std = np.std(flux)

    if std == 0:
        return np.ones(len(flux), dtype=bool)

    z = np.abs(flux - median) / std

    return z < sigma


# ============================================================
# GAP CHECK
# ============================================================

def largest_gap_fraction(time):

    time = time[np.isfinite(time)]

    if len(time) < 2:
        return 1.0

    time = np.sort(time)

    gaps = np.diff(time)

    largest_gap = np.max(gaps)

    sector_duration = (
        time.max() - time.min()
    )

    if sector_duration <= 0:
        return 1.0

    return largest_gap / sector_duration


# ============================================================
# REPRESENTATION
# ============================================================

def make_representation(time, flux):

    # --------------------------------------------------------
    # REMOVE NaNs
    # --------------------------------------------------------

    valid = (
        np.isfinite(time) &
        np.isfinite(flux)
    )

    time = time[valid]
    flux = flux[valid]

    # --------------------------------------------------------
    # MIN POINTS
    # --------------------------------------------------------

    if len(time) < MIN_POINTS:
        return None

    # --------------------------------------------------------
    # GAP CHECK
    # --------------------------------------------------------

    gap_fraction = largest_gap_fraction(time)

    if gap_fraction > MAX_GAP_FRACTION:
        return None

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    order = np.argsort(time)

    time = time[order]
    flux = flux[order]

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    median_flux = np.median(flux)

    if median_flux == 0:
        return None

    flux = flux / median_flux

    # --------------------------------------------------------
    # REMOVE OUTLIERS
    # --------------------------------------------------------

    keep = remove_outliers(flux, sigma=5)

    time = time[keep]
    flux = flux[keep]

    # --------------------------------------------------------
    # POST FILTER CHECK
    # --------------------------------------------------------

    if len(time) < MIN_POINTS:
        return None

    # --------------------------------------------------------
    # RESAMPLE
    # --------------------------------------------------------

    new_time = np.linspace(
        time.min(),
        time.max(),
        TARGET_LENGTH
    )

    try:

        interp = interp1d(
            time,
            flux,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate"
        )

        new_flux = interp(new_time)

    except Exception:

        return None

    # --------------------------------------------------------
    # FINAL FINITE CHECK
    # --------------------------------------------------------

    if not np.all(np.isfinite(new_flux)):
        return None

    # --------------------------------------------------------
    # STANDARDIZE
    # --------------------------------------------------------

    new_flux = (
        new_flux - np.mean(new_flux)
    ) / (
        np.std(new_flux) + 1e-8
    )

    return new_flux.astype(np.float32)


# ============================================================
# MAIN
# ============================================================

all_rows = []

for csv_path, label in DATASETS:

    print("\n===================================")
    print("PROCESSING:", csv_path)
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

            sector_rows = []

            # ------------------------------------------------
            # PROCESS EACH SECTOR
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

                        total_points = len(time)

                        valid = (
                            np.isfinite(time) &
                            np.isfinite(flux)
                        )

                        usable_points = int(valid.sum())

                        if total_points == 0:
                            continue

                        usable_fraction = (
                            usable_points /
                            total_points
                        )

                        # ------------------------------------
                        # QUALITY FILTER
                        # ------------------------------------

                        if (
                            usable_fraction <
                            MIN_USABLE_FRACTION
                        ):
                            continue

                        # ------------------------------------
                        # GAP FILTER
                        # ------------------------------------

                        gap_fraction = (
                            largest_gap_fraction(
                                time[valid]
                            )
                        )

                        if (
                            gap_fraction >
                            MAX_GAP_FRACTION
                        ):
                            continue

                        sector_rows.append({

                            "tic_id":
                                tic_id,

                            "sector":
                                int(hdr0["SECTOR"]),

                            "label":
                                label,

                            "usable_fraction":
                                usable_fraction,

                            "time":
                                time,

                            "flux":
                                flux
                        })

                except Exception as e:

                    print(
                        "FITS ERROR:",
                        fits_name,
                        e
                    )

            # ------------------------------------------------
            # REMOVE DUPLICATE SECTORS
            # ------------------------------------------------

            unique_rows = {}

            for row in sector_rows:

                key = (
                    row["tic_id"],
                    row["sector"]
                )

                if key not in unique_rows:

                    unique_rows[key] = row

            sector_rows = list(
                unique_rows.values()
            )

            # ------------------------------------------------
            # TOP 3 SECTORS
            # ------------------------------------------------

            if len(sector_rows) == 0:
                continue

            sector_rows = sorted(
                sector_rows,
                key=lambda x: x["usable_fraction"],
                reverse=True
            )

            sector_rows = sector_rows[
                :MAX_SECTORS_PER_TIC
            ]

            # ------------------------------------------------
            # REPRESENTATIONS
            # ------------------------------------------------

            for srow in sector_rows:

                vector = make_representation(
                    srow["time"],
                    srow["flux"]
                )

                if vector is None:
                    continue

                row = {

                    "tic_id":
                        srow["tic_id"],

                    "sector":
                        srow["sector"],

                    "label":
                        srow["label"]
                }

                for i, val in enumerate(vector):

                    row[f"flux_{i}"] = float(val)

                all_rows.append(row)

        except Exception as e:

            print(
                "ROW ERROR:",
                tic_id,
                e
            )


# ============================================================
# EXPORT
# ============================================================

dataset_df = pd.DataFrame(all_rows)

dataset_df = dataset_df.drop_duplicates(
    subset=["tic_id", "sector"]
)

dataset_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n===================================")
print("DONE")
print("===================================")

print("Total samples:", len(dataset_df))

print("\nLabel distribution:")
print(
    dataset_df["label"]
    .value_counts()
)

print("\nUnique TIC IDs:")
print(
    dataset_df["tic_id"]
    .nunique()
)

print("\nSaved:")
print(OUTPUT_CSV)