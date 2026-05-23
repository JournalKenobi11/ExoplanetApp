import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import binned_statistic

import lightkurve as lk


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = r"/path/to/tess_fits_archive"

GLOBAL_BINS = 2001
LOCAL_BINS = 201

LOCAL_PHASE_WIDTH = 0.1

OUTPUT_DIR = "dataset_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# INPUT CSVS
# ============================================================

datasets = [
    ("final_confirmed_planets.csv", 1, "confirmed"),
    ("final_toi_false_positives.csv", 0, "toi_fp"),
    ("final_eclipsing_binaries.csv", 0, "eb"),
    ("final_random_noisy_stars.csv", 0, "random"),
]


# ============================================================
# HELPERS
# ============================================================

def load_and_stitch_lightcurve(tic_id):

    tic_dir = os.path.join(DATA_ROOT, str(tic_id))

    if not os.path.exists(tic_dir):
        return None

    fits_files = [
        os.path.join(tic_dir, f)
        for f in os.listdir(tic_dir)
        if f.endswith(".fits")
    ]

    if len(fits_files) == 0:
        return None

    lcs = []

    for fp in fits_files:

        try:

            lc = lk.read(fp)

            if hasattr(lc, "PDCSAP_FLUX"):
                lc = lc.PDCSAP_FLUX

            lc = lc.remove_nans()

            lc = lc.normalize()

            lcs.append(lc)

        except Exception:
            continue

    if len(lcs) == 0:
        return None

    try:
        stitched = lk.LightCurveCollection(lcs).stitch()
        return stitched

    except Exception:
        return None


def median_bin(phase, flux, bins, phase_min, phase_max):

    edges = np.linspace(
        phase_min,
        phase_max,
        bins + 1
    )

    stat, _, _ = binned_statistic(
        phase,
        flux,
        statistic="median",
        bins=edges
    )

    centers = (edges[:-1] + edges[1:]) / 2

    valid = ~np.isnan(stat)

    if valid.sum() < 10:
        return None

    interpolated = np.interp(
        centers,
        centers[valid],
        stat[valid]
    )

    return interpolated.astype(np.float32)


def create_global_view(folded):

    phase = folded.phase.value
    flux = folded.flux.value

    return median_bin(
        phase,
        flux,
        bins=GLOBAL_BINS,
        phase_min=-0.5,
        phase_max=0.5
    )


def create_local_view(folded):

    phase = folded.phase.value
    flux = folded.flux.value

    mask = np.abs(phase) < LOCAL_PHASE_WIDTH

    if mask.sum() < 20:
        return None

    return median_bin(
        phase[mask],
        flux[mask],
        bins=LOCAL_BINS,
        phase_min=-LOCAL_PHASE_WIDTH,
        phase_max=LOCAL_PHASE_WIDTH
    )


# ============================================================
# MAIN PROCESSING
# ============================================================

global_views = []
local_views = []

labels = []
tic_ids = []
class_names = []


for csv_path, label, class_name in datasets:

    print(f"\nProcessing: {class_name}")

    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df)):

        try:

            tic_id = int(row["tic_id"])

            lc = load_and_stitch_lightcurve(tic_id)

            if lc is None:
                continue

            # ------------------------------------------------
            # PERIOD / EPOCH
            # ------------------------------------------------

            if class_name == "random":

                # Random fold for nonplanet noisy stars
                period = np.random.uniform(1, 15)

                epoch = lc.time.value.min()

            else:

                period = row["period"]
                epoch = row["epoch"]

            if pd.isna(period) or pd.isna(epoch):
                continue

            # ------------------------------------------------
            # FOLD
            # ------------------------------------------------

            folded = lc.fold(
                period=period,
                epoch_time=epoch
            )

            # ------------------------------------------------
            # GLOBAL / LOCAL VIEWS
            # ------------------------------------------------

            global_view = create_global_view(folded)

            if global_view is None:
                continue

            local_view = create_local_view(folded)

            if local_view is None:
                continue

            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            global_views.append(global_view)
            local_views.append(local_view)

            labels.append(label)
            tic_ids.append(tic_id)
            class_names.append(class_name)

        except Exception:
            continue


# ============================================================
# EXPORT
# ============================================================

global_views = np.array(global_views)
local_views = np.array(local_views)

labels = np.array(labels)
tic_ids = np.array(tic_ids)
class_names = np.array(class_names)

np.save(
    os.path.join(OUTPUT_DIR, "global_views.npy"),
    global_views
)

np.save(
    os.path.join(OUTPUT_DIR, "local_views.npy"),
    local_views
)

np.save(
    os.path.join(OUTPUT_DIR, "labels.npy"),
    labels
)

np.save(
    os.path.join(OUTPUT_DIR, "tic_ids.npy"),
    tic_ids
)

np.save(
    os.path.join(OUTPUT_DIR, "class_names.npy"),
    class_names)

print("\n===================================")
print("DONE")
print("===================================")

print("Samples:", len(labels))
print("Global shape:", global_views.shape)
print("Local shape:", local_views.shape)
print("Labels shape:", labels.shape)