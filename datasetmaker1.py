import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

import lightkurve as lk

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = r"/path/to/tess_fits_archive"

OUTPUT_DIR = "processed_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GLOBAL_BINS = 2001
LOCAL_BINS = 201

LOCAL_VIEW_WIDTH_IN_TRANSIT_DURATIONS = 4

FLATTEN_WINDOW = 401
OUTLIER_SIGMA = 5


# ============================================================
# INPUT CSVs
# ============================================================

CONFIRMED_CSV = "final_confirmed_planets.csv"
TOI_FP_CSV = "final_toi_false_positives.csv"
EB_CSV = "final_eclipsing_binaries.csv"
RANDOM_CSV = "final_random_noisy_stars.csv"


# ============================================================
# REQUIRED CSV COLUMNS
# ============================================================

"""
Expected columns:

tic_id
period
epoch

For random noisy stars:
period/epoch may be absent.
"""


# ============================================================
# HELPERS
# ============================================================

def periodic_interpolate(x, y, target_x):
    """
    Periodic interpolation to avoid edge artifacts.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_ext = np.concatenate([
        x - 1.0,
        x,
        x + 1.0
    ])

    y_ext = np.concatenate([y, y, y])

    interp = interp1d(
        x_ext,
        y_ext,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    return interp(target_x)


def robust_median_bin(phase, flux, bins):
    """
    Median binning in phase space.
    """

    edges = np.linspace(-0.5, 0.5, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    binned = np.full(bins, np.nan)

    inds = np.digitize(phase, edges) - 1

    for i in range(bins):
        mask = inds == i

        if np.any(mask):
            binned[i] = np.nanmedian(flux[mask])

    valid = ~np.isnan(binned)

    if valid.sum() < 10:
        return None

    interp = interp1d(
        centers[valid],
        binned[valid],
        bounds_error=False,
        fill_value="extrapolate"
    )

    return interp(centers)


def load_all_sectors(tic_id):
    """
    Load and stitch all FITS sectors for one TIC.
    """

    tic_dir = os.path.join(DATA_ROOT, str(tic_id))

    if not os.path.exists(tic_dir):
        return None

    files = [
        os.path.join(tic_dir, f)
        for f in os.listdir(tic_dir)
        if f.endswith(".fits")
    ]

    if len(files) == 0:
        return None

    lcs = []

    for f in files:
        try:
            lc = lk.read(f)

            if hasattr(lc, "PDCSAP_FLUX"):
                lc = lc.PDCSAP_FLUX

            lc = lc.remove_nans()
            lc = lc.normalize()

            lc = lc.flatten(window_length=FLATTEN_WINDOW)

            lc = lc.remove_outliers(sigma=OUTLIER_SIGMA)

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


def create_global_view(folded_lc):
    """
    Create wide folded morphology view.
    """

    phase = folded_lc.phase.value
    flux = folded_lc.flux.value

    result = robust_median_bin(
        phase,
        flux,
        GLOBAL_BINS
    )

    return result


def create_local_view(folded_lc, duration_days=None):
    """
    Create transit-centered local morphology view.
    """

    phase = folded_lc.phase.value
    flux = folded_lc.flux.value

    if duration_days is None:
        width = 0.05

    else:
        period = folded_lc.period.value

        transit_phase_width = duration_days / period

        width = (
            transit_phase_width
            * LOCAL_VIEW_WIDTH_IN_TRANSIT_DURATIONS
        )

    mask = np.abs(phase) < width

    if mask.sum() < 20:
        return None

    phase_local = phase[mask]
    flux_local = flux[mask]

    edges = np.linspace(
        -width,
        width,
        LOCAL_BINS + 1
    )

    centers = (edges[:-1] + edges[1:]) / 2

    binned = np.full(LOCAL_BINS, np.nan)

    inds = np.digitize(phase_local, edges) - 1

    for i in range(LOCAL_BINS):
        m = inds == i

        if np.any(m):
            binned[i] = np.nanmedian(flux_local[m])

    valid = ~np.isnan(binned)

    if valid.sum() < 10:
        return None

    interp = interp1d(
        centers[valid],
        binned[valid],
        bounds_error=False,
        fill_value="extrapolate"
    )

    return interp(centers)


def process_row(row, label, class_name):
    """
    Process one TIC object.
    """

    tic_id = row["tic_id"]

    period = row.get("period", np.nan)
    epoch = row.get("epoch", np.nan)

    lc = load_all_sectors(tic_id)

    if lc is None:
        return None

    # --------------------------------------------------------
    # RANDOM NEGATIVE HANDLING
    # --------------------------------------------------------

    if class_name == "random":

        if np.isnan(period):
            period = np.random.uniform(1.0, 15.0)

        if np.isnan(epoch):
            epoch = lc.time.value.min()

    # --------------------------------------------------------
    # REQUIRE PERIOD/EPOCH
    # --------------------------------------------------------

    if np.isnan(period) or np.isnan(epoch):
        return None

    try:

        folded = lc.fold(
            period=period,
            epoch_time=epoch
        )

    except Exception:
        return None

    global_view = create_global_view(folded)

    if global_view is None:
        return None

    duration = row.get("duration", None)

    local_view = create_local_view(
        folded,
        duration_days=duration
    )

    if local_view is None:
        return None

    return {
        "tic_id": tic_id,
        "label": label,
        "class": class_name,
        "global_view": global_view.astype(np.float32),
        "local_view": local_view.astype(np.float32)
    }


# ============================================================
# LOAD CSVS
# ============================================================

datasets = [
    (pd.read_csv(CONFIRMED_CSV), 1, "confirmed"),
    (pd.read_csv(TOI_FP_CSV), 0, "toi_fp"),
    (pd.read_csv(EB_CSV), 0, "eclipsing_binary"),
    (pd.read_csv(RANDOM_CSV), 0, "random")
]


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

results = []

for df, label, class_name in datasets:

    print(f"\nProcessing: {class_name}")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        try:

            result = process_row(
                row,
                label,
                class_name
            )

            if result is not None:
                results.append(result)

        except Exception:
            continue


# ============================================================
# EXPORT
# ============================================================

global_views = np.stack([
    r["global_view"] for r in results
])

local_views = np.stack([
    r["local_view"] for r in results
])

labels = np.array([
    r["label"] for r in results
], dtype=np.int64)

tic_ids = np.array([
    r["tic_id"] for r in results
])

classes = np.array([
    r["class"] for r in results
])

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
    os.path.join(OUTPUT_DIR, "classes.npy"),
    classes
)

print("\n===================================")
print("FINAL DATASET")
print("===================================")

print("Samples:", len(results))
print("Global shape:", global_views.shape)
print("Local shape:", local_views.shape)
print("Labels shape:", labels.shape)