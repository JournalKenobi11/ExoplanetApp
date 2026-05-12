import io
import os
import sys
import sqlite3
import tempfile
import warnings

import numpy as np
import requests
import torch

from astropy.io import fits
from astroquery.mast import Observations
from astropy.utils.exceptions import AstropyWarning

from preprocess import (
    zscore_signal,
    generate_fft_features,
    bls_extract,
)

from model import TessPrecisionRecallNet


warnings.simplefilter(
    "ignore",
    AstropyWarning
)


DB_NAME = "tess_cache.db"

PREPROCESS_VERSION = "v1_zscore_3197"


device = torch.device("cpu")


THRESHOLD = 0.50


def resource_path(relative_path):

    try:
        base_path = sys._MEIPASS

    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(
        base_path,
        relative_path
    )


checkpoint = torch.load(
    resource_path(
        "tess_precision_recall_model_hardfp.pth"
    ),
    map_location=device
)


model = TessPrecisionRecallNet()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

model.eval()


def initialize_database():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS lightcurves (

            tic_id INTEGER PRIMARY KEY,

            flux BLOB NOT NULL,

            preprocessing TEXT NOT NULL
        )
        """
    )

    conn.commit()

    conn.close()


initialize_database()


def serialize_array(array):

    buffer = io.BytesIO()

    np.save(buffer, array)

    return buffer.getvalue()


def deserialize_array(blob):

    buffer = io.BytesIO(blob)

    buffer.seek(0)

    return np.load(buffer)


def load_cached_flux(tic_id):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT flux, preprocessing
        FROM lightcurves
        WHERE tic_id = ?
        """,
        (tic_id,)
    )

    row = cursor.fetchone()

    conn.close()

    if row is None:
        return None

    flux_blob, preprocessing = row

    if preprocessing != PREPROCESS_VERSION:
        return None

    flux = deserialize_array(flux_blob)

    return flux.astype(np.float32)


def save_flux_to_cache(tic_id, flux):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    flux_blob = serialize_array(flux)

    cursor.execute(
        """
        INSERT OR REPLACE INTO lightcurves
        (
            tic_id,
            flux,
            preprocessing
        )
        VALUES (?, ?, ?)
        """,
        (
            tic_id,
            flux_blob,
            PREPROCESS_VERSION
        )
    )

    conn.commit()

    conn.close()


def resize_flux(flux, target_len=3197):

    flux = np.asarray(
        flux,
        dtype=np.float32
    )

    if len(flux) == target_len:
        return flux

    x_old = np.linspace(
        0,
        1,
        len(flux)
    )

    x_new = np.linspace(
        0,
        1,
        target_len
    )

    resized = np.interp(
        x_new,
        x_old,
        flux
    )

    return resized.astype(np.float32)


def download_tic_flux(tic_id):

    obs = Observations.query_criteria(
        target_name=str(tic_id),
        obs_collection="TESS",
        dataproduct_type="timeseries",
    )

    if len(obs) == 0:

        raise ValueError(
            f"No observations for TIC {tic_id}"
        )

    products = Observations.get_product_list(obs)

    products = Observations.filter_products(
        products,
        productSubGroupDescription=["LC"],
        extension="fits",
    )

    if len(products) == 0:

        raise ValueError(
            f"No LC FITS products for TIC {tic_id}"
        )

    product = products[0]

    data_uri = str(product["dataURI"])

    url = "https://mast.stsci.edu/api/v0.1/Download/file"

    with tempfile.NamedTemporaryFile(
        suffix=".fits",
        delete=False
    ) as tmp:

        response = requests.get(
            url,
            params={"uri": data_uri},
            timeout=180
        )

        response.raise_for_status()

        tmp.write(response.content)

        tmp_path = tmp.name

    try:

        with fits.open(
            tmp_path,
            memmap=False
        ) as hdul:

            data = hdul[1].data

            if "PDCSAP_FLUX" in data.names:

                flux = data["PDCSAP_FLUX"]

            elif "SAP_FLUX" in data.names:

                flux = data["SAP_FLUX"]

            else:

                raise ValueError(
                    "No usable flux column"
                )

        flux = np.asarray(
            flux,
            dtype=np.float32
        )

        flux = flux[
            np.isfinite(flux)
        ]

        if len(flux) < 100:

            raise ValueError(
                "Flux too short"
            )

        return flux

    finally:

        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def fetch_tic_flux(tic_id):

    cached_flux = load_cached_flux(tic_id)

    if cached_flux is not None:

        print(
            f"TIC {tic_id}: loaded from cache"
        )

        return cached_flux

    print(
        f"TIC {tic_id}: downloading from MAST"
    )

    flux = download_tic_flux(tic_id)

    flux = resize_flux(flux)

    flux = zscore_signal(flux)

    save_flux_to_cache(
        tic_id,
        flux
    )

    print(
        f"TIC {tic_id}: cached successfully"
    )

    return flux


def run_inference(tic_id):

    flux = fetch_tic_flux(tic_id)

    raw = flux

    fft = generate_fft_features(raw)

    fold, stats = bls_extract(raw)

    raw_tensor = (
        torch.from_numpy(raw)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    fft_tensor = (
        torch.from_numpy(fft)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    fold_tensor = (
        torch.from_numpy(fold)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    stats_tensor = (
        torch.from_numpy(stats)
        .float()
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        logits = model(
            raw_tensor,
            fft_tensor,
            fold_tensor,
            stats_tensor
        )

        probability = float(
            torch.sigmoid(logits).item()
        )

    return {
        "tic_id": int(tic_id),
        "probability": probability,
        "candidate": bool(
            probability >= THRESHOLD
        ),
        "period": float(stats[0]),
        "bls_power": float(stats[1]),
        "duration": float(stats[2]),
        "depth": float(stats[3]),
        "snr": float(stats[4]),
    }