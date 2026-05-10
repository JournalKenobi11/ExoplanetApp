import numpy as np
import torch
import lightkurve as lk
from astroquery.mast import Observations
from astropy.io import fits
import requests
import tempfile
import os

from preprocess import (
    zscore_signal,
    generate_fft_features,
    bls_extract,
)

from model import TessPrecisionRecallNet


device = torch.device("cpu")


checkpoint = torch.load(
    "tess_precision_recall_model_hardfp.pth",
    map_location=device
)


model = TessPrecisionRecallNet()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

model.eval()


THRESHOLD = 0.50


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


def fetch_tic_flux(tic_id):

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

        os.remove(tmp_path)

def run_inference(tic_id):

    flux = fetch_tic_flux(tic_id)

    flux = resize_flux(flux)

    raw = zscore_signal(flux)

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