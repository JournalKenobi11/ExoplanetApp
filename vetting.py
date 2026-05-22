import os
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

from astropy.timeseries import BoxLeastSquares

from src.inference import run_inference


OUTPUT_DIR = "vetting_reports"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def download_lightcurve(tic_id):

    search = lk.search_lightcurve(
        f"TIC {tic_id}",
        mission="TESS"
    )

    lc = (
        search
        .download_all()
        .stitch()
    )

    lc = lc.remove_nans()

    lc = lc.normalize()

    return lc


def compute_bls(time, flux):

    periods = np.linspace(
        0.5,
        30,
        5000
    )

    durations = np.linspace(
        0.05,
        0.3,
        10
    )

    bls = BoxLeastSquares(
        time,
        flux
    )

    result = bls.power(
        periods,
        durations
    )

    idx = np.argmax(
        result.power
    )

    best_period = result.period[idx]

    best_duration = result.duration[idx]

    transit_time = result.transit_time[idx]

    best_power = result.power[idx]

    return (
        result,
        best_period,
        best_duration,
        transit_time,
        best_power
    )


def fold_data(
    time,
    flux,
    period,
    transit_time
):

    phase = (
        (time - transit_time + 0.5 * period)
        % period
    ) / period - 0.5

    order = np.argsort(phase)

    return (
        phase[order],
        flux[order]
    )


def odd_even_masks(
    time,
    period,
    transit_time
):

    transit_numbers = np.floor(
        (time - transit_time) / period
    ).astype(int)

    odd_mask = (
        transit_numbers % 2 == 1
    )

    even_mask = (
        transit_numbers % 2 == 0
    )

    return (
        odd_mask,
        even_mask
    )


def generate_vetting_report(tic_id):

    print(
        f"Generating vetting report for TIC {tic_id}"
    )

    inference_result = run_inference(
        tic_id
    )

    probability = inference_result[
        "probability"
    ]

    lc = download_lightcurve(
        tic_id
    )

    time = np.asarray(
        lc.time.value,
        dtype=np.float32
    )

    flux = np.asarray(
        lc.flux.value,
        dtype=np.float32
    )

    mask = (
        np.isfinite(time)
        &
        np.isfinite(flux)
    )

    time = time[mask]

    flux = flux[mask]

    (
        bls_result,
        best_period,
        best_duration,
        transit_time,
        best_power
    ) = compute_bls(
        time,
        flux
    )

    folded_phase, folded_flux = fold_data(
        time,
        flux,
        best_period,
        transit_time
    )

    odd_mask, even_mask = odd_even_masks(
        time,
        best_period,
        transit_time
    )

    odd_phase, odd_flux = fold_data(
        time[odd_mask],
        flux[odd_mask],
        best_period,
        transit_time
    )

    even_phase, even_flux = fold_data(
        time[even_mask],
        flux[even_mask],
        best_period,
        transit_time
    )

    fig = plt.figure(
        figsize=(16, 12)
    )

    grid = fig.add_gridspec(
        3,
        2
    )

    ax1 = fig.add_subplot(
        grid[0, :]
    )

    ax2 = fig.add_subplot(
        grid[1, 0]
    )

    ax3 = fig.add_subplot(
        grid[1, 1]
    )

    ax4 = fig.add_subplot(
        grid[2, 0]
    )

    ax5 = fig.add_subplot(
        grid[2, 1]
    )

    # FULL LIGHTCURVE

    ax1.plot(
        time,
        flux,
        ".",
        markersize=1
    )

    ax1.set_title(
        f"TIC {tic_id} - Full Lightcurve"
    )

    ax1.set_xlabel(
        "Time"
    )

    ax1.set_ylabel(
        "Normalized Flux"
    )

    # BLS PERIODOGRAM

    ax2.plot(
        bls_result.period,
        bls_result.power
    )

    ax2.axvline(
        best_period,
        linestyle="--"
    )

    ax2.set_title(
        "BLS Periodogram"
    )

    ax2.set_xlabel(
        "Period (days)"
    )

    ax2.set_ylabel(
        "BLS Power"
    )

    # FOLDED LIGHTCURVE

    ax3.plot(
        folded_phase,
        folded_flux,
        ".",
        markersize=2
    )

    ax3.set_title(
        "Phase Folded Transit"
    )

    ax3.set_xlabel(
        "Phase"
    )

    ax3.set_ylabel(
        "Normalized Flux"
    )

    ax3.set_xlim(
        -0.1,
        0.1
    )

    # ODD TRANSITS

    ax4.plot(
        odd_phase,
        odd_flux,
        ".",
        markersize=2
    )

    ax4.set_title(
        "Odd Transits"
    )

    ax4.set_xlim(
        -0.1,
        0.1
    )

    ax4.set_xlabel(
        "Phase"
    )

    ax4.set_ylabel(
        "Flux"
    )

    # EVEN TRANSITS

    ax5.plot(
        even_phase,
        even_flux,
        ".",
        markersize=2
    )

    ax5.set_title(
        "Even Transits"
    )

    ax5.set_xlim(
        -0.1,
        0.1
    )

    ax5.set_xlabel(
        "Phase"
    )

    ax5.set_ylabel(
        "Flux"
    )

    fig.suptitle(
        (
            f"TransitAI Vetting Report\n"
            f"TIC {tic_id}\n"
            f"Probability = {probability:.4f} | "
            f"Period = {best_period:.4f} d | "
            f"BLS Power = {best_power:.2f}"
        ),
        fontsize=16
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"TIC_{tic_id}_vetting.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )

    return output_path


if __name__ == "__main__":

    tic_ids = [
        231909308,
        258918933,
        31181554,
        235980310,
        401089010,
        327756689,
        72874033,
    ]

    for tic in tic_ids:

        try:

            generate_vetting_report(
                tic
            )

        except Exception as exc:

            print(
                f"FAILED TIC {tic}: {exc}"
            )