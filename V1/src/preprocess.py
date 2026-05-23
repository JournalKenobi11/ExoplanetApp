import numpy as np
from astropy.timeseries import BoxLeastSquares


def zscore_signal(signal):

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    return (
        (signal - signal.mean())
        / (signal.std() + 1e-8)
    ).astype(np.float32)


def generate_fft_features(raw):

    fft = np.abs(
        np.fft.rfft(raw)
    )[:512]

    fft = (
        (fft - fft.mean())
        / (fft.std() + 1e-8)
    )

    return fft.astype(np.float32)


def bls_extract(signal, bins=512):

    time = np.arange(
        len(signal),
        dtype=np.float32
    )

    bls = BoxLeastSquares(
        time,
        signal
    )

    result = bls.power(
        np.linspace(20, 500, 300),
        np.linspace(0.5, 5, 10)
    )

    idx = int(
        np.argmax(result.power)
    )

    period = float(result.period[idx])

    power = float(result.power[idx])

    duration = float(result.duration[idx])

    phase = (time % period) / period

    order = np.argsort(phase)

    phase_sorted = phase[order]

    signal_sorted = signal[order]

    edges = np.linspace(
        0,
        1,
        bins + 1
    )

    bin_ids = np.searchsorted(
        edges,
        phase_sorted,
        side="right"
    ) - 1

    bin_ids = np.clip(
        bin_ids,
        0,
        bins - 1
    )

    sums = np.bincount(
        bin_ids,
        weights=signal_sorted,
        minlength=bins
    )

    counts = np.bincount(
        bin_ids,
        minlength=bins
    )

    folded = np.zeros(
        bins,
        dtype=np.float32
    )

    filled = counts > 0

    folded[filled] = (
        sums[filled]
        / counts[filled]
    )

    folded = (
        (folded - folded.mean())
        / (folded.std() + 1e-8)
    ).astype(np.float32)

    depth = float(
        np.min(folded)
    )

    snr = float(
        abs(depth)
        / (np.std(folded) + 1e-8)
    )

    stats = np.array(
        [
            period,
            power,
            duration,
            depth,
            snr
        ],
        dtype=np.float32
    )

    return folded, stats