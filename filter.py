import numpy as np
from scipy.signal import butter, sosfilt


def lpf(audio, sample_rate, cutoff_freq, order):

    nyquist = sample_rate * 0.5

    if cutoff_freq >= nyquist:
        raise ValueError("cutoff_freq must be below Nyquist frequency")

    # Create stable Butterworth LPF using second-order sections
    sos = butter(
        N=order,
        Wn=cutoff_freq,
        btype="low",
        fs=sample_rate,
        output="sos"
    )

    audio = np.asarray(audio)

    # Mono
    if audio.ndim == 1:
        return sosfilt(sos, audio)

    # Stereo / multichannel
    elif audio.ndim == 2:
        filtered = np.empty_like(audio)

        for ch in range(audio.shape[1]):
            filtered[:, ch] = sosfilt(sos, audio[:, ch])

        return filtered

    else:
        raise ValueError("audio must be 1D (mono) or 2D (multichannel)")