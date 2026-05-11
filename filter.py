import numpy as np
from scipy.signal import butter, sosfilt

from wave import Soundwave
import fft
import analysis as anls

def align_signals(audio1, audio2, sample_rate):
    lag, _ = anls.estimate_delay(audio1, audio2, sample_rate)

    if lag > 0:
        audio2_aligned = audio2[lag:]
        audio_aligned = audio1[:len(audio2_aligned)]
    elif lag < 0:
        audio_aligned = audio1[-lag:]
        audio2_aligned = audio2[:len(audio_aligned)]
    else:
        audio_aligned = audio1
        audio2_aligned = audio2

    return audio_aligned, audio2_aligned, lag

def efficient_dcblock_filter(audio):
    """
    This filter simply subtracts the average from the signal.
    For real time signals, this method would use a moving average instead,
    but it is an efficient way to remove DC components without phase shifting and with a very low cutoff.
    This may be favored over a SW HPF if we really just want to filter out DC in a resource constrained system.
    """
    audio = audio - np.mean(audio)
    return audio

def one_band_pass_filter(audio, sample_rate, cutoff_freq, order, high=False):
    """
    Either a LPF or HPF
    """

    nyquist = sample_rate * 0.5

    if cutoff_freq >= nyquist:
        raise ValueError("cutoff_freq must be below Nyquist frequency")

    ftype = "low"
    if high:
        ftype = "high"

    # Create stable Butterworth LPF using second-order sections
    sos = butter(
        N=order,
        Wn=cutoff_freq,
        btype=ftype,
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
