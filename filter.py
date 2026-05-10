import numpy as np
from scipy.signal import butter, sosfilt

from wave import Soundwave
import fft

def comb_filter_sine(
    sine: Soundwave,
    delay_ms: float,
    sample_rate: int,
    duration: float = 1.0,
    mix: float = 1.0,
):
    """
    Creates a comb-filtered sine wave by mixing the original
    with a delayed copy.
    """
    # Create the sine wave using our custom sound wave class
    original = sine.generate(sample_rate, duration)

    # Delay conversion
    delay_samples = int(delay_ms * sample_rate / 1000)

    # Create delayed copy
    delayed = np.zeros_like(original)

    if delay_samples < len(original):
        delayed[delay_samples:] = original[:-delay_samples]

    # Mix original + delayed
    output = original + (mix * delayed)

    # Comb filter spacing
    delay_sec = delay_ms / 1000
    spacing_hz = 1 / delay_sec

    print(f"\nComb Filter Info")
    print(f"----------------")
    print(f"Delay: {delay_ms:.3f} ms")
    print(f"Delay samples: {delay_samples}")
    print(f"Comb spacing: {spacing_hz:.2f} Hz")

    print("\nBoost frequencies:")
    for n in range(6):
        boost_freq = n * spacing_hz
        print(f"  {boost_freq:.2f} Hz")

    print("\nNotch frequencies:")
    for n in range(6):
        notch_freq = (n + 0.5) * spacing_hz
        print(f"  {notch_freq:.2f} Hz")

    return output

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


if __name__ == "__main__":
    sample_rate = 48000
    sine = Soundwave(freq=10000, amplitude=1.2)

    # Delay of 1 ms means the delta f will be 1kHz, notches will be every x.5kHz
    combed = comb_filter_sine(sine=sine, delay_ms=1, sample_rate=sample_rate, duration=0.02)
    bin_freqs, bin_magnitudes = fft.fft(combed, sample_rate)
    fft.fft_plt(bin_freqs, bin_magnitudes)