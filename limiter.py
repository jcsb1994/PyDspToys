import numpy as np
import matplotlib.pyplot as plt


def analyze_limited_sine(
    freq=1000,
    amplitude=1.0,
    ceiling=0.8,
    sample_rate=48000,
    duration=0.02,
    phase=0.0,
    limiter_type="hard",
    plot=True,
):
    """
    Generate a sine wave, apply a limiter/clipping stage,
    analyze harmonic distortion with FFT, and print diagnostics.

    Parameters
    ----------
    freq : float
        Fundamental sine frequency (Hz)

    amplitude : float
        Input sine amplitude

    ceiling : float
        Limiter ceiling amplitude

    sample_rate : int
        Sampling rate (Hz)

    duration : float
        Signal duration (seconds)

    phase : float
        Initial phase (radians)

    limiter_type : str
        "hard" or "soft"

    plot : bool
        Show waveform + FFT plots
    """

    # ------------------------------------------------------------
    # Time vector
    # ------------------------------------------------------------
    t = np.arange(0, duration, 1 / sample_rate)

    # ------------------------------------------------------------
    # Generate sine
    # ------------------------------------------------------------
    x = amplitude * np.sin(2 * np.pi * freq * t + phase)

    # ------------------------------------------------------------
    # Apply limiter
    # ------------------------------------------------------------
    if limiter_type == "hard":
        y = np.clip(x, -ceiling, ceiling)

    elif limiter_type == "soft":
        # Soft saturation using tanh
        drive = amplitude / ceiling
        y = ceiling * np.tanh(drive * x / ceiling)

    else:
        raise ValueError("limiter_type must be 'hard' or 'soft'")

    # ------------------------------------------------------------
    # Clipping analysis
    # ------------------------------------------------------------
    clipped_samples = np.sum(np.abs(x) > ceiling)
    clipped_percent = 100 * clipped_samples / len(x)

    peak_before = np.max(np.abs(x))
    peak_after = np.max(np.abs(y))

    rms_before = np.sqrt(np.mean(x**2))
    rms_after = np.sqrt(np.mean(y**2))

    crest_before = peak_before / rms_before
    crest_after = peak_after / rms_after

    # ------------------------------------------------------------
    # FFT analysis
    # ------------------------------------------------------------
    N = len(y)

    window = np.hanning(N)
    yw = y * window

    fft = np.fft.rfft(yw)
    fft_mag = np.abs(fft)

    freqs = np.fft.rfftfreq(N, 1 / sample_rate)

    # Normalize
    fft_mag /= np.max(fft_mag)

    # ------------------------------------------------------------
    # Harmonic extraction
    # ------------------------------------------------------------
    nyquist = sample_rate / 2
    max_harmonic = int(nyquist // freq)

    harmonics = []

    fundamental_mag = None

    for n in range(1, max_harmonic + 1):
        target_freq = n * freq

        idx = np.argmin(np.abs(freqs - target_freq))
        mag = fft_mag[idx]

        harmonics.append((n, target_freq, mag))

        if n == 1:
            fundamental_mag = mag

    # ------------------------------------------------------------
    # THD estimation
    # ------------------------------------------------------------
    harmonic_power = 0.0

    for n, f, mag in harmonics[1:]:
        harmonic_power += mag**2

    thd = np.sqrt(harmonic_power) / fundamental_mag
    thd_percent = thd * 100

    # ------------------------------------------------------------
    # Print diagnostics
    # ------------------------------------------------------------
    print("\n=== LIMITER ANALYSIS ===")
    print(f"Limiter type           : {limiter_type}")
    print(f"Fundamental frequency  : {freq:.2f} Hz")
    print(f"Input amplitude        : {amplitude:.4f}")
    print(f"Ceiling                : {ceiling:.4f}")

    print("\n--- CLIPPING ---")
    print(f"Clipped samples        : {clipped_samples}")
    print(f"Clipped percentage     : {clipped_percent:.2f}%")

    print("\n--- SIGNAL METRICS ---")
    print(f"Peak before            : {peak_before:.4f}")
    print(f"Peak after             : {peak_after:.4f}")
    print(f"RMS before             : {rms_before:.4f}")
    print(f"RMS after              : {rms_after:.4f}")
    print(f"Crest factor before    : {crest_before:.4f}")
    print(f"Crest factor after     : {crest_after:.4f}")

    print("\n--- DISTORTION ---")
    print(f"Estimated THD          : {thd_percent:.2f}%")

    print("\n--- HARMONICS ---")
    for n, f, mag in harmonics[:10]:
        db = 20 * np.log10(max(mag, 1e-12))
        print(
            f"H{n:02d} | {f:8.1f} Hz | "
            f"Normalized Mag = {mag:.5f} | {db:7.2f} dB"
        )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        # Time domain
        axs[0].plot(t * 1000, x, label="Original")
        axs[0].plot(t * 1000, y, label="Limited", alpha=0.8)

        axs[0].axhline(ceiling, linestyle="--")
        axs[0].axhline(-ceiling, linestyle="--")

        axs[0].set_title("Waveform")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend()
        axs[0].grid(True)

        # FFT
        axs[1].plot(freqs, 20 * np.log10(fft_mag + 1e-12))

        axs[1].set_xlim(0, min(20000, nyquist))
        axs[1].set_ylim(-140, 5)

        axs[1].set_title("FFT Spectrum")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Magnitude (dBFS)")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    return {
        "time": t,
        "original": x,
        "limited": y,
        "freqs": freqs,
        "fft_mag": fft_mag,
        "thd_percent": thd_percent,
        "clipped_percent": clipped_percent,
        "harmonics": harmonics,
    }


# Example
if __name__ == "__main__":
    analyze_limited_sine(
        freq=10000,
        amplitude=1.5,
        ceiling=1.5,
        limiter_type="hard",
    )