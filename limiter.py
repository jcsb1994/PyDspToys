import numpy as np
import matplotlib.pyplot as plt


def limiter_simulator(freq, amplitude, sample_rate, duration, ceiling, plot=False):
    """
    Simulate a mono audio limiter on a sine wave.

    Args:
        freq (float): Frequency of the sine wave in Hz.
        amplitude (float): Amplitude of the sine wave (0.0 to 1.0).
        sample_rate (int): Sample rate in Hz.
        duration (float): Duration of the simulation in seconds.
        ceiling (float): Limiter ceiling (max output amplitude, 0.0 to 1.0).

    Returns:
        np.ndarray: Limited mono audio stream.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * freq * t)
    limited = np.clip(sine_wave, -ceiling, ceiling)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(t, sine_wave, label="Original Sine Wave")
        #Add a horizontal line to indicate the ceiling
        plt.axhline(ceiling, color='r', linestyle='--', label="Ceiling")
        plt.plot(t, limited, label="Limited Output", linestyle='--')
        plt.title("Limiter Simulation")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.show()

    return limited



def sine_distortion_analysis(freq, stream, sample_rate, plot=False):
    """
    Analyze a mono stream for sine distortion.

    Args:
        freq (float): Expected sine frequency in Hz.
        stream (np.ndarray): Mono audio stream.
        sample_rate (int): Sample rate in Hz.
    """
    # 1. FFT
    # ===============================

    # N is the standard name for the length of the input signal in FFT analysis
    # (Using N and sample rate you can infer the time duration of the input signal)
    N = len(stream)

    # The bin size is dependent on 2 things: the sample rate and the length of the input signal (N).
    # The bin size represents the frequency resolution of the FFT and is sample_rate / N
    # This is because in order to see a frequency component in the FFT output, you need at least one full cycle of that frequency
    # So for example to see a 0.5Hz (bin size 0.5) N must be double the sample rate
    bin_size = sample_rate // N

    # This returns a list of size N with bins in the format i+j.
    # Because of the Nyquist theorem, we can only detect up to sample_rate/2
    # Only half of the bins are used for 0Hz to Nyquist frequency (other half is the same with negative freqs)
    fft = np.fft.fft(stream)

    # The frequency bins from np.fft.fftfreq(N, 1/sample_rate) go from 0 up to just below the Nyquist frequency, then from
    # negative Nyquist back up to just below 0. So, the first half (bin_freqs[:N//2]) corresponds to 0 Hz up to just below the Nyquist frequency
    pos_freqs_len = N // 2
    bin_magnitudes = np.abs(fft)
    bin_freqs = np.fft.fftfreq(N, 1/sample_rate)
    # We are working with real signals so take the positive frequencies only (the first half of the FFT output)
    bin_magnitudes = bin_magnitudes[:pos_freqs_len]
    bin_freqs = bin_freqs[:pos_freqs_len]

    print(f"Bin Size: {bin_size} Hz")
    print(f"Number of Bins: {len(bin_freqs)}")
    print(f"Frequency of first few bins: {bin_freqs[:10]} Hz")
    print(f"Magnitudes of first few bins: {bin_magnitudes[:10]}")

    # 2. Analysis
    # ===============================

    # Plot the frequency spectrum for visualization
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(bin_freqs, 20 * np.log10(bin_magnitudes + 1e-12))
        plt.title("Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid()
        plt.show()

    # Find the peak frequency (it should be close to the expected frequency if it's a pure sine wave)
    peak_idx = np.argmax(bin_magnitudes)
    peak_freq = bin_freqs[peak_idx]
    peak_magnitude = bin_magnitudes[peak_idx]

    # Check for significant harmonics
    lowest_harmonic_checked = 2
    highest_harmonic_checked = 5
    harmonics = [freq * n for n in range(lowest_harmonic_checked, highest_harmonic_checked+1) if freq * n < sample_rate / 2]
    harmonic_mags = [bin_magnitudes[np.argmin(np.abs(bin_freqs - h))] for h in harmonics]
    harmonic_ratio = sum(harmonic_mags) / (peak_magnitude + 1e-12)

    # Flag if any significant harmonics are odd harmonics (which are more indicative of distortion / aliasing)
    print("Harmonics:", harmonics)
    for h in harmonics:
        # For each harmonic, check if there is a significant peak within accepted margin of any odd harmonic multiple
        for i in range(lowest_harmonic_checked, highest_harmonic_checked+1):
            curr_odd_multiple = freq * (i if i % 2 == 1 else i + 1)  # Check odd multiples (1st, 3rd, 5th, etc.)
            percentage_accepted = 0.1
            above_10_percent_of_fundamental_multiple = h > (curr_odd_multiple - (freq * percentage_accepted))
            below_10_percent_of_fundamental_multiple = h < (curr_odd_multiple + (freq * percentage_accepted))

            if below_10_percent_of_fundamental_multiple and above_10_percent_of_fundamental_multiple:
                print(f"Warning: Significant odd harmonic detected at {h:.2f} Hz")
                break


    print(f"Peak Frequency: {peak_freq:.2f} Hz")
    print(f"Harmonic Ratio: {harmonic_ratio:.4f}")
    print(f"Expected Frequency: {freq} Hz")
    print(f"Main Magnitude: {peak_magnitude:.4f}")
    print(f"Harmonic Magnitudes: {harmonic_mags}")
    print("Analysis Result:")

    if np.abs(peak_freq - freq) < 2 and harmonic_ratio < 0.1:
        print("The stream is a pure sine wave.")
    else:
        print("The stream is a complex wave (distorted or contains harmonics).")


if __name__ == "__main__":
    sample_rate = 48000
    limited = limiter_simulator(freq=440, amplitude=1.2, sample_rate=sample_rate, duration=0.02, ceiling=1.0, plot=False)
    sine_distortion_analysis(freq=440, stream=limited, sample_rate=sample_rate, plot=True)

    # print(limited)
