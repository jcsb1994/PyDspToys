import numpy as np
import matplotlib.pyplot as plt
import filter
from wave import Soundwave
import fft

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

    bin_freqs, bin_magnitudes = fft.fft(stream, sample_rate)
    if plot:
        fft.fft_plt(bin_freqs, bin_magnitudes)

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
    suspect_distortion = False
    for h in harmonics:
        # For each harmonic, check if there is a significant peak within accepted margin of any odd harmonic multiple
        for i in range(lowest_harmonic_checked, highest_harmonic_checked+1):
            curr_odd_multiple = freq * (i if i % 2 == 1 else i + 1)  # Check odd multiples (1st, 3rd, 5th, etc.)
            percentage_accepted = 0.1
            above_accepted_of_fundamental_multiple = h > (curr_odd_multiple - (freq * percentage_accepted))
            below_accepted_of_fundamental_multiple = h < (curr_odd_multiple + (freq * percentage_accepted))

            if below_accepted_of_fundamental_multiple and above_accepted_of_fundamental_multiple:
                print(f"Warning: Significant odd harmonic detected at {h:.2f} Hz")
                suspect_distortion = True
                break

    if suspect_distortion:
        # Apply a LPF
        # 8th-order filter drops at 48 dB per octave (each octave doubles the frequency)
        stream = filter.lpf(stream, sample_rate, freq, order=8) # Input sine will be -3dB (cutoff) but we can filter out harmonics
        bin_freqs, bin_magnitudes = fft.fft(stream, sample_rate)
        if plot:
            fft.fft_plt(bin_freqs, bin_magnitudes)

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
    sine = Soundwave(freq=440, amplitude=1.2)

    limited = limiter_simulator(freq=sine.freq, amplitude=sine.amplitude, sample_rate=sample_rate, duration=0.02, ceiling=1.0, plot=False)
    sine_distortion_analysis(freq=sine.freq, stream=limited, sample_rate=sample_rate, plot=True)

