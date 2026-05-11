# Libs
import numpy as np
import matplotlib.pyplot as plt
# App
import filter
from wave import Soundwave
import fft
import analysis as anls

def limiter(freq, amplitude, sample_rate, duration, ceiling, plot=False):
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

if __name__ == "__main__":
    sample_rate = 48000
    sine = Soundwave(freq=440, amplitude=1.2)

    limited = limiter(freq=sine.freq, amplitude=sine.amplitude, sample_rate=sample_rate, duration=0.02, ceiling=1.0, plot=False)
    anls.sine_distortion_analysis(freq=sine.freq, stream=limited, sample_rate=sample_rate, plot=True)

