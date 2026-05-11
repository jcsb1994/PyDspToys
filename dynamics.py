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
    Apply a limiter on a sine wave.

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


