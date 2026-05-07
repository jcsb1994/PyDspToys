import numpy as np
import matplotlib.pyplot as plt
from typing import List

def complex_wave(frequencies : List[float], duration=1.0, sample_rate=1000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Start with a "silent" array of zeros
    complex_wave = np.zeros_like(t)
    π = np.pi
    # Add each sine wave to the total
    for f in frequencies:
        complex_wave += np.sin(2 * π * f * t) # Multiply by t as we progress in time

    return t, complex_wave

if __name__ == "__main__":
    freq_list = [3.3, 10, 30, 90, 270, 710]
    t, y = complex_wave(freq_list)

    plt.figure(figsize=(10, 4))
    plt.plot(t, y)
    plt.title(f"Composite Wave: {freq_list} Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
