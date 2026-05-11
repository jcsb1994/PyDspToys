import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass
@dataclass
class Soundwave:
    freq: float
    amplitude: float
    source: Tuple[float, float] = (0.0, 0.0) # 2D space
    def generate(self, sample_rate, duration):
        """
        Generate the sine wave as a mono stream
        """
        t = np.arange(int(sample_rate * duration)) / sample_rate
        wave = self.amplitude * np.sin(2 * np.pi * self.freq * t)
        return wave

def gen_inpulse(duration, sample_rate):
    """
    Generate a mono impulse stream
    """
    impulse = np.zeros(int(duration * sample_rate))
    impulse[0] = 1.0
    return impulse

def gen_complex_wave(frequencies : List[float], sample_rate, duration=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Start with a "silent" array of zeros
    complex_wave = np.zeros_like(t)
    π = np.pi
    # Add each sine wave to the total
    for f in frequencies:
        complex_wave += np.sin(2 * π * f * t) # Multiply by t as we progress in time

    return t, complex_wave

def apply_gain(audio, gain_db):
    """
    Apply gain in dB.
    """
    gain_linear = 10 ** (gain_db / 20)
    return np.asarray(audio) * gain_linear

if __name__ == "__main__":
    freq_list = [3.3, 10, 30, 90, 270, 710]
    t, y = gen_complex_wave(freq_list)

    plt.figure(figsize=(10, 4))
    plt.plot(t, y)
    plt.title(f"Composite Wave: {freq_list} Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
