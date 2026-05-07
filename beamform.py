import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


SPEED_OF_SOUND = 343.0  # m/s


@dataclass
class Mic:
    x: float
    y: float

@dataclass
class Soundwave:
    freq: float
    source: Tuple[float, float]
    amplitude: float


def beamform_simulation(
    signal : Soundwave,
    noise : Soundwave,
    mics: List[Mic],
    duration: float = 1.0,
    sample_rate: int = 48000,
    beamforming: bool = True,
    steering_source: Tuple[float, float] | None = None,
):
    """
    Simulates a microphone array receiving:
      - desired sinewave signal
      - noise sinewave from another direction

    Performs:
      - time-of-arrival calculation
      - optional delay-and-sum beamforming
      - stereo spatial rendering

    Returns:
      {
        "time": np.ndarray,
        "mono": np.ndarray,
        "stereo": np.ndarray shape (N, 2),
        "mic_signals": List[np.ndarray],
        "signal_delays": List[float],
        "noise_delays": List[float]
      }
    """

    if len(mics) < 1 or len(mics) > 4:
        raise ValueError("Mic count must be between 1 and 4")

    if steering_source is None:
        steering_source = signal.source

    t = np.arange(int(duration * sample_rate)) / sample_rate

    signal.source = np.array(signal.source)
    noise.source = np.array(noise.source)
    steering_source = np.array(steering_source)

    mic_signals = []
    signal_delays = []
    noise_delays = []

    # ------------------------------------------------------------
    # Generate received signals at each mic
    # ------------------------------------------------------------
    for mic in mics:
        mic_pos = np.array([mic.x, mic.y])

        # Distance to sources
        d_signal = np.linalg.norm(signal.source - mic_pos)
        d_noise = np.linalg.norm(noise.source - mic_pos)

        # Time delays
        signal_delay = d_signal / SPEED_OF_SOUND
        noise_delay = d_noise / SPEED_OF_SOUND

        signal_delays.append(signal_delay)
        noise_delays.append(noise_delay)

        # Delayed sinewaves
        signal_wave = signal.amplitude * np.sin(
            2 * np.pi * signal.freq * (t - signal_delay)
        )

        noise_wave = noise.amplitude * np.sin(
            2 * np.pi * noise.freq * (t - noise_delay)
        )

        combined = signal_wave + noise_wave
        mic_signals.append(combined)

    mic_signals = np.array(mic_signals)

    # ------------------------------------------------------------
    # Delay-and-sum beamforming
    # ------------------------------------------------------------
    if beamforming:
        aligned = []

        for i, mic in enumerate(mics):
            mic_pos = np.array([mic.x, mic.y])

            d_steer = np.linalg.norm(steering_source - mic_pos)
            steer_delay = d_steer / SPEED_OF_SOUND

            relative_delay = steer_delay - min(signal_delays)

            shift_samples = int(round(relative_delay * sample_rate))

            shifted = np.roll(mic_signals[i], -shift_samples)
            aligned.append(shifted)

        aligned = np.array(aligned)

        mono = np.mean(aligned, axis=0)

    else:
        mono = np.mean(mic_signals, axis=0)

    # ------------------------------------------------------------
    # Stereo spatialization
    # ------------------------------------------------------------
    source_x = signal.source[0]

    # Pan from [-1,1]
    pan = np.clip(source_x / 5.0, -1.0, 1.0)

    left_gain = np.sqrt((1 - pan) / 2)
    right_gain = np.sqrt((1 + pan) / 2)

    left = mono * left_gain
    right = mono * right_gain

    stereo = np.stack([left, right], axis=1)

    return {
        "time": t,
        "mono": mono,
        "stereo": stereo,
        "mic_signals": mic_signals,
        "signal_delays": signal_delays,
        "noise_delays": noise_delays,
    }


import numpy as np
import matplotlib.pyplot as plt


def plot_beamforming_streams(
    stereo_stream: np.ndarray, # Shape (N,2) stereo output from beamform_simulation()
    signal: Soundwave,
    noise: Soundwave,
    sample_rate: int = 48000,
    title: str = "Beamforming Streams",
):
    """
    Plots:
      1. Pure signal wave
      2. Pure noise wave
      3. Left stereo output
      4. Right stereo output

    Each stream is shown in a different color.
    """

    if stereo_stream.ndim != 2 or stereo_stream.shape[1] != 2:
        raise ValueError("stereo_stream must have shape (N,2)")

    n = stereo_stream.shape[0]

    t = np.arange(n) / sample_rate

    # Reference source waves
    signal_wave = signal.amplitude * np.sin(
        2 * np.pi * signal.freq * t
    )

    noise_wave = noise.amplitude * np.sin(
        2 * np.pi * noise.freq * t
    )

    left = stereo_stream[:, 0]
    right = stereo_stream[:, 1]

    plt.figure(figsize=(14, 7))

    plt.plot(t, signal_wave, label="Signal Wave", linewidth=1)
    plt.plot(t, noise_wave, label="Noise Wave", linewidth=1)
    plt.plot(t, left, label="Left Output", linewidth=1)
    plt.plot(t, right, label="Right Output", linewidth=1)

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":

    mics = [
        Mic(-0.15, 0.0),
        Mic(-0.05, 0.0),
        Mic(0.05, 0.0),
        Mic(0.15, 0.0),
    ]

    sig = Soundwave(440.0, (-2.0, 3.0), 1.0)
    nse = Soundwave(6000.0, (2.0, 3.0), 1.0) # Low frequency noise

    result = beamform_simulation(
        signal = sig,
        noise = nse,
        mics=mics,
        duration=0.03,
        sample_rate=48000,
        beamforming=False,
    )

    stereo = result["stereo"]

    print("Stereo", stereo)
    print("Stereo shape:", stereo.shape)
    print("Signal delays:", result["signal_delays"])
    print("Noise delays:", result["noise_delays"])

    plot_beamforming_streams(
        stereo_stream=result["stereo"],
        signal = sig,
        noise = sig,
    )