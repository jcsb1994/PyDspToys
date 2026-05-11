# Libs
# App
from wave import Soundwave
import noise
import analysis as anls

def comb_filter_simulation(
    audio,
    delay,
    sample_rate: int,
    mix: float = 1.0,
):
    """
    Creates a comb-filtered sine wave by mixing the original
    with a delayed copy.
    """
    audio1 = audio
    audio2 = noise.gen_delayed_copy(audio, sample_rate, delay)

    delay_samples, _ = anls.estimate_delay(audio1, audio2, sample_rate)

    # Mix original + delayed
    output = audio1 + (mix * audio2)

    # Comb filter spacing
    delay_ms = delay * 1000
    spacing_hz = 1 / delay

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

    return output,

if __name__ == "__main__":
    sample_rate = 48000
    sine = Soundwave(freq=10000, amplitude=1.2)
    audio_stream = sine.generate(sample_rate, 0.2)

    # Delay of 1 ms means the delta f will be 1kHz, notches will be every x.5kHz
    combed = comb_filter_simulation(audio_stream, delay=0.001, sample_rate=sample_rate)
