# Libs
import argparse
import numpy as np
# App
from wave import Soundwave, gen_inpulse
import dynamics as dyn
import filter as filt
import noise
import analysis as anls
import fft

stage_options = {
    "dc": noise.add_dc_offset,
    "sine": noise.add_dc_offset,
}


def main():
    global verbose
    parser = argparse.ArgumentParser(description="Script to run various DSP simulations for testing purposes")
    parser.add_argument('--chain', '-c', type=str, required=True, help='The chain to run, coma-separated')
    parser.add_argument('--duration', '-d', type=float, default=0.1, help='Duration of the simulation in seconds')
    parser.add_argument('--sample_rate', '-r', type=int, default=48000, help='Sample rate of the input signal')
    # Debug
    parser.add_argument('--verbose', action='store_true', help='Enable verbose print statement for debugging the script')

    # Arg parsing
    args = parser.parse_args()
    sample_rate = args.sample_rate
    duration = args.duration
    # fundamental_freq = 1000 # Eventually migrate as an arg
    # peak_amplitude = 1.001
    # sine = Soundwave(freq=fundamental_freq, amplitude=peak_amplitude)
    # audio_stream = sine.generate(sample_rate, duration)

    stages = args.chain.split("->")
    audio_stream = np.zeros(int(duration * sample_rate))
    print(stages)
    for stage in stages:
        audio_stream = stage_options[stage](audio_stream, 50)
    bin_freqs, bin_magnitudes = fft.fft(audio_stream, sample_rate)
    fft.fft_plt(bin_freqs, bin_magnitudes)

    return

if __name__ == "__main__":
    main()