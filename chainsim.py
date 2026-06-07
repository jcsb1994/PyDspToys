# Libs
import argparse
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
# App
from wave import Soundwave, gen_inpulse
import dynamics as dyn
import filter as filt
import noise
import analysis as anls
import fft

sample_rate = 0
duration = 0

def timeplot(audio_stream, stg_args):
    for i in range(len(audio_stream)):
        plt.plot(audio_stream[i])
    plt.legend()
    plt.show()
    return audio_stream

def pan(audio_stream, pan_percent):
    if len(audio_stream) != 2:
        print("Warning: Skipped panning for mono or compex streams")
        return audio_stream
    pan = abs(pan_percent / 100)
    if pan_percent > 0:
        audio_stream[1] += audio_stream[0] * pan
        audio_stream[0] *= (1 - pan)
    if pan_percent < 0:
        audio_stream[0] += audio_stream[1] * pan
        audio_stream[1] *= (1 - pan)
    return audio_stream

stage_options = {
    # Signal and noise
    "dc": lambda audio_stream, stg_args: ( # Args: 0:offset
        noise.add_dc_offset(audio_stream, float(stg_args[0]))),
    "sine": lambda audio_stream, stg_args: (
        audio_stream + Soundwave(freq=float(stg_args[0]), amplitude=float(stg_args[1])).generate(sample_rate, duration)),
    # Processing
    "limiter": lambda audio_stream, stg_args: ( # Args: 0:ceiling
        np.clip(audio_stream, -float(stg_args[0]), float(stg_args[0]))),
    "bitcrush": lambda audio_stream, stg_args: (
        (steps := 2 ** int(stg_args[0]) - 1), # bit nb
        np.round((audio_stream + 1) / 2 * steps) / steps * 2 - 1)[-1], # Normalize to 0-1, quantize, and scale back, -1 to return just the last line
        # Analysis
    "pan": lambda audio_stream, stg_args: ( # Args: 0:-100to+100% (L-R)
        pan(audio_stream, float(stg_args[0]))),
    "fft": lambda audio_stream, stg_args: (
        fft.fft_stereo_plot(audio_stream, sample_rate, stg_args[0])),
    "spectrogram": lambda audio_stream, stg_args: (
        fft.spectrogram_plt(stream, sample_rate), audio_stream),
    "timeplot": lambda audio_stream, stg_args: (
        timeplot(audio_stream, stg_args)
    )
}

def main():
    global verbose, sample_rate, duration
    parser = argparse.ArgumentParser(description="Script to run various DSP simulations for testing purposes")
    parser.add_argument('--chain', '-c', type=str, required=True, help='The chain to run, coma-separated')
    parser.add_argument('--duration', '-d', type=float, default=0.01, help='Duration of the simulation in seconds')
    parser.add_argument('--sample_rate', '-r', type=int, default=48000, help='Sample rate of the input signal')
    parser.add_argument('--nb_chans', '-n', type=int, default=2, help='Number of audio channels going through the chain')
    # Debug
    parser.add_argument('--verbose', action='store_true', help='Enable verbose print statement for debugging the script')

    # Arg parsing
    args = parser.parse_args()
    sample_rate = args.sample_rate
    duration = args.duration
    nb_chans = args.nb_chans
    # fundamental_freq = 1000 # Eventually migrate as an arg
    # peak_amplitude = 1.001
    # sine = Soundwave(freq=fundamental_freq, amplitude=peak_amplitude)
    # audio_stream = sine.generate(sample_rate, duration)

    audio_stream = []
    for _ in range(nb_chans):
        audio_stream.append(
            np.zeros(int(duration * sample_rate))
        )
    print(f"Nb of chans {nb_chans} and len {len(audio_stream)}")

    stages = args.chain.split("->")
    for stage in stages:
        stg_args = None
        if stage.endswith(")"):
            stage, raw_args = stage.split("(", 1)
            stg_args = raw_args.rstrip(")").split(",")
        audio_stream = stage_options[stage](audio_stream, stg_args)

    return

if __name__ == "__main__":
    main()