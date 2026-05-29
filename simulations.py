# Libs
import argparse
# App
from wave import Soundwave, gen_inpulse
import dynamics as dyn
import filter as filt
import noise
import analysis as anls
import fft

def convolved_impulse_simulation(duration, sample_rate):
    stream = gen_inpulse(duration, sample_rate)
    # 10ms delay
    stream = noise.add_convolution(stream, sample_rate, delay=0.010, decay=0.5)
    # Recursively convolve, now should see 4 impulses with 5 ms delay
    stream = noise.add_convolution(stream, sample_rate, delay=0.005, decay=0.5)
    return stream

def comb_filter_simulation(
    audio,
    delay,
    sample_rate: int,
    mix: float = 1.0,
):
    """
    Creates a comb-filtered audio stream by mixing the original with a delayed copy.
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

    return output


simulations = ["comb", "impulse", "limiter", "dcblock"]

def main():
    global verbose, globalConditions
    parser = argparse.ArgumentParser(description="Script to run various DSP simulations for testing purposes")
    parser.add_argument('--simulation', '-s', type=str, required=True, choices=simulations, help='The simulation to run')
    parser.add_argument('--duration', '-d', type=float, default=1.0, help='Duration of the simulation in seconds')
    parser.add_argument('--sample_rate', '-r', type=int, default=48000, help='Sample rate of the input signal')
    # Debug
    parser.add_argument('--verbose', action='store_true', help='Enable verbose print statement for debugging the script')

    # Arg parsing
    args = parser.parse_args()
    sample_rate = args.sample_rate
    duration = args.duration
    fundamental_freq = 1000 # Eventually migrate as an arg
    peak_amplitude = 1.2
    sine = Soundwave(freq=fundamental_freq, amplitude=peak_amplitude)
    audio_stream = sine.generate(sample_rate, duration)

    # TODO: Simulation-specific parameters should be stored in a json file, so if you want to change some specific delay could do it there

    if args.simulation == "comb":
        """
        Simulation of a sine being layered with a delayed copy in order to create an (undesired) comb filter effect
        """
        audio_stream = sine.generate(sample_rate, duration)
        anls.plot_time_domain(audio_stream)
        # Delay of 1 ms means the delta f will be 1kHz, notches will be every x.5kHz
        audio_stream = comb_filter_simulation(audio_stream, delay=0.001, sample_rate=sample_rate)

    elif args.simulation == "impulse":
        audio_stream = convolved_impulse_simulation(duration, sample_rate)
        anls.plot_time_domain(audio_stream)

    elif args.simulation == "limiter":
        limited = dyn.limiter(freq=sine.freq, amplitude=sine.amplitude, sample_rate=sample_rate, duration=duration, ceiling=1.0, plot=True)
        anls.sine_distortion_analysis(freq=sine.freq, stream=limited, sample_rate=sample_rate, plot=True)

    elif args.simulation == "dcblock":
        """
        Simulation adds DC components, then removes them
        """
        audio_stream = noise.add_dc_offset(audio_stream, 50)
        bin_freqs, bin_magnitudes = fft.fft(audio_stream, sample_rate)
        fft.fft_plt(bin_freqs, bin_magnitudes)
        audio_stream = filt.efficient_dcblock_filter(audio_stream)
        # audio_stream = filt.one_band_pass_filter(audio_stream, sample_rate, cutoff_freq=1000, order=4)
        bin_freqs, bin_magnitudes = fft.fft(audio_stream, sample_rate)
        fft.fft_plt(bin_freqs, bin_magnitudes)

if __name__ == "__main__":
    main()