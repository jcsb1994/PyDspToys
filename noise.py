"""
Tools to generate noise and audio artifacts for DSP simulations.
"""

import numpy as np

def add_dc_offset(audio, offset):
    """
    Add 0Hz contents
    """
    audio = np.asarray(audio)
    return audio + offset

def gen_delayed_copy(audio, sample_rate, delay):
    """
    Generate a copy of input audio stream with samples delayed by specified amount.
    First samples are zero filled.
    """
    audio = np.asarray(audio)

    # Delay conversion
    delay_samples = int(delay * sample_rate)

    if delay_samples <= 0:
        return np.copy(audio)

    # Create delayed copy
    delayed = np.zeros_like(audio)

    if delay_samples < len(audio):
        delayed[delay_samples:] = audio[:-delay_samples]

    return delayed

def add_convolution(audio, sample_rate, delay, decay):
    """
    Add a simple echo using convolution.
    This function creates an "impulse response" representing an echo system, then convolves the input signal with it.

    This keeps the original signal and applies a quieter copy of it (decay adjusts how quiet, delay how delayed the copy is)
    y[n] = x[n] + decay * x[n - D]
        x[n] = input signal
        y[n] = output signal
        D    = delay in samples
    """

    # Convert delay from seconds into samples.
    delay_samples = int(delay * sample_rate)

    # Create the impulse response array: +1 for the original sample
    # Lowercase h: time-domain impulse response
    # The length of h is the number of taps needed to create the delay
    h = np.zeros(delay_samples + 1)

    # We adjust two taps in the filter: first and last samples
    # [1, 0, 0, ... (delay_samples times), decay] is applied for each sample
    # so each sample adds zeroes + trailing echo down the stream
    h[0] = 1.0
    h[-1] = decay # e.g. 0.3 would do a 30% volume echo

    # Slide the impulse response h[] across the input signal audio[].
    # Every input sample generates a tiny shifted copy of the impulse response, which are all summed
    return np.convolve(audio, h)