"""
Tools to generate noise for DSP simulations.
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
    # Delay conversion
    delay_samples = int(delay * sample_rate)

    # Create delayed copy
    delayed = np.zeros_like(audio)

    if delay_samples < len(audio):
        delayed[delay_samples:] = audio[:-delay_samples]

    return delayed
