import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def fft(stream, sample_rate):

    # N is the standard name for the length of the input signal in FFT analysis
    # (Using N and sample rate you can infer the time duration of the input signal)
    N = len(stream)

    # The bin size is dependent on 2 things: the sample rate and the length of the input signal (N).
    # The bin size represents the frequency resolution of the FFT and is sample_rate / N
    # This is because in order to see a frequency component in the FFT output, you need at least one full cycle of that frequency
    # So for example to see a 0.5Hz (bin size 0.5) N must be double the sample rate
    bin_size = sample_rate // N

    # This returns a list of size N with bins in the format i+j.
    # Because of the Nyquist theorem, we can only detect up to sample_rate/2
    # Only half of the bins are used for 0Hz to Nyquist frequency (other half is the same with negative freqs)
    fft = np.fft.fft(stream)

    # The frequency bins from np.fft.fftfreq(N, 1/sample_rate) go from 0 up to just below the Nyquist frequency, then from
    # negative Nyquist back up to just below 0. So, the first half (bin_freqs[:N//2]) corresponds to 0 Hz up to just below the Nyquist frequency
    pos_freqs_len = N // 2
    bin_magnitudes = np.abs(fft)
    bin_freqs = np.fft.fftfreq(N, 1/sample_rate)
    # We are working with real signals so take the positive frequencies only (the first half of the FFT output)
    bin_magnitudes = bin_magnitudes[:pos_freqs_len]
    bin_freqs = bin_freqs[:pos_freqs_len]

    print(f"Bin Size: {bin_size} Hz")
    print(f"Number of Bins: {len(bin_freqs)}")
    print(f"Frequency of first few bins: {bin_freqs[:10]} Hz")
    print(f"Magnitudes of first few bins: {bin_magnitudes[:10]}")

    return bin_freqs, bin_magnitudes

def fft_plt(bin_freqs, bin_magnitudes):
    plt.figure(figsize=(10, 4))
    plt.plot(bin_freqs, 20 * np.log10(bin_magnitudes + 1e-12))
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid()
    plt.show()

def spectrogram_plt(stream, sample_rate):
    f, t, Sxx = signal.spectrogram(stream, sample_rate)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()