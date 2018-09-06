# stft-playground.py
#
# This is a play-ground project to experiment with FFT/STFT, SciPy and Numpy. It
# operates on data generated by the program itself.

import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal as signal

# Default amplitude of the generated sine wave signal.
DEFAULT_AMPLITUDE = 50

# Default frequency of the generated sine wave signal.
DEFAULT_FREQUENCY = 18000

# Default sampling rate of the generated sine wave signal.
DEFAULT_SAMPLING_RATE = 44100

# Default bin size used in the calculation of the Short-Term Fourier Transform.
DEFAULT_STFT_BIN_SIZE = 1000

def generate_sine_wave(frequency, sampling_rate, amplitude=DEFAULT_AMPLITUDE):
    t = np.arange(sampling_rate)
    y = amplitude * np.sin(2 * np.pi * t * (frequency / sampling_rate)).astype(np.float32)
    return t, y

def find_signal_frequency(signal_data, sampling_rate,
        bin_size=DEFAULT_STFT_BIN_SIZE):
    '''
    Find the signal frequency by using an STFT.
    '''
    # Calculate the signal STFT.
    freqs, time, Zxx = signal.stft(signal_data, fs=sampling_rate,
            nperseg=bin_size)
    # STFT returns complex numbers. Convert these complex numbers into amplitude
    # so we can work in the Real space.
    amp_Zxx = np.abs(Zxx)
    # Find the frequency that has the global maximum.
    peak_freq = 0
    peak_freq_max = -1
    for i in range(len(freqs)):
        max_ind = amp_Zxx[i].argmax()
        if amp_Zxx[i][max_ind] > peak_freq_max:
            peak_freq = freqs[i]
            peak_freq_max = amp_Zxx[i][max_ind]
    return peak_freq

def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--amplitude', type=int,
        default=DEFAULT_AMPLITUDE)
    parser.add_argument('--frequency', type=int,
        default=DEFAULT_FREQUENCY)
    parser.add_argument('--sampling_rate', type=int,
        default=DEFAULT_SAMPLING_RATE)
    parser.add_argument('--with_plot', action='store_true')
    args = parser.parse_args()
    # Print set-up.
    print('-----------------------------------')
    print('Configuration:')
    print('  amplitude:     %f' % args.amplitude)
    print('  frequency:     %f' % args.frequency)
    print('  sampling rate: %f' % args.sampling_rate)
    print('-----------------------------------')
    # When using STFT, calculate the number of samples that will be used. Let's
    # assume 40ms has been elapsed.
    stft_nsamples = int(args.sampling_rate * 40e-3)
    # Generate sine wave signal and add some noise.
    t, y = generate_sine_wave(args.frequency, args.sampling_rate,
            args.amplitude)
    noise = np.random.normal(0, 50, len(t))
    noisy_signal = y + noise
    # Determine the measured frequency.
    measured_freq = find_signal_frequency(noisy_signal[0:stft_nsamples],
        args.sampling_rate)
    print('noisy signal measured frequency: %dHz' % measured_freq)
    # Plot some diagrams.
    if args.with_plot:
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        # Plot the noisy sine wave signal.
        plt.subplot(211)
        plt.plot(t, noisy_signal)
        plt.title('Noise + Sine Wave')
        #  Plot the stft.
        plt.subplot(212)
        plt.title('STFT')
        f, t, Zxx = signal.stft(noisy_signal, fs=args.sampling_rate,
            nperseg=DEFAULT_STFT_BIN_SIZE)
        plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=args.amplitude)
        plt.show()

if __name__ == '__main__':
    main()
