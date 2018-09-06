import argparse
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pyaudio
import scipy as sp
import scipy.signal as signal
import wave

# Default frequency used by the speaker. In the final application, there will be
# multiple speakers having their own frequencies. For now, let's assume we have
# one speaker.
DEFAULT_SPEAKER_FREQUENCY = 19007.1

# Default number of msec spent recording an audio chunk before trying to detect
# the signal frequency.
DEFAULT_RECORDING_PERIOD = 40

# Default amplitude for the sine wave signal.
DEFAULT_AMPLITUDE = 50

# Default frequency for the sine wave signal, in Hz.
DEFAULT_FREQUENCY = 18000

# Default sampling rate for the sine wave signal, in Hz.
DEFAULT_SAMPLING_RATE = 44100

# Speed of the sound in normal condition (temperature, pressure, etc.).
SOUND_SPEED = 343.0

# Number of frequency bins for the STFT calculation.
STFT_FREQ_BIN_SIZE = 1000


def generate_sine_wave(amplitude, frequency, sampling_rate):
    t = np.arange(sampling_rate)
    y = amplitude * np.sin(2 * np.pi * t * (frequency / sampling_rate)).astype(np.float32)
    return t, y


def signal_find_frequency_with_fft(signal):
    # Fourier Transform produces sequences of numbers in complex space. Use
    # abs() to get the amplitude spectrum.
    fft = np.abs(np.fft.fft(signal))
    # Generating a sine wave means summing two complex exponentials - one at a
    # positive frequency, one at a negative frequency. Since we took the abs()
    # values, the FFT will show two positive equal peaks. The second peak
    # corresponds to the negative frequency. Let's take half of FFT result to
    # detect the first peak.
    return fft[0:int(len(fft)/2)].argmax()


def signal_find_frequency_with_stft(sig, sampling_rate):
    freqs, time, Zxx = signal.stft(sig, fs=sampling_rate,
            nperseg=STFT_FREQ_BIN_SIZE)
    # Convert the complex numbers into amplitudes.
    aZxx = np.abs(Zxx)
    # Find the frequency band that has the highest peak.
    peak_freq = 0
    peak_freq_max = -1
    for i in range(len(freqs)):
        # Find the local maximum and compare it with the current global maximum.
        max_ind = aZxx[i].argmax()
        if aZxx[i][max_ind] > peak_freq_max:
            # Found a new global maximum.
            peak_freq = freqs[i]
            peak_freq_max = aZxx[i][max_ind]
    # Return the frequency band that contains the global maximum. This is the
    # best match for the signal frequency.
    return peak_freq


def calculate_velocity_from_doppler_shift(f0, f1, sound_speed=SOUND_SPEED):
    return sound_speed * (f1 - f0) / (f1 + f0)


def main_test():
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
    print('amplitude:     %f' % args.amplitude)
    print('frequency:     %f' % args.frequency)
    print('sampling rate: %f' % args.sampling_rate)
    print('')
    # When using STFT, calculate the number of samples that will be used. Let's
    # assume 40ms has been elapsed.
    stft_nsamples = int( args.sampling_rate * 40e-3 )
    # Generate sine wave signal and add some noise.
    t, y = generate_sine_wave(args.amplitude, args.frequency, args.sampling_rate)
    n = np.random.normal(0, 50, len(t))
    u = y + n
    print('noisy signal freq (w/ fft) : %dHz' % (signal_find_frequency_with_fft(u)))
    print('noisy signal freq (w/ stft): %dHz' %
            (signal_find_frequency_with_stft(u[0:stft_nsamples], args.sampling_rate)))
    # Calculate velocity.
    f0 = signal_find_frequency_with_fft(y)
    f1 = signal_find_frequency_with_stft(u[0:stft_nsamples], args.sampling_rate)
    vel = calculate_velocity_from_doppler_shift(f0, f1)
    print('estimated velocity: %f m/s' % vel)
    # Plot some diagrams.
    if args.with_plot:
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        # Plot the noisy sine wave signal.
        plt.subplot(211)
        plt.plot(t, u)
        plt.title('Noise + Sine Wave')
        #  Plot the stft.
        plt.subplot(212)
        plt.title('STFT')
        f, t, Zxx = signal.stft(u, fs=args.sampling_rate, nperseg=10000)
        plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=args.amplitude)
        plt.show()

def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--speaker_frequency', type=int,
        default=DEFAULT_SPEAKER_FREQUENCY)
    parser.add_argument('--recording_period', type=int,
        default=DEFAULT_RECORDING_PERIOD)
    parser.add_argument('--sampling_rate', type=int,
        default=DEFAULT_SAMPLING_RATE)
    args = parser.parse_args()
    # Initialize pyAudio.
    audio = pyaudio.PyAudio()
    # Start recording.
    stream = audio.open(format=pyaudio.paInt16, channels=1,
        rate=args.sampling_rate, input=True)
    chunk_size = int(args.sampling_rate * args.recording_period / 1000.0)
    # Main loop.
    xs = []
    ys = []
    pos = 0
    while True:
        # Record one chunk.
        chunk = stream.read(chunk_size)
        # Use an STFT to determine the signal frequency.
        audio_data = np.fromstring(chunk, np.int16)
        measured_freq = signal_find_frequency_with_stft(audio_data,
            args.sampling_rate)
        # Ignore doppler shift that are too big.
        if abs(measured_freq - args.speaker_frequency) <= 50:
            # Calculate the velocity from the doppler shift.
            velocity = calculate_velocity_from_doppler_shift(
                args.speaker_frequency, measured_freq)
            pos = pos + velocity * (args.recording_period / 1000.0)
        print('position: %f' % pos)


if __name__ == '__main__':
    main()
    #main_test()
