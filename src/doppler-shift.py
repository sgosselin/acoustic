import argparse
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pyaudio
import scipy as sp
import scipy.signal as signal

# Default frequency used by the speaker.
DEFAULT_SPEAKER_FREQUENCY = 19007.1

# Default recording period for each audio chunk before being processed.
DEFAULT_RECORDING_PERIOD = 40

# Default sampling rate when processing each audio chunk.
DEFAULT_SAMPLING_RATE = 44100

# Speed of the sound in normal condition (temperature, pressure, etc.).
SOUND_SPEED = 343.0

# Default bin size used in the calculation of the Short-Term Fourier Transform.
DEFAULT_STFT_BIN_SIZE = 1000

def calculate_velocity_from_doppler_shift(f0, f1, sound_speed=SOUND_SPEED):
    return sound_speed * (f1 - f0) / (f1 + f0)

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
        measured_freq = find_signal_frequency(audio_data,
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
