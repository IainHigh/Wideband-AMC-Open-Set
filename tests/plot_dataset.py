#!/usr/bin/python3

import os
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from scipy.signal import butter, filtfilt, firwin
import sys

np.Inf = np.inf  # Fix for a bug in the numpy library

# Results can be compared to the IQEngine Signal Generator:
# https://iqengine.org/siggen

# Read the configs/system_parameters.json file.
with open("./configs/system_parameters.json") as f:
    system_parameters = json.load(f)

working_directory = system_parameters["Working_Directory"]
sys.path.append(working_directory)

rng_seed = system_parameters["Random_Seed"]
np.random.seed(rng_seed)

dataset_directory = system_parameters["Dataset_Directory"]


#####################################################################
####################### MODIFIABLE VARIABLES ########################
#####################################################################

# Directories:
dataset_name = "default"

time_domain_output_path = "./tests/figures/Time-Domain"
frequency_domain_output_path = "./tests/figures/Frequency-Domain"
constellation_diagram_output_path = "./tests/figures/Constellation-Diagrams"
spectrogram_output_path = "./tests/figures/Spectrograms"

# Time domain plotting parameters - time domain is too large to plot all at once
time_domain_length = 200
time_domain_start_index = 0

# Spectrogram plotting parameters:
spectrogram_fft_size = 4096  # Size of the FFT

#####################################################################
######################### HELPER FUNCTIONS #########################
#####################################################################


def bandpass_filter(data, sampling_rate, lowcut, highcut, order=10):
    """
    Designs and applies a Butterworth bandpass filter to a real-valued signal.
    Ensures that the critical frequencies are strictly between 0 and 1.
    """
    nyq = 0.5 * sampling_rate

    # Prevent the lowcut from being <= 0 by clamping it to a small value (e.g., 1 Hz)
    if lowcut <= 0:
        lowcut = 1.0

    # Prevent the highcut from being >= nyq by clamping it just below nyq if needed
    if highcut >= nyq:
        highcut = nyq - 1.0

    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data)
    return y


def fir_bandpass_filter(data, sampling_rate, lowcut, highcut, numtaps=101, beta=8.6):
    nyq = 0.5 * sampling_rate
    taps = firwin(
        numtaps, [lowcut / nyq, highcut / nyq], pass_zero=False, window=("kaiser", beta)
    )
    # Using filtfilt for zero-phase filtering
    y = filtfilt(taps, [1.0], data)
    return y


def bandpass_complex(x, sampling_rate, lowcut, highcut):
    """
    Applies the bandpass filter separately to the real and imaginary parts of a complex signal.
    """
    # real_filtered = bandpass_filter(x.real, sampling_rate, lowcut, highcut)
    # imag_filtered = bandpass_filter(x.imag, sampling_rate, lowcut, highcut)
    real_filtered = fir_bandpass_filter(x.real, sampling_rate, lowcut, highcut)
    imag_filtered = fir_bandpass_filter(x.imag, sampling_rate, lowcut, highcut)
    return real_filtered + 1j * imag_filtered


def delete_existing_plots():
    for path in [
        time_domain_output_path,
        frequency_domain_output_path,
        constellation_diagram_output_path,
        spectrogram_output_path,
    ]:
        for file in os.listdir(path):
            os.remove(f"{path}/{file}")


def get_data(file):
    ## get meta
    with open(file + ".sigmf-meta") as _f:
        f_meta = json.load(_f)

    ## get data
    with open(file + ".sigmf-data", "rb") as _f:
        f_data = np.load(_f)

    # Extract metadata
    annotation = f_meta["annotations"][0]
    modscheme = annotation["rfml_labels"]["modclass"]
    sampling_rate = annotation["sampling_rate"]
    center_frequencies = annotation["center_frequencies"]
    # Try to get additional parameters needed for bandwidth calculation.
    sps = f_meta["annotations"][1]["filter"]["sps"]
    beta = f_meta["annotations"][1]["filter"]["rolloff"]

    return f_data, modscheme, center_frequencies, sampling_rate, sps, beta


#####################################################################
######################### PLOTTER FUNCTIONS #########################
#####################################################################


def plot_time_domain_diagram(f_data, sampling_rate):
    # Ensure we have enough data
    num_samples = len(f_data) // 2  # Number of I/Q pairs

    # Adjust time_domain_length dynamically if needed
    plot_length = min(time_domain_length, num_samples - time_domain_start_index)

    # Extract I/Q components with correct indexing
    I = f_data[
        time_domain_start_index * 2 : (time_domain_start_index + plot_length) * 2 : 2
    ]
    Q = f_data[
        time_domain_start_index * 2
        + 1 : (time_domain_start_index + plot_length) * 2 : 2
    ]

    # Ensure I and Q are the same length
    min_length = min(len(I), len(Q))
    I = I[:min_length]
    Q = Q[:min_length]

    # Generate correct time axis based on actual sample length
    time_axis = np.arange(min_length) / sampling_rate

    plt.figure()
    plt.plot(time_axis, I, label="I", alpha=0.8)
    plt.plot(time_axis, Q, label="Q", alpha=0.8)
    plt.grid()
    plt.title(f"Time Domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(f"{time_domain_output_path}/TimeDomain.png")
    plt.close()


def plot_frequency_domain_diagram(f_data, center_frequencies, sampling_rate):
    I = f_data[0::2]
    Q = f_data[1::2]
    x = I + 1j * Q  # Construct complex signal

    # Compute FFT and shift spectrum
    X = np.fft.fft(x)
    X = np.fft.fftshift(X)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), d=1 / sampling_rate))

    # Convert FFT magnitude to dB
    PSD = 10 * np.log10(np.abs(X) ** 2 + 1e-12)  # Avoid log(0)

    # Duplicate and shift the spectrum
    freqs_extended = np.concatenate((freqs, freqs + sampling_rate))
    PSD_extended = np.concatenate((PSD, PSD))  # Repeat FFT data

    # Keep only positive frequencies
    valid_idx = freqs_extended >= 0
    freqs_final = freqs_extended[valid_idx]
    PSD_final = PSD_extended[valid_idx]

    plt.figure()
    plt.plot(freqs_final, PSD_final, label="PSD")
    for f in center_frequencies:
        plt.axvline(
            f, color="red", linestyle="--", alpha=0.7, label=f"Center {f/1e6:.2f} MHz"
        )

    plt.grid()
    plt.title(f"Frequency Domain")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dB]")
    plt.legend()
    plt.savefig(f"{frequency_domain_output_path}/FrequencyDomain.png")
    plt.close()


def plot_constellation_diagram(
    f_data, modschemes, center_frequencies, sampling_rate, channel_bw
):
    """
    For each channel, isolate the desired band using a bandpass filter,
    downconvert it to baseband (undo the frequency shift), and then plot
    the constellation diagram.
    """
    # Extract composite I/Q signal
    I_total = f_data[0::2]
    Q_total = f_data[1::2]
    x = I_total + 1j * Q_total  # Composite wideband signal

    for i, f_c in enumerate(center_frequencies):
        modscheme = modschemes[i]

        # Define bandpass filter cutoff frequencies for channel f_c.
        lowcut = f_c - channel_bw / 2
        highcut = f_c + channel_bw / 2

        # Isolate the channel by filtering the composite signal.
        x_filtered = bandpass_complex(x, sampling_rate, lowcut, highcut)

        # Create a time vector for downconversion.
        t = np.arange(len(x_filtered)) / sampling_rate

        # Downconvert by applying the conjugate of the carrier frequency shift.
        # This shifts the channel to baseband.
        x_downconverted = x_filtered * np.exp(-1j * 2 * np.pi * f_c * t)

        # Extract the baseband I and Q components.
        I_unshifted = np.real(x_downconverted)
        Q_unshifted = np.imag(x_downconverted)

        # Perform density estimation for a nicer visualization.
        xy = np.vstack([I_unshifted, Q_unshifted])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        I_unshifted, Q_unshifted, z = I_unshifted[idx], Q_unshifted[idx], z[idx]

        # Plot the constellation diagram.
        plt.figure()
        plt.scatter(I_unshifted, Q_unshifted, c=z, s=1, cmap="viridis")
        plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.xlabel("I (In-phase)")
        plt.ylabel("Q (Quadrature)")
        plt.title(f"Constellation Diagram ({modscheme}) at {f_c/1e6:.2f} MHz")
        cbar = plt.colorbar()
        cbar.set_label("Density")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(
            f"{constellation_diagram_output_path}/{modscheme}_{int(f_c/1e6)}MHz.png"
        )
        plt.close()


def plot_spectrogram(f_data, sampling_rate):
    I = f_data[0::2]
    Q = f_data[1::2]
    x = I + 1j * Q

    num_rows = int(np.floor(len(x) / spectrogram_fft_size))
    spectrogram = np.zeros((num_rows, spectrogram_fft_size))

    for i in range(num_rows):
        segment = x[i * spectrogram_fft_size : (i + 1) * spectrogram_fft_size]
        spectrogram[i, :] = 10 * np.log10(np.abs(np.fft.fft(segment)) ** 2)

    plt.figure()
    plt.imshow(
        spectrogram,
        aspect="auto",
        extent=[0, sampling_rate, 0, len(x) / sampling_rate],
        cmap="inferno",
    )
    plt.colorbar(label="Power [dB]")
    plt.title(f"Spectrogram")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Time [s]")
    plt.savefig(f"{spectrogram_output_path}/Spectrogram.png")
    plt.close()


def main():
    delete_existing_plots()
    dataset_path = dataset_directory + "/" + dataset_name
    files = os.listdir(os.path.abspath(dataset_path))
    files = [os.path.join(dataset_path, f.split(".")[0]) for f in files]
    files = list(set(files))

    for file in files:
        # Get the data and metadata. Note that we now also retrieve sps and beta.
        f_data, modschemes, center_frequencies, sampling_rate, sps, beta = get_data(
            file
        )
        plot_time_domain_diagram(f_data, sampling_rate)
        plot_frequency_domain_diagram(f_data, center_frequencies, sampling_rate)

        # Compute channel_bw from the symbol rate and roll-off factor if available.
        if sps is not None and beta is not None:
            symbol_rate = sampling_rate / sps
            channel_bw = symbol_rate * (1 + beta)
            plot_constellation_diagram(
                f_data, modschemes, center_frequencies, sampling_rate, channel_bw
            )
        else:
            print(
                "Symbol rate or roll-off factor not available. Cannot compute channel bandwidth."
            )

        plot_spectrogram(f_data, sampling_rate)


if __name__ == "__main__":
    main()
