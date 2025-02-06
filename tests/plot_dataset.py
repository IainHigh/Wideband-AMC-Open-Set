#!/usr/bin/python3

import os
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

np.Inf = np.inf  # Fix for a bug in the numpy library

# Results can be compared to the IQEngine Signal Generator:
# https://iqengine.org/siggen

#####################################################################
####################### MODIFIABLE VARIABLES ########################
#####################################################################

# Directories:
dataset_path = "/exports/eddie/scratch/s2062378/data/default"

time_domain_output_path = "./tests/figures/Time-Domain"
frequency_domain_output_path = "./tests/figures/Frequency-Domain"
constellation_diagram_output_path = "./tests/figures/Constellation-Diagrams"
spectrogram_output_path = "./tests/figures/Spectrograms"

# Time domain plotting parameters - time domain is too large to plot all at once
time_domain_length = 200
time_domain_start_index = 0

# Spectrogram plotting parameters:
spectrogram_fft_size = 2048  # Size of the FFT

#####################################################################
#################### END OF MODIFIABLE VARIABLES #####################
#####################################################################

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
    modscheme = f_meta["annotations"][0]["rfml_labels"]["modclass"]
    sampling_rate = f_meta["annotations"][0]["sampling_rate"]
    center_frequencies = f_meta["annotations"][0]["center_frequencies"]

    return f_data, modscheme, center_frequencies, sampling_rate

def plot_time_domain_diagram(f_data, modscheme, sampling_rate):
    # Ensure we have enough data
    num_samples = len(f_data) // 2  # Number of I/Q pairs

    # Adjust time_domain_length dynamically if needed
    plot_length = min(time_domain_length, num_samples - time_domain_start_index)
    
    # Extract I/Q components with correct indexing
    I = f_data[time_domain_start_index * 2 : (time_domain_start_index + plot_length) * 2 : 2]
    Q = f_data[time_domain_start_index * 2 + 1 : (time_domain_start_index + plot_length) * 2 : 2]

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
    plt.title(f"Time Domain ({modscheme})")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(f"{time_domain_output_path}/{modscheme}.png")
    plt.close()


def plot_frequency_domain_diagram(f_data, modscheme, center_frequencies, sampling_rate):
    I = f_data[0::2]
    Q = f_data[1::2]
    x = I + 1j * Q  # Construct complex signal

    # Compute FFT and shift spectrum
    X = np.fft.fft(x)
    X = np.fft.fftshift(X)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), d=1/sampling_rate))

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
        plt.axvline(f, color="red", linestyle="--", alpha=0.7, label=f"Center {f/1e6:.2f} MHz")
    
    plt.grid()
    plt.title(f"Frequency Domain ({modscheme})")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dB]")
    plt.legend()
    plt.savefig(f"{frequency_domain_output_path}/{modscheme}.png")
    plt.close()

def plot_constellation_diagram(f_data, modscheme):
    I = f_data[0::2]
    Q = f_data[1::2]


    xy = np.vstack([I, Q])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    I, Q, z = I[idx], Q[idx], z[idx]

    plt.figure()
    plt.scatter(I, Q, c=z, s=1, cmap="viridis")
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.xlabel("I (In-phase)")
    plt.ylabel("Q (Quadrature)")
    plt.title(f"Constellation Diagram ({modscheme})")
    cbar = plt.colorbar()
    cbar.set_label("Density")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(f"{constellation_diagram_output_path}/{modscheme}.png")
    plt.close()

def plot_spectrogram(f_data, modscheme, sampling_rate):
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
    plt.title(f"Spectrogram ({modscheme})")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Time [s]")
    plt.savefig(f"{spectrogram_output_path}/{modscheme}.png")
    plt.close()

def main():
    delete_existing_plots()
    files = os.listdir(os.path.abspath(dataset_path))
    files = [os.path.join(dataset_path, f.split(".")[0]) for f in files]
    files = list(set(files))

    for file in files:
        f_data, modscheme, center_frequencies, sampling_rate = get_data(file)
        plot_time_domain_diagram(f_data, modscheme, sampling_rate)
        plot_frequency_domain_diagram(f_data, modscheme, center_frequencies, sampling_rate)
        plot_constellation_diagram(f_data, modscheme)
        plot_spectrogram(f_data, modscheme, sampling_rate)

if __name__ == "__main__":
    main()
