#!/usr/bin/python3

import os
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

np.Inf = np.inf  # Fix for a bug in the numpy library

# Results can be compared to the IQEngine Signal Generator:
# https://iqengine.org/siggen

# TODO:
# - Generate signals on IQEngine Signal Generator and compare results
# - Test generating with different signal parameters (defaults.json)

#####################################################################
####################### MODIFIABLE VARIABLES ########################
#####################################################################

# Directories:
dataset_path = "../data/default"

time_domain_output_path = "../tests/figures/Time-Domain"
frequency_domain_output_path = "../tests/figures/Frequency-Domain"
constellation_diagram_output_path = "../tests/figures/Constellation-Diagrams"
spectrogram_output_path = "../tests/figures/Spectrograms"

# Time domain plotting parameters - time domain is too large to plot all at once
time_domain_length = 300
time_domain_start_index = 0

# Spectrogram plotting parameters:
spectrogram_fft_size = 1024  # Size of the FFT
spectrogram_sample_rate = 1e6  # Sample rate of the signal

#####################################################################
#################### END OF MODIFIABLE VARIABLES ####################
#####################################################################


def delete_existing_plots():
    # Empty out the Time-Domain diagram directory
    for file in os.listdir(time_domain_output_path):
        os.remove(f"{time_domain_output_path}/{file}")

    # Empty out the Frequency-Domain diagram directory
    for file in os.listdir(frequency_domain_output_path):
        os.remove(f"{frequency_domain_output_path}/{file}")

    # Empty out the Constellation diagram directory
    for file in os.listdir(constellation_diagram_output_path):
        os.remove(f"{constellation_diagram_output_path}/{file}")

    # Empty out the Spectrogram directory
    for file in os.listdir(spectrogram_output_path):
        os.remove(f"{spectrogram_output_path}/{file}")


def get_data(file):
    ## get meta
    with open(file + ".sigmf-meta") as _f:
        f_meta = json.load(_f)
    f_meta = f_meta["annotations"][0]

    ## get data
    with open(file + ".sigmf-data", "rb") as _f:
        f_data = np.load(_f)

    modscheme = f_meta["rfml_labels"]["modclass"]
    return f_data, modscheme


def plot_time_domain_diagram(f_data, modscheme):
    I = f_data[
        time_domain_start_index
        * 2 : (time_domain_length + time_domain_start_index)
        * 2 : 2
    ]
    Q = f_data[
        (time_domain_start_index * 2)
        + 1 : ((time_domain_length + time_domain_start_index) * 2)
        + 1 : 2
    ]

    # Plot
    plt.figure()
    plt.plot(I, label="I")
    plt.plot(Q, label="Q")
    plt.grid()
    plt.title(f"Time Domain Diagram ({modscheme})")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [V]")
    plt.legend()
    plt.savefig(f"{time_domain_output_path}/{modscheme}.png")
    plt.close()


def plot_frequency_domain_diagram(f_data, modscheme):
    I = f_data[0::2]
    Q = f_data[1::2]

    # combine the real (I) and imaginary (Q) parts into a complex signal
    x = I + 1j * Q

    # Fourier Transform
    X = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2)
    f = np.linspace(-0.5, 0.5, len(X))

    # Plot
    plt.figure()
    plt.plot(f, X)
    plt.grid()
    plt.title(f"Frequency Domain Diagram ({modscheme})")
    plt.xlabel("Frequency [Hz Normalized]")
    plt.ylabel("PSD [dB]")
    plt.savefig(f"{frequency_domain_output_path}/{modscheme}.png")
    plt.close()


def plot_constellation_diagram(f_data, modscheme):
    I = f_data[0::2]
    Q = f_data[1::2]

    # Density-based coloring
    xy = np.vstack([I, Q])
    z = gaussian_kde(xy)(xy)  # Compute density
    idx = z.argsort()  # Sort points by density for better visualization
    I, Q, z = I[idx], Q[idx], z[idx]

    # Plot
    plt.figure()
    plt.title(f"Constellation Diagram ({modscheme})")
    plt.scatter(I, Q, c=z, s=1, cmap="viridis")  # Density-colored scatter plot
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")  # Horizontal axis
    plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")  # Vertical axis
    plt.xlabel("I (In-phase)")
    plt.ylabel("Q (Quadrature)")
    plt.grid(True, linestyle="--", alpha=0.5)
    cbar = plt.colorbar()  # Add colorbar to explain density
    cbar.set_label("Density")
    plt.savefig(f"{constellation_diagram_output_path}/{modscheme}.png")
    plt.close()


def plot_spectrogram(f_data, modscheme):
    I = f_data[0::2]
    Q = f_data[1::2]

    # combine the real (I) and imaginary (Q) parts into a complex signal
    x = I + 1j * Q

    num_rows = int(np.floor(len(x) / spectrogram_fft_size))
    spectrogram = np.zeros((num_rows, spectrogram_fft_size))
    for i in range(num_rows):
        spectrogram[i, :] = 10 * np.log10(
            np.abs(
                np.fft.fftshift(
                    np.fft.fft(
                        x[i * spectrogram_fft_size : (i + 1) * spectrogram_fft_size]
                    )
                )
            )
            ** 2
        )

    # Plot
    plt.figure()
    plt.imshow(
        spectrogram,
        aspect="auto",
        extent=[
            spectrogram_sample_rate / -2,
            spectrogram_sample_rate / 2,
            0,
            len(x) / spectrogram_sample_rate,
        ],
    )
    plt.title(f"Spectrogram ({modscheme})")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Time [s]")
    plt.savefig(f"{spectrogram_output_path}/{modscheme}.png")
    plt.close()


def main():
    delete_existing_plots()

    # Get the list of files
    files = os.listdir(os.path.abspath(dataset_path))
    files = [os.path.join(dataset_path, f.split(".")[0]) for f in files]
    files = list(set(files))

    for file in files:
        f_data, modscheme = get_data(file)
        plot_time_domain_diagram(f_data, modscheme)
        plot_frequency_domain_diagram(f_data, modscheme)
        plot_constellation_diagram(f_data, modscheme)
        plot_spectrogram(f_data, modscheme)


if __name__ == "__main__":
    main()
