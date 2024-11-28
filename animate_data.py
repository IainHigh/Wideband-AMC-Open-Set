import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
np.Inf = np.inf # Fix for a bug in the numpy library

# Function to process a single signal file
def process_signal(meta_file, data_file):
    # Load metadata
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Extract datatype from metadata
    datatype_map = {
        "cf32_le": np.complex64,  # Complex float
        "i16_le": np.int16,       # Integer
        # Add other mappings if needed
    }
    datatype = metadata["global"].get("core:datatype", "cf32_le")
    sample_type = datatype_map.get(datatype, np.complex64)
    
    # Read signal data
    with open(data_file, "rb") as f:
        signal = np.fromfile(f, dtype=sample_type)

    # Extract sample rate
    sample_rate = metadata["global"].get("core:sample_rate", 1e6)  # Default to 1 MHz if not specified
    signal = signal / np.max(np.abs(signal))
    return signal, sample_rate

# Function to plot a spectrogram
def plot_spectrogram(signal, sample_rate, title):
    f, t, Sxx = spectrogram(signal, fs=sample_rate, nperseg=256, return_onesided=False)
    Sxx = Sxx / np.max(Sxx)  # Normalize PSD
    f = np.abs(f)  # Convert to positive frequencies.
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
    plt.colorbar(label="Power Spectral Density (dB)")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    os.makedirs("plots", exist_ok=True)  # Ensure the 'plots' directory exists
    plt.savefig(f"plots/{title.replace(' ', '_')}.png")
    plt.close()  # Explicitly close the figure

# Process all signal files in the directory
def process_all_signals(base_dir):
    for subdir, _, files in os.walk(base_dir):
        # Filter for .sigmf-meta files
        meta_files = [f for f in files if f.endswith(".sigmf-meta")]
        
        for meta_file in meta_files:
            # Build paths for metadata and data files
            meta_path = os.path.join(subdir, meta_file)
            data_file = meta_file.replace(".sigmf-meta", ".sigmf-data")
            data_path = os.path.join(subdir, data_file)

            if os.path.exists(data_path):
                # Process and plot the signal
                signal, sample_rate = process_signal(meta_path, data_path)
                title = f"Spectrogram for {meta_file}"
                plot_spectrogram(signal, sample_rate, title)
            else:
                print(f"Data file {data_path} not found. Skipping.")

# Base directory containing data
base_directory = "data"

# Run the processing function
process_all_signals(base_directory)
