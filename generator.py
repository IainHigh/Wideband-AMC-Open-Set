#!/usr/bin/python3

## python imports
import argparse
import sys
import json
import numpy as np
import os
import ctypes
from tqdm import tqdm
from utils import *

CALCULATE_BER_SNR = False  # Flag to determine if we should calculate the BER to SNR values for BPSK modulation scheme.

buf = 4096
halfbuf = 2048

rng_seed = None

# Read the configs/system_parameters.json file.
with open("./configs/system_parameters.json") as f:
    system_parameters = json.load(f)

working_directory = system_parameters["Working_Directory"]
sys.path.append(working_directory)

dataset_directory = system_parameters["Dataset_Directory"]

## load c modules
clinear = ctypes.CDLL(os.path.abspath("./cmodules/linear_modulate"))
cam = ctypes.CDLL(os.path.abspath("./cmodules/am_modulate"))
cfm = ctypes.CDLL(os.path.abspath("./cmodules/fm_modulate"))
cfsk = ctypes.CDLL(os.path.abspath("./cmodules/fsk_modulate"))
ctx = ctypes.CDLL(os.path.abspath("./cmodules/rrc_tx"))
cchan = ctypes.CDLL(os.path.abspath("./cmodules/channel"))

import numpy as np


def calculate_ber_BPSK(xI, xQ, yI, yQ, sps, trim):
    """
    Calculate the Bit Error Rate (BER) for BPSK.

    Parameters
    ----------
    xI, xQ : array_like
        Transmitted (baseband) in-phase and quadrature components.
    yI, yQ : array_like
        Received in-phase and quadrature components (after channel, frequency shift, etc.).
    sps : int
        Samples per symbol.
    trim : int
        Number of samples to trim from beginning and end.

    Returns
    -------
    ber : float
        Bit error rate.
    """
    # Convert inputs to numpy arrays
    xI = np.array(xI)
    xQ = np.array(xQ)
    yI = np.array(yI)
    yQ = np.array(yQ)

    # Trim the signals (to match what is saved/used)
    tx_I = xI[trim:-trim]
    tx_Q = xQ[trim:-trim]
    rx_I = yI[trim:-trim]
    rx_Q = yQ[trim:-trim]

    rx_complex = rx_I + 1j * rx_Q
    tx_complex = tx_I + 1j * tx_Q

    tx_symbols = tx_complex[::sps]
    rx_symbols = rx_complex[::sps]

    # Demap the symbols to bits (BPSK decision on the real part).
    tx_bits = (np.real(tx_symbols) >= 0).astype(int)
    rx_bits = (np.real(rx_symbols) >= 0).astype(int)

    # Demap the symbols to bits (BPSK decision on the real part).
    tx_bits = (np.real(tx_symbols) >= 0).astype(int)
    rx_bits = (np.real(rx_symbols) >= 0).astype(int)

    # # Calculate bit errors and BER.
    bit_errors = np.sum(tx_bits != rx_bits)
    total_bits = len(tx_bits)
    ber = bit_errors / total_bits

    return ber


def generate_linear(config):
    verbose = ctypes.c_int(config["verbose"])
    n_samps = config["n_samps"] + buf
    sampling_rate = config["sampling_rate"]

    # Generate time vector for mixing
    t = np.arange(n_samps - buf) / sampling_rate

    sig_params = [
        (_sps, _beta, _delay, _dt)
        for _sps in config["symbol_rate"]
        for _beta in config["rrc_filter"]["beta"]
        for _delay in config["rrc_filter"]["delay"]
        for _dt in config["rrc_filter"]["dt"]
    ]
    idx = np.random.choice(len(sig_params), config["n_captures"])
    sig_params = [sig_params[_idx] for _idx in idx]
    idx = np.random.choice(len(config["channel_params"]), config["n_captures"])
    channel_params = [config["channel_params"][_idx] for _idx in idx]
    channel_type = config["channel_type"]

    center_frequencies = config["center_frequencies"]

    ber_dict = {}
    mod_list = []
    for i in tqdm(range(0, config["n_captures"]), desc=f"Generating Data"):
        seed = ctypes.c_int(rng_seed + i)

        I_total = np.zeros(n_samps - buf, dtype=np.float32)
        Q_total = np.zeros(n_samps - buf, dtype=np.float32)

        for center_freq in center_frequencies:
            # Choose a random element from the modulation list
            index = np.random.randint(0, len(config["modulation"]) - 1)
            mod = config["modulation"][index]
            modtype = ctypes.c_int(mod[0])
            mod_list.append(mod[-1])

            # Extract channel parameters based on type
            if channel_type == "awgn":
                snr, fo, po = channel_params[i]
                snr = ctypes.c_float(snr)
                fo = ctypes.c_float(fo)
                po = ctypes.c_float(po)

            elif channel_type == "rayleigh":
                (
                    snr,
                    fo,
                    po,
                    awgn_flag,
                    path_delays,
                    path_gains,
                ) = channel_params[i]

                assert len(path_delays) == len(
                    path_gains
                ), "Path delays and path gains must have the same length."
                snr = ctypes.c_float(snr)
                fo = ctypes.c_float(fo)
                po = ctypes.c_float(po)
                num_taps = ctypes.c_int(len(path_delays))
                awgn = ctypes.c_int(awgn_flag)

                # Convert path_delays and path_gains to ctypes arrays
                path_delays_ctypes = (ctypes.c_float * len(path_delays))(*path_delays)
                path_gains_ctypes = (ctypes.c_float * len(path_gains))(*path_gains)

            elif channel_type == "rician":
                (
                    snr,
                    fo,
                    po,
                    k_factor,
                    awgn_flag,
                    path_delays,
                    path_gains,
                ) = channel_params[i]
                assert len(path_delays) == len(
                    path_gains
                ), "Path delays and path gains must have the same length."
                snr = ctypes.c_float(snr)
                fo = ctypes.c_float(fo)
                po = ctypes.c_float(po)
                k_factor = ctypes.c_float(k_factor)
                num_taps = ctypes.c_int(len(path_delays))
                awgn = ctypes.c_int(awgn_flag)

                # Convert path_delays and path_gains to ctypes arrays
                path_delays_ctypes = (ctypes.c_float * len(path_delays))(*path_delays)
                path_gains_ctypes = (ctypes.c_float * len(path_gains))(*path_gains)

            else:
                raise ValueError("Undefined channel type.")

            order = ctypes.c_int(mod[1])
            sps = ctypes.c_int(sig_params[i][0])
            beta = ctypes.c_float(sig_params[i][1])
            delay = ctypes.c_uint(int(sig_params[i][2]))
            dt = ctypes.c_float(sig_params[i][3])

            # Adjust n_sym for chunk processing
            n_sym = int(
                np.ceil(n_samps / sps.value)
            )  # Ensure the right number of symbols

            # Create return arrays
            s = (ctypes.c_uint * n_sym)(*np.zeros(n_sym, dtype=int))
            smI = (ctypes.c_float * n_sym)(*np.zeros(n_sym))
            smQ = (ctypes.c_float * n_sym)(*np.zeros(n_sym))
            xI = (ctypes.c_float * n_samps)(*np.zeros(n_samps))
            xQ = (ctypes.c_float * n_samps)(*np.zeros(n_samps))
            yI = (ctypes.c_float * n_samps)(*np.zeros(n_samps))
            yQ = (ctypes.c_float * n_samps)(*np.zeros(n_samps))

            # Call C modules for chunk processing
            clinear.linear_modulate(
                modtype, order, ctypes.c_int(n_sym), s, smI, smQ, verbose, seed
            )
            ctx.rrc_tx(
                ctypes.c_int(n_sym), sps, delay, beta, dt, smI, smQ, xI, xQ, verbose
            )

            # Channel Type
            if channel_type == "awgn":
                cchan.channel(snr, n_sym, sps, fo, po, xI, xQ, yI, yQ, verbose, seed)
            elif channel_type == "rayleigh":
                cchan.rayleigh_channel(
                    snr,
                    n_sym,
                    sps,
                    fo,
                    po,
                    num_taps,
                    awgn,
                    xI,
                    xQ,
                    yI,
                    yQ,
                    path_delays_ctypes,
                    path_gains_ctypes,
                    verbose,
                    seed,
                )
            elif channel_type == "rician":
                cchan.rician_channel(
                    snr,
                    n_sym,
                    sps,
                    fo,
                    po,
                    k_factor,
                    num_taps,
                    awgn,
                    xI,
                    xQ,
                    yI,
                    yQ,
                    path_delays_ctypes,
                    path_gains_ctypes,
                    verbose,
                    seed,
                )

            I = np.array(yI)[halfbuf:-halfbuf]
            Q = np.array(yQ)[halfbuf:-halfbuf]

            # Apply frequency shift for wideband signal
            freq_shift = np.exp(1j * 2 * np.pi * center_freq * t)

            I_shifted = I * np.real(freq_shift) - Q * np.imag(freq_shift)
            Q_shifted = I * np.imag(freq_shift) + Q * np.real(freq_shift)

            # Sum to create the wideband signal
            I_total += I_shifted
            Q_total += Q_shifted

            if (mod[-1] == "bpsk") and (CALCULATE_BER_SNR):
                ber = calculate_ber_BPSK(xI, xQ, yI, yQ, sps.value, trim=halfbuf)
                if snr.value not in ber_dict:
                    ber_dict[snr.value] = [ber]
                else:
                    ber_dict[snr.value].append(ber)

        # Normalize final signal
        # max_amp = max(np.max(np.abs(I_total)), np.max(np.abs(Q_total)))
        # if max_amp > 0:
        #    I_total /= max_amp
        #    Q_total /= max_amp

        # Metadata
        metadata = {
            "modname": mod_list,
            "order": order.value,
            "n_samps": n_samps - buf,
            "sampling_rate": config["sampling_rate"],
            "center_frequencies": center_frequencies,
            "channel_type": config["channel_type"],
            "snr": snr.value,
            "filter_type": "rrc",
            "sps": sps.value,
            "fo": fo.value,
            "po": po.value,
            "delay": delay.value,
            "beta": beta.value,
            "dt": dt.value,
            "savepath": config["savepath"],
            "savename": config["savename"],
        }

        # Save the concatenated data for this capture in SigMF format
        save_sigmf(I_total, Q_total, metadata, i)

    if CALCULATE_BER_SNR:
        # After processing all captures, calculate and print the average BER per SNR.
        avg_ber_dict = {
            snr: sum(ber_list) / len(ber_list) for snr, ber_list in ber_dict.items()
        }

        # Sort the dictionary by SNR
        avg_ber_dict = dict(sorted(avg_ber_dict.items()))

        if avg_ber_dict != {}:
            print("Average BER per SNR:")
            for snr, avg_ber in avg_ber_dict.items():
                print(f"SNR = {snr}: AVG_BER = {avg_ber}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to configuration file to use for data generation.",
    )
    parser.add_argument(
        "rng_seed",
        type=int,
        nargs="?",
        help="Random seed for data generation.",
    )
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)

    # If a rng seed is provided, use it. Otherwise, use the one from the config file.
    if args.rng_seed is not None:
        rng_seed = args.rng_seed
    else:
        rng_seed = system_parameters["Random_Seed"]
    np.random.seed(rng_seed)

    print("RNG Seed:", rng_seed)

    with open("./configs/defaults.json") as f:
        defaults = json.load(f)

    config = map_config(config, defaults, dataset_directory)

    ## Generate the data
    generate_linear(config)

    if config["archive"]:
        archive_sigmf(config["savepath"])
