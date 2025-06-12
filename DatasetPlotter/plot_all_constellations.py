#!/usr/bin/python3
import ctypes
import importlib.util
import os
import numpy as np
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("maps", os.path.join("utils", "maps.py"))
maps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maps)
mod_str2int = maps.mod_str2int

# load shared library
clinear = ctypes.CDLL(os.path.abspath("./cmodules/linear_modulate"))

# set argument types for the new function
clinear.linear_constellation.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]


def plot_constellation(mod_scheme, output):
    if mod_scheme not in mod_str2int:
        raise ValueError(f"Unknown modulation scheme '{mod_scheme}'")

    modinfo = mod_str2int[mod_scheme]
    modtype = ctypes.c_int(modinfo[0])
    order = ctypes.c_int(modinfo[1])

    i_arr = (ctypes.c_float * order.value)()
    q_arr = (ctypes.c_float * order.value)()

    clinear.linear_constellation(modtype, order, i_arr, q_arr)

    I = np.frombuffer(i_arr, dtype=np.float32)
    Q = np.frombuffer(q_arr, dtype=np.float32)

    # Normalise to [-1, 1] range
    if np.max(np.abs(I)) != 0:
        I = I / np.max(np.abs(I))
    if np.max(np.abs(Q)) != 0:
        Q = Q / np.max(np.abs(Q))

    # Plot the constellation diagram
    plt.figure()
    plt.scatter(I, Q)
    plt.axhline(0, color="gray", ls="--", lw=0.5)
    plt.axvline(0, color="gray", ls="--", lw=0.5)
    plt.gca().set_aspect("equal", "box")
    plt.title(mod_scheme)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output, dpi=900)
    plt.close()


def main():
    mod_schemes = [
        "bpsk",
        "qpsk",
        "8psk",
        "16psk",
        "32psk",
        "64psk",
        "128psk",
        "256psk",
        "4qam",
        "8qam",
        "16qam",
        "32qam",
        "64qam",
        "128qam",
        "256qam",
        "2ask",
        "4ask",
        "8ask",
        "16ask",
        "32ask",
        "64ask",
        "128ask",
        "256ask",
        "2apsk",
        "4apsk",
        "8apsk",
        "16apsk",
        "32apsk",
        "64apsk",
        "128apsk",
        "256apsk",
        "2dpsk",
        "4dpsk",
        "8dpsk",
        "16dpsk",
        "32dpsk",
        "64dpsk",
        "128dpsk",
        "256dpsk",
    ]
    output_dir = "./DatasetPlotter/figures/all_mod_schemes"
    os.makedirs(output_dir, exist_ok=True)

    # Empty the output directory
    for file in os.listdir(output_dir):
        os.remove(f"{output_dir}/{file}")

    for scheme in mod_schemes:
        output_name = os.path.join(output_dir, f"{scheme.lower()}_constellation.png")
        plot_constellation(scheme.lower(), output_name)


if __name__ == "__main__":
    main()
