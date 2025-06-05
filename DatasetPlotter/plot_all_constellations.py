#!/usr/bin/python3
"""
plot_unique_constellations.py
─────────────────────────────
Iterate through all SigMF captures in <Dataset_Directory>/<DATASET_NAME>,
create one constellation diagram per *new* modulation scheme, and save it.
"""

import os, sys, json, numpy as np, matplotlib.pyplot as plt
from scipy.signal import filtfilt, firwin
from scipy.stats import gaussian_kde
from tqdm import tqdm

np.Inf = np.inf

DATASET_NAME = "all_mod_schemes"  # dataset name in Dataset_Directory
OUTPUT_DIR = "./DatasetPlotter/figures/all_mod_schemes"
MAX_POINTS = 8192


def prepare_output_dir(path: str):
    os.makedirs(path, exist_ok=True)
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))


def bandpass_complex(x, fs, low, high, numtaps=101, beta=8.6):
    nyq = 0.5 * fs
    taps = firwin(
        numtaps,
        [max(low, 1.0) / nyq, min(high, nyq - 1.0) / nyq],
        pass_zero=False,
        window=("kaiser", beta),
    )
    return filtfilt(taps, [1.0], x.real) + 1j * filtfilt(taps, [1.0], x.imag)


def plot_constellation(iq, outfile, title):
    iq = iq[:MAX_POINTS] if len(iq) > MAX_POINTS else iq
    I, Q = iq.real, iq.imag
    z = gaussian_kde(np.vstack((I, Q)))(np.vstack((I, Q)))
    order = z.argsort()
    I, Q, z = I[order], Q[order], z[order]
    plt.figure()
    plt.scatter(I, Q, c=z, s=1, cmap="viridis")
    plt.axhline(0, color="gray", ls="--", lw=0.5)
    plt.axvline(0, color="gray", ls="--", lw=0.5)
    plt.title(title)
    plt.xlabel("I (In-phase)")
    plt.ylabel("Q (Quadrature)")
    plt.colorbar(label="Density")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def get_data(base):
    with open(base + ".sigmf-meta") as f:
        meta = json.load(f)
    with open(base + ".sigmf-data", "rb") as f:
        raw = np.load(f)

    ann0 = meta["annotations"][0]
    mods = ann0["rfml_labels"]["modclass"]
    fs = ann0["sampling_rate"]
    fcs = ann0["center_frequencies"]

    try:  # pull pulse-shaping info if present
        filt = meta["annotations"][1]["filter"]
        sps, beta = filt["sps"], filt["rolloff"]
    except (KeyError, IndexError):
        sps = beta = None

    iq = raw[0::2] + 1j * raw[1::2]  # interleaved I,Q → complex
    return iq, mods, fcs, fs, sps, beta


def main():
    # load system parameters (paths, rng seed, etc.)
    with open("./configs/system_parameters.json") as f:
        sp = json.load(f)
    sys.path.append(sp["Working_Directory"])
    np.random.seed(sp["Random_Seed"])

    dataset_path = os.path.join(sp["Dataset_Directory"], DATASET_NAME)
    if not os.path.isdir(dataset_path):
        sys.exit(f"Dataset path '{dataset_path}' not found.")

    prepare_output_dir(OUTPUT_DIR)
    seen = set()

    bases = {
        os.path.join(dataset_path, f.split(".")[0])
        for f in os.listdir(dataset_path)
        if f.endswith(".sigmf-data")
    }

    for base in tqdm(sorted(bases), desc="Processing captures"):
        try:
            iq, mods, fcs, fs, sps, beta = get_data(base)
        except Exception as e:
            print(f"[WARN] skipping '{base}': {e}")
            continue
        sps = sps[0] if isinstance(sps, list) else sps  # handle single value or list
        bw = (fs / sps) * (1 + beta)

        for i, m in enumerate(mods):
            if m in seen:  # already have this modulation plotted
                continue
            fc = fcs[i] if i < len(fcs) else 0.0
            chan = bandpass_complex(iq, fs, fc - bw / 2, fc + bw / 2)
            t = np.arange(len(chan)) / fs
            bb = chan * np.exp(-1j * 2 * np.pi * fc * t)  # down-convert

            if sps:
                bb = bb[int(sps) // 2 :: int(sps)]  # sample once per symbol

            outfile = os.path.join(OUTPUT_DIR, f"{m}.png")
            plot_constellation(bb, outfile, f"{m}  (first capture)")
            seen.add(m)
            print(f"[INFO] wrote {outfile}")

    print(f"\nDone — generated {len(seen)} unique constellation diagrams.")


if __name__ == "__main__":
    main()
