## python imports
import numpy as np
import os
from datetime import datetime

from .maps import mod_str2int


def check_range(x, positive=True):
    start, stop, step = x
    if positive:
        assert all(i >= 0.0 for i in x)
    return (stop >= start) and (step <= (stop - start))


def map_config(config, defaults, dataset_dir):
    mapped = {}

    ## num samples
    if "n_samps" in config.keys():
        assert isinstance(config["n_samps"], int), "n_samps must be an integer."
        assert config["n_samps"] > 0, "n_samps must be greater than zero."
        mapped["n_samps"] = config["n_samps"]
    else:
        print("No n_samps value provided. Using defaults.")
        mapped["n_samps"] = defaults["n_samps"]

    ## num captures
    if "n_captures" in config.keys():
        assert isinstance(config["n_captures"], int), "n_capures must be an integer."
        assert config["n_captures"] > 0, "n_captures must be greater than zero."
        mapped["n_captures"] = config["n_captures"]
    else:
        print("No n_captures value provided. Using defaults.")
        mapped["n_captures"] = defaults["n_captures"]

    ## mods
    if "modulation" in config.keys():
        try:
            mapped["modulation"] = []
            for i in config["modulation"]:
                mapped["modulation"].append(mod_str2int[i])
        except ValueError:
            print("Invalid modulation scheme found.")
    else:
        print("No modulations value provided. Using defaults.")
        mapped["modulation"] = []
        for i in defaults["modulation"]:
            mapped["modulation"].append(mod_str2int[i])

    ## symbol rate
    if "symbol_rate" in config.keys():
        if isinstance(config["symbol_rate"], list):
            for i in config["symbol_rate"]:
                assert isinstance(i, int), "symbol_rate must be an integer."
                assert i > 0, "symbol_rate must be greater than zero."
            mapped["symbol_rate"] = config["symbol_rate"]
        elif isinstance(config["symbol_rate"], int):
            assert config["symbol_rate"] > 0, "symbol_rate must be greater than zero."
            mapped["symbol_rate"] = [config["symbol_rate"]]
        else:
            raise ValueError("Invalid symbol rate type.")
    else:
        print("No symbol rate provided. Using defaults.")
        mapped["symbol_rate"] = defaults["symbol_rate"]

    ## Sampling rate (for wideband signals)
    if "sampling_rate" in config.keys():
        assert isinstance(
            config["sampling_rate"], (int, float)
        ), "sampling_rate must be a number."
        assert config["sampling_rate"] > 0, "sampling_rate must be positive."
        mapped["sampling_rate"] = config["sampling_rate"]
    else:
        print("No sampling rate provided. Using defaults.")
        mapped["sampling_rate"] = defaults.get(
            "sampling_rate", 20e6
        )  # Default to 2 MHz if not provided.

    # Randomly generated center frequencies
    if "randomly_generated_center_frequencies" in config.keys():
        # Assert "center_frequencies" is not also provided
        assert (
            "center_frequencies" not in config.keys()
        ), "center_frequencies and randomly_generated_center_frequencies cannot both be provided."
        assert isinstance(
            config["randomly_generated_center_frequencies"], list
        ), "randomly_generated_center_frequencies must be a list."
        mapped["center_frequencies"] = config["randomly_generated_center_frequencies"]
        mapped["center_frequencies_random"] = True
    ## Center frequencies (for wideband signals)
    elif "center_frequencies" in config.keys():
        assert isinstance(
            config["center_frequencies"], list
        ), "center_frequencies must be a list."
        assert all(
            isinstance(f, (int, float)) and f > 0 for f in config["center_frequencies"]
        ), "All center frequencies must be positive numbers."
        mapped["center_frequencies_random"] = False
        mapped["center_frequencies"] = config["center_frequencies"]
    else:
        print("No center frequencies provided. Using defaults.")
        mapped["center_frequencies"] = defaults.get(
            "center_frequencies", [mapped["sampling_rate"] / 4]
        )  # Default to half Nyquist frequency.
        mapped["center_frequencies_random"] = False

    ## If center frequencies list is empty, default to a single frequency at half Nyquist
    if not mapped["center_frequencies"]:
        mapped["center_frequencies"] = [mapped["sampling_rate"] / 2]

    ## Validate that the total bandwidth does not exceed the sampling rate
    # max_bandwidth = max(mapped["center_frequencies"]) - min(mapped["center_frequencies"])
    # assert max_bandwidth < mapped["sampling_rate"], "Total bandwidth exceeds sampling rate. Adjust center frequencies."

    ## AM/FM parameters
    mapped["am_defaults"] = {}
    if "am_defaults" in config.keys():
        if "modulation_index" in config["am_defaults"].keys():
            tmp = config["am_defaults"]["modulation_index"]
        else:
            print("No AM modulation index provided. Using defaults.")
            tmp = defaults["am_defaults"]["modulation_index"]
    else:
        print("No AM defaults provided. Using defaults.")
        tmp = defaults["am_defaults"]["modulation_index"]
    if isinstance(tmp, list):
        assert check_range(tmp)
        mapped["am_defaults"]["modulation_index"] = np.arange(
            tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
        )
    elif isinstance(tmp, float) and tmp > 0.0:
        mapped["am_defaults"]["modulation_index"] = [tmp]
    else:
        raise ValueError("Invalid AM modulation index.")

    mapped["fmnb_defaults"] = {}
    if "fmnb_defaults" in config.keys():
        if "modulation_factor" in config["fmnb_defaults"].keys():
            tmp = config["fmnb_defaults"]["modulation_factor"]
        else:
            print("No FM-NB modulation factor provided. Using defaults.")
            tmp = defaults["fmnb_defaults"]["modulation_factor"]
    else:
        print("No FM-NB defaults provided. Using defaults.")
        tmp = defaults["fmnb_defaults"]["modulation_factor"]
    if isinstance(tmp, list):
        assert check_range(tmp)
        mapped["fmnb_defaults"]["modulation_factor"] = np.arange(
            tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
        )
    elif isinstance(tmp, float) and tmp > 0.0:
        mapped["fmnb_defaults"]["modulation_factor"] = [tmp]
    else:
        raise ValueError("Invalid FM-NB modulation factor.")

    mapped["fmwb_defaults"] = {}
    if "fmwb_defaults" in config.keys():
        if "modulation_factor" in config["fmwb_defaults"].keys():
            tmp = config["fmwb_defaults"]["modulation_factor"]
        else:
            print("No FM-WB modulation factor provided. Using defaults.")
            tmp = defaults["fmwb_defaults"]["modulation_factor"]
    else:
        print("No FM-WB defaults provided. Using defaults.")
        tmp = defaults["fmwb_defaults"]["modulation_factor"]
    if isinstance(tmp, list):
        assert check_range(tmp)
        mapped["fmwb_defaults"]["modulation_factor"] = np.arange(
            tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
        )
    elif isinstance(tmp, float) and tmp > 0.0:
        mapped["fmwb_defaults"]["modulation_factor"] = [tmp]
    else:
        raise ValueError("Invalid FM-WB modulation factor.")

    ## filters
    if "filter" in config.keys():
        for i, f in enumerate(config["filter"]):
            if f["type"] in ["rrc", "gaussian"]:
                filter_type = f["type"] + "_filter"
                mapped[filter_type] = {}

                d = defaults["filter"]
                tmp = [i["type"] == f["type"] for i in d]
                d_i = np.where(tmp)[0][0]

                if "beta" in f.keys():
                    tmp = f["beta"]
                else:
                    print("No filter beta provided. Using defaults.")
                    tmp = d[d_i]["beta"]
                if isinstance(tmp, list):
                    assert check_range(tmp)
                    mapped[filter_type]["beta"] = np.arange(
                        tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
                    )
                elif isinstance(tmp, float) and tmp > 0.0:
                    mapped[filter_type]["beta"] = [tmp]
                else:
                    raise ValueError("Invalid filter beta.")

                if "dt" in f.keys():
                    tmp = f["dt"]
                else:
                    print("No filter dt provided. Using defaults.")
                    tmp = d[d_i]["dt"]
                if isinstance(tmp, list):
                    assert check_range(tmp)
                    mapped[filter_type]["dt"] = np.arange(
                        tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
                    )
                elif isinstance(tmp, float) and tmp >= 0.0:
                    mapped[filter_type]["dt"] = [tmp]
                else:
                    raise ValueError("Invalid filter dt.")

                if "delay" in f.keys():
                    tmp = f["delay"]
                else:
                    print("No filter delay provided. Using defaults")
                    tmp = d[d_i]["delay"]
                if isinstance(tmp, list):
                    assert check_range(tmp)
                    mapped[filter_type]["delay"] = np.arange(
                        tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
                    )
                elif isinstance(tmp, int) and tmp >= 0:
                    mapped[filter_type]["delay"] = [tmp]
                else:
                    raise ValueError("Invalid filter delay.")
            else:
                raise ValueError("Invalid filter type.")
    else:
        print("No filter parameters provided. Using defaults.")
        mapped["rrc_filter"] = {}
        tmp = defaults["filter"][0]["beta"]
        mapped["rrc_filter"]["beta"] = np.arange(
            tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
        )
        tmp = defaults["filter"][0]["dt"]
        mapped["rrc_filter"]["dt"] = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
        tmp = defaults["filter"][0]["delay"]
        mapped["rrc_filter"]["delay"] = np.arange(
            tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
        )
        mapped["gaussian_filter"] = {}
        tmp = defaults["filter"][1]["beta"]
        mapped["gaussian_filter"]["beta"] = np.arange(
            tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
        )
        tmp = defaults["filter"][1]["dt"]
        mapped["gaussian_filter"]["dt"] = np.arange(
            tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
        )
        tmp = defaults["filter"][1]["delay"]
        mapped["gaussian_filter"]["delay"] = np.arange(
            tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
        )

    ## channel
    if "channel" in config.keys():
        if config["channel"]["type"] == "awgn":
            mapped["channel_type"] = "awgn"

            if "snr" in config["channel"].keys():
                tmp = config["channel"]["snr"]
            else:
                print("No channel SNR parameters provided. Using defaults.")
                tmp = defaults["channel"]["snr"]
            if isinstance(tmp, list):
                assert check_range(tmp, positive=False)
                snr_list = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
            elif isinstance(tmp, int):
                snr_list = [tmp]
            else:
                raise ValueError("Invalid SNR range.")

            if "fo" in config["channel"].keys():
                tmp = config["channel"]["fo"]
            else:
                tmp = defaults["channel"]["fo"]
            if isinstance(tmp, list):
                assert check_range(tmp, positive=False)
                fo_list = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
            elif isinstance(tmp, float) and tmp >= 0.0:
                fo_list = [tmp]
            else:
                raise ValueError("Invalid FO range.")
            fo_list = [_fo * np.pi for _fo in fo_list]

            if "po" in config["channel"].keys():
                tmp = config["channel"]["po"]
            else:
                tmp = defaults["channel"]["po"]
            if isinstance(tmp, list):
                assert check_range(tmp, positive=False)
                po_list = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
            elif isinstance(tmp, float) and tmp >= 0.0:
                po_list = [tmp]
            else:
                raise ValueError("Invalid FO range.")

            mapped["channel_params"] = [
                (snr, fo, po) for snr in snr_list for fo in fo_list for po in po_list
            ]

        elif config["channel"]["type"] == "rayleigh":
            mapped["channel_type"] = "rayleigh"

            # AWGN parameter
            awgn_enabled = config["channel"].get(
                "awgn", defaults["channel"].get("awgn", True)
            )
            awgn_flag = 1 if awgn_enabled else 0

            # SNR range
            if "snr" in config["channel"].keys():
                tmp = config["channel"]["snr"]
            else:
                print("No channel SNR parameters provided. Using defaults.")
                tmp = defaults["channel"]["snr"]
            if isinstance(tmp, list):
                assert check_range(tmp, positive=False)
                snr_list = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
            elif isinstance(tmp, int):
                snr_list = [tmp]
            else:
                raise ValueError("Invalid SNR range.")

            # FO and PO ranges
            if "fo" in config["channel"].keys():
                tmp = config["channel"]["fo"]
            else:
                tmp = defaults["channel"]["fo"]
            if isinstance(tmp, list):
                assert check_range(tmp, positive=False)
                fo_list = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
            elif isinstance(tmp, float) and tmp >= 0.0:
                fo_list = [tmp]
            else:
                raise ValueError("Invalid FO range.")
            fo_list = [_fo * np.pi for _fo in fo_list]

            if "po" in config["channel"].keys():
                tmp = config["channel"]["po"]
            else:
                tmp = defaults["channel"]["po"]
            if isinstance(tmp, list):
                assert check_range(tmp, positive=False)
                po_list = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
            elif isinstance(tmp, float) and tmp >= 0.0:
                po_list = [tmp]
            else:
                raise ValueError("Invalid PO range.")

            # Path delays
            path_delays = config["channel"].get(
                "path_delays", defaults["channel"].get("path_delays", [0])
            )
            if not isinstance(path_delays, list) or not all(
                isinstance(d, (int, float)) for d in path_delays
            ):
                raise ValueError("Invalid path delays.")

            # Path gains
            path_gains = config["channel"].get(
                "path_gains", defaults["channel"].get("path_gains", [0])
            )
            if not isinstance(path_gains, list) or len(path_gains) != len(path_delays):
                raise ValueError("Invalid path gains.")

            # Combine parameters for Rayleigh
            mapped["channel_params"] = [
                (snr, fo, po, awgn_flag, path_delays, path_gains)
                for snr in snr_list
                for fo in fo_list
                for po in po_list
            ]

        elif config["channel"]["type"] == "rician":
            mapped["channel_type"] = "rician"

            # K-factor
            k_factor = config["channel"].get("k_factor", 10.0)
            if not isinstance(k_factor, (int, float)) or k_factor < 0.0:
                raise ValueError("Invalid K-factor.")

            # AWGN
            awgn_enabled = config["channel"].get(
                "awgn", defaults["channel"].get("awgn", True)
            )
            awgn_flag = 1 if awgn_enabled else 0

            # SNR range
            if "snr" in config["channel"].keys():
                tmp = config["channel"]["snr"]
            else:
                print("No channel SNR parameters provided. Using defaults.")
                tmp = defaults["channel"]["snr"]
            if isinstance(tmp, list):
                assert check_range(tmp, positive=False)
                snr_list = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
            elif isinstance(tmp, int):
                snr_list = [tmp]
            else:
                raise ValueError("Invalid SNR range.")

            # FO and PO ranges
            if "fo" in config["channel"].keys():
                tmp = config["channel"]["fo"]
            else:
                tmp = defaults["channel"]["fo"]
            if isinstance(tmp, list):
                assert check_range(tmp, positive=False)
                fo_list = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
            elif isinstance(tmp, float) and tmp >= 0.0:
                fo_list = [tmp]
            else:
                raise ValueError("Invalid FO range.")
            fo_list = [_fo * np.pi for _fo in fo_list]

            if "po" in config["channel"].keys():
                tmp = config["channel"]["po"]
            else:
                tmp = defaults["channel"]["po"]
            if isinstance(tmp, list):
                assert check_range(tmp, positive=False)
                po_list = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
            elif isinstance(tmp, float) and tmp >= 0.0:
                po_list = [tmp]
            else:
                raise ValueError("Invalid PO range.")

            # Path delays
            path_delays = config["channel"].get(
                "path_delays", defaults["channel"].get("path_delays", [0])
            )
            if not isinstance(path_delays, list) or not all(
                isinstance(d, (int, float)) for d in path_delays
            ):
                raise ValueError("Invalid path delays.")

            # Path gains
            path_gains = config["channel"].get(
                "path_gains", defaults["channel"].get("path_gains", [0])
            )
            if not isinstance(path_gains, list) or len(path_gains) != len(path_delays):
                raise ValueError("Invalid path gains.")

            # Combine parameters for Rician
            mapped["channel_params"] = [
                (snr, fo, po, k_factor, awgn_flag, path_delays, path_gains)
                for snr in snr_list
                for fo in fo_list
                for po in po_list
            ]
        else:
            raise ValueError("Invalid channel type.")
    else:
        print("No channel parameters provided. Using defaults.")
        mapped["channel_type"] = "awgn"
        tmp = defaults["channel"]["snr"]
        if isinstance(tmp, list):
            assert check_range(tmp, positive=False)
            snr_list = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
        elif isinstance(tmp, int):
            snr_list = [tmp]
        tmp = defaults["channel"]["fo"]
        if isinstance(tmp, list):
            assert check_range(tmp, positive=False)
            fo_list = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
        elif isinstance(tmp, float) and tmp >= 0.0:
            fo_list = [tmp]
        fo_list = [_fo * np.pi for _fo in fo_list]
        tmp = defaults["channel"]["po"]
        if isinstance(tmp, list):
            assert check_range(tmp, positive=False)
            po_list = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
        elif isinstance(tmp, float) and tmp >= 0.0:
            po_list = [tmp]

        mapped["channel_params"] = [
            (snr, fo, po) for snr in snr_list for fo in fo_list for po in po_list
        ]

    ## savename
    if "savepath" in config.keys():
        tmp = dataset_dir + "/" + config["savepath"]
    else:
        print("No savename provided. Using defaults.")
        tmp = dataset_dir + "/" + defaults["savepath"]
    if os.path.exists(tmp):
        ## modify pathname with date (DD-MM-YY) and time (H-M-S)
        t = datetime.today()
        mapped["savepath"] = tmp + "_" + t.strftime("%d-%m-%y-%H-%M-%S")
    else:
        mapped["savepath"] = tmp
    os.makedirs(mapped["savepath"])
    mapped["savename"] = mapped["savepath"].split("/")[-1]

    ## verbosity
    if "verbose" in config.keys():
        if config["verbose"] in [0, 1]:
            mapped["verbose"] = config["verbose"]
        else:
            raise ValueError("Verbosity may only be 0 or 1")
    else:
        print("No verbosity parameter provided. Using defaults.")
        mapped["verbose"] = defaults["verbose"]

    ## archive
    if "archive" in config.keys():
        assert isinstance(config["archive"], bool), "archive must be a boolean."
        mapped["archive"] = config["archive"]
    else:
        print("No archive parameter provided. Using defaults.")
        mapped["archive"] = defaults["archive"]

    return mapped
