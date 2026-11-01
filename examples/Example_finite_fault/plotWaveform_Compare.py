# ================================================================
#   ESS, Southern University of Science and Technology
#
#   File Name: plotWaveformCompare.py
#   Discription: Compare waveforms recorded by the stations between two
#                runs (WENO vs CGFDM).  CGFDM is drawn as a blue solid
#                line, WENO as a red dashed line.
#
#   Usage:
#       python plotWaveformCompare.py [WENO_DIR] [CGFDM_DIR]
#
#       WENO_DIR   default: ./data/WENO
#       CGFDM_DIR  default: ./data/CGFDM
#
#       Each folder must contain Vx_<station>.txt, Vy_<station>.txt and
#       Vz_<station>.txt, i.e. the files written by plotWaveform.py.
#
#   Reference:
#      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
#      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
#
# =================================================================*/

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import signal

WENO_STYLE = dict(color="red", linestyle="--", linewidth=1.0)
CGFDM_STYLE = dict(color="blue", linestyle="-", linewidth=1.0)


def load_trace(folder, component, key):
    fileName = os.path.join(folder, "%s_%s.txt" % (component, key))
    if not os.path.isfile(fileName):
        raise FileNotFoundError("missing waveform file: %s" % fileName)
    return np.loadtxt(fileName)


def main():
    parser = argparse.ArgumentParser(
        description="Compare station waveforms between WENO and CGFDM runs.")
    parser.add_argument("weno_dir", nargs="?", default="./data/WENO",
                        help="folder with WENO waveform txt files (default: ./data/WENO)")
    parser.add_argument("cgfdm_dir", nargs="?", default="./data/CGFDM",
                        help="folder with CGFDM waveform txt files (default: ./data/CGFDM)")
    parser.add_argument("--out", default="./img/Waveforms_compare.png",
                        help="output figure (default: ./img/Waveforms_compare.png)")
    parser.add_argument("--fl", type=float, default=1e-10,
                        help="high-pass corner frequency (Hz), default: 1e-10")
    parser.add_argument("--fh", type=float, default=2.0,
                        help="low-pass corner frequency (Hz), default: 2.0")
    parser.add_argument("--no-filter", action="store_true",
                        help="plot the raw traces without bandpass filtering")
    args = parser.parse_args()

    with open("params.json") as jsonsFile:
        params = json.load(jsonsFile)
    with open("station.json") as stationFile:
        station = json.load(stationFile)["station(point)"]

    station_keys = list(station.keys())
    station_num = len(station_keys)

    dt = params["DT"]
    tmax = params["TMAX"]

    b = a = None
    if not args.no_filter:
        wn1 = args.fl / (1.0 / dt / 2.0)
        wn2 = args.fh / (1.0 / dt / 2.0)
        print("bandpass: %.3e ~ %.3e (normalized)" % (wn1, wn2))
        b, a = signal.butter(2, [wn1, wn2], "bandpass")

    config = {"font.size": 12}
    rcParams.update(config)

    fig = plt.figure(figsize=(18, 12))
    fig.subplots_adjust(wspace=0.08)
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]

    components = [("Vx", "v_x"), ("Vy", "v_y"), ("Vz", "v_z")]
    disWid = 2
    legend_handles = None

    for row, key in enumerate(station_keys):
        offset = row * disWid

        for col, (component, title) in enumerate(components):
            weno = load_trace(args.weno_dir, component, key)
            cgfdm = load_trace(args.cgfdm_dir, component, key)

            num = min(weno.size, cgfdm.size)
            if weno.size != cgfdm.size:
                print("Warning: %s_%s length mismatch (WENO %d, CGFDM %d), "
                      "using the first %d samples"
                      % (component, key, weno.size, cgfdm.size, num))

            weno = weno[:num]
            cgfdm = cgfdm[:num]
            t = np.arange(num) * dt

            if b is not None:
                weno = signal.filtfilt(b, a, weno)
                cgfdm = signal.filtfilt(b, a, cgfdm)

            line_cgfdm, = axes[col].plot(t, cgfdm + offset,
                                         label="CGFDM", **CGFDM_STYLE)
            line_weno, = axes[col].plot(t, weno + offset,
                                        label="WENO", **WENO_STYLE)
            if legend_handles is None:
                # both styles are drawn in the same order, so the handles
                # from the first station represent all stations
                legend_handles = (line_cgfdm, line_weno)

    for col, (component, title) in enumerate(components):
        ax = axes[col]
        ax.set_xlim([0, tmax])
        ax.set_ylim(-1.5, -1.5 + 2 * station_num + 1)
        ax.set_xlabel("time(s)")
        ax.set_title(title)

    axes[0].set_yticks(np.arange(0, 2 * station_num, 2))
    axes[0].set_yticklabels(station_keys, fontsize=12)
    axes[0].set_ylabel("Amplitude (m/s)")
    for ax in axes[1:]:
        ax.set_yticks([])

    axes[0].legend(handles=list(legend_handles),
                   labels=["CGFDM", "WENO"], loc="upper right")

    outDir = os.path.dirname(args.out)
    if outDir:
        os.makedirs(outDir, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print("saved %s" % args.out)


if __name__ == "__main__":
    main()
