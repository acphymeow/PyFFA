from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Fill paths here. Each file should have columns r, fi, bz.
INPUT_FILES = [
    "./reverse_R_F_offset_grid_output/k7p63_B1p65_R265_Fp0p75_s49p97_offm30mm_r_fi_bz.csv",
]


def load_r_fi_bz(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    r = np.asarray(data["r"], dtype=float)
    fi = np.asarray(data["fi"], dtype=float)
    bz = np.asarray(data["bz"], dtype=float)
    ok = np.isfinite(r) & np.isfinite(fi) & np.isfinite(bz)
    r, fi, bz = r[ok], fi[ok], bz[ok]
    x = r * np.cos(fi)
    y = r * np.sin(fi)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    return s, bz


def main():
    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    for file_name in INPUT_FILES:
        path = Path(file_name)
        s, bz = load_r_fi_bz(path)
        ax.plot(s, bz, lw=1.5, label=path.stem)
    ax.axhline(1.0, color="0.3", lw=0.9, ls="--", label="1 T")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("Bz [T]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    plt.show()


if __name__ == "__main__":
    main()
