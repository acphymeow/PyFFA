"""Compare the corrected LSC-off run with the old-kernel LSC-off run."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import comparison_core as analysis
from beam_size import measure_first_tune_record_densest_phase_slice_sigma_x
from dispersion import centered_dispersion_curves


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT as ROOT
OLD = ROOT / "HPC_dir" / "data_coupling_withoutLsc500" / "FFT_Tune_Dump_0"
NEW = ROOT / "HPC_dir" / "Newdata_coupling_withoutLsc100" / "FFT_Tune_Dump_0"
OUT = HERE / "results" / "new_lsc_off_kernel_fix"
N_TURNS = 90


def load_tunes(folder, lattice):
    files = sorted(folder.glob("particle_*_angle_90.0_rank_*.npz"))
    data = np.empty((N_TURNS, len(files), 5), dtype=np.float32)
    valid = 0
    for i, path in enumerate(files):
        particle = np.load(path)["particles"]
        if particle.shape[0] < N_TURNS:
            print(f"skip {path.name}: only {particle.shape[0]} records")
            continue
        if particle.shape[1] != 17:
            raise ValueError(f"unexpected dump shape: {path} {particle.shape}")
        data[:, valid] = particle[:N_TURNS, [0, 1, 2, 3, 5]]
        valid += 1
        if (i + 1) % 5000 == 0:
            print(f"{folder.parent.name}: loaded {i + 1}/{len(files)}", flush=True)
    data = data[:, :valid]
    x, xp, z, zp, _ = analysis.project_3d_coordinates(data.astype(np.float64))
    qr_phase, qr_fft = analysis.both_estimators(
        x, xp, lattice["beta_r"][0], lattice["alpha_r"][0], lattice["qr"]
    )
    qz_phase, qz_fft = analysis.both_estimators(
        z, zp, lattice["beta_z"][0], lattice["alpha_z"][0], lattice["qz"],
        fft_half_width=0.32,
    )
    return qr_phase, qz_phase, qr_fft, qz_fft


def metrics(qr, qz):
    p = (1, 5, 50, 95, 99)
    return np.r_[len(qr), qr.mean(), qr.std(), np.percentile(qr, p),
                 qz.mean(), qz.std(), np.percentile(qz, p)]


def run_matched_toy(lattice):
    measurement = measure_first_tune_record_densest_phase_slice_sigma_x(
        NEW, analysis.SEO
    )
    phi, dispersion_x, dispersion_xp = centered_dispersion_curves()
    dx_profile = np.interp(lattice["theta"], phi, dispersion_x,
                           period=2*np.pi)
    sigma_x_profile = measurement.sigma_x_m*np.sqrt(
        lattice["beta_r"]/lattice["beta_r"][0]
    )
    analysis.OUT = OUT
    analysis.TOY_TRACK_TURNS = 100
    analysis.TUNE_START = 10
    analysis.N_TURNS = N_TURNS
    analysis.N_TOY = 768
    analysis.TOY_STEPS_PER_CELL = 50
    analysis.TOY_USE_RESULT_CACHE = False
    analysis.TOY_CASES = ("off",)
    analysis.TOY_PROGRESS_CHUNKS = 10
    analysis.FFT_HALF_WIDTH_R = 0.25
    analysis.FFT_HALF_WIDTH_Z = 0.32
    analysis.TOY_SIGMA_R0 = None
    analysis.TOY_SIGMA_Z0 = None
    analysis.TOY_SIGMA_R_PROFILE = sigma_x_profile
    analysis.TOY_SIGMA_R_PROFILE_IS_TOTAL = True
    analysis.TOY_DISPERSION_DX0_M_PER_MEV = float(np.interp(
        0.0, phi, dispersion_x, period=2*np.pi
    ))
    analysis.TOY_DISPERSION_DXP0_RAD_PER_MEV = float(np.interp(
        0.0, phi, dispersion_xp, period=2*np.pi
    ))
    analysis.TOY_DISPERSION_DX_PROFILE = dx_profile
    analysis.TOY_DISPERSION_SIGMA_E_MEV = None
    analysis.TOY_FULL_DISPERSION = True
    analysis.TOY_COMMON_ORBIT_LAUNCH = False
    toy = analysis.run_toy_100(lattice)
    return toy["qr_off_fft"], toy["qz_off_fft"], measurement.sigma_x_m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lattice = analysis.build_lattice()
    old = load_tunes(OLD, lattice)
    new = load_tunes(NEW, lattice)
    toy_qr, toy_qz, sigma_x_turn10 = run_matched_toy(lattice)
    toy = (toy_qr, toy_qz)

    names = ("N", "Qr_mean", "Qr_std", "Qr_p01", "Qr_p05", "Qr_p50",
             "Qr_p95", "Qr_p99", "Qz_mean", "Qz_std", "Qz_p01", "Qz_p05",
             "Qz_p50", "Qz_p95", "Qz_p99")
    rows = np.vstack((metrics(old[2], old[3]), metrics(new[2], new[3]),
                      metrics(*toy)))
    with (OUT / "fft_summary.csv").open("w", encoding="utf-8") as stream:
        stream.write("case," + ",".join(names) + "\n")
        for case, row in zip(("old_3d_90", "new_3d_90", "toy_matched_90"), rows):
            stream.write(case + "," + ",".join(f"{value:.9g}" for value in row) + "\n")

    np.savez_compressed(
        OUT / "old_new_lsc_off_tunes.npz",
        old_qr_phase=old[0], old_qz_phase=old[1],
        old_qr_fft=old[2], old_qz_fft=old[3],
        new_qr_phase=new[0], new_qz_phase=new[1],
        new_qr_fft=new[2], new_qz_fft=new[3],
        toy_qr_fft=toy[0], toy_qz_fft=toy[1],
        toy_sigma_x_turn10_m=sigma_x_turn10,
    )

    old_star = rows[0, [3, 10]]
    new_star = rows[1, [3, 10]]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    for ax, title, qr, qz in zip(
        axes, ("Old 3-D LSC off (stale kernel)", "New 3-D LSC off (updated kernel)"),
        (old[2], new[2]), (old[3], new[3]),
    ):
        ax.hist2d(qr, qz, bins=130, range=((2.9, 3.3), (2.5, 2.9)),
                  norm=LogNorm(), cmap="viridis")
        ax.scatter(*old_star, marker="x", s=55, linewidths=1.8,
                   color="orange", label="old 1% endpoint")
        ax.scatter(*new_star, marker="*", s=115, color="red", edgecolor="white",
                   linewidth=0.6, label="new 1% endpoint", zorder=5)
        ax.set(xlim=(2.9, 3.3), ylim=(2.5, 2.9), xlabel=r"$Q_r$", ylabel=r"$Q_z$",
               title=title)
        ax.legend(fontsize=8, loc="lower right")
    fig.savefig(OUT / "old_vs_new_lsc_off.png", dpi=250, bbox_inches="tight")
    fig.savefig(OUT / "old_vs_new_lsc_off.pdf", bbox_inches="tight")
    plt.close(fig)

    print("columns:", names)
    for case, row in zip(("old", "new", "matched toy"), rows):
        print(case, dict(zip(names, row)))


if __name__ == "__main__":
    main()
