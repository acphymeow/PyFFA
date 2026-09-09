"""Compare corrected-kernel LSC-on 3-D tunes with nominal-60 theory."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

import comparison_core as analysis
from dispersion import centered_dispersion_curves
from longitudinal_driver import EMITTANCE_RMS
from plotting import plot_tune_comparison


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT as REPO, CANONICAL_TUNE
NEW = REPO / "HPC_dir" / "Newdata_coupling_withLsc100" / "FFT_Tune_Dump_0"
OLD_TUNES = CANONICAL_TUNE / "results" / "lsc_on" / "withlsc500_first100_3d_tunes.npz"
OUT = HERE / "results" / "new_lsc_on_nominal60"
OUTPUT = OUT / "new_3d_vs_nominal60_theory_lsc_on"
N_SAMPLES = 100
COMMON_STAR = (3.125, 2.685)


def load_new_3d(lattice):
    files = sorted(NEW.glob("particle_*_angle_90.0_rank_*.npz"))
    data = np.empty((N_SAMPLES, len(files), 5), dtype=np.float32)
    for index, path in enumerate(files):
        particle = np.load(path)["particles"]
        data[:, index] = particle[:, [0, 1, 2, 3, 5]]
        if (index + 1) % 5000 == 0:
            print(f"loaded {index + 1}/{len(files)}", flush=True)
    x, xp, z, zp, _ = analysis.project_3d_coordinates(data.astype(np.float64))
    qr_phase, qr_fft = analysis.both_estimators(
        x, xp, lattice["beta_r"][0], lattice["alpha_r"][0], lattice["qr"])
    qz_phase, qz_fft = analysis.both_estimators(
        z, zp, lattice["beta_z"][0], lattice["alpha_z"][0], lattice["qz"],
        fft_half_width=0.25)
    return qr_phase, qz_phase, qr_fft, qz_fft


def stats(qr, qz):
    return np.r_[qr.mean(), qr.std(), np.percentile(qr, (1, 5, 50, 95, 99)),
                 qz.mean(), qz.std(), np.percentile(qz, (1, 5, 50, 95, 99))]


def run_nominal60_theory(lattice):
    phi, dispersion_x, dispersion_xp = centered_dispersion_curves()
    analysis.OUT = OUT
    analysis.TOY_TRACK_TURNS = 110
    analysis.TUNE_START = 10
    analysis.N_TURNS = N_SAMPLES
    analysis.N_TOY = 2048
    analysis.TOY_STEPS_PER_CELL = 50
    analysis.TOY_USE_RESULT_CACHE = False
    analysis.TOY_SEED = None
    analysis.TOY_CASES = ("on",)
    analysis.TOY_PROGRESS_CHUNKS = 10
    analysis.FFT_HALF_WIDTH_R = 0.25
    analysis.FFT_HALF_WIDTH_Z = 0.25

    analysis.TOY_SIGMA_R0 = None
    analysis.TOY_SIGMA_Z0 = None
    analysis.TOY_SIGMA_R_PROFILE = None
    analysis.TOY_SIGMA_R_PROFILE_IS_TOTAL = False

    analysis.TOY_FULL_DISPERSION = True
    analysis.TOY_DISPERSION_DX0_M_PER_MEV = float(np.interp(
        0.0, phi, dispersion_x, period=2.0*np.pi))
    analysis.TOY_DISPERSION_DXP0_RAD_PER_MEV = float(np.interp(
        0.0, phi, dispersion_xp, period=2.0*np.pi))
    analysis.TOY_DISPERSION_DX_PROFILE = np.interp(
        lattice["theta"], phi, dispersion_x, period=2.0*np.pi)
    analysis.TOY_DISPERSION_SIGMA_E_MEV = None
    analysis.TOY_COMMON_ORBIT_LAUNCH = True
    return analysis.run_toy_100(lattice)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lattice = analysis.build_lattice()
    new = load_new_3d(lattice)
    old = np.load(OLD_TUNES)
    theory = run_nominal60_theory(lattice)
    qr_theory, qz_theory = theory["qr_on_fft"], theory["qz_on_fft"]

    columns = ("Qr_mean", "Qr_std", "Qr_p01", "Qr_p05", "Qr_p50",
               "Qr_p95", "Qr_p99", "Qz_mean", "Qz_std", "Qz_p01",
               "Qz_p05", "Qz_p50", "Qz_p95", "Qz_p99")
    rows = np.vstack((
        stats(old["qr_fft"], old["qz_fft"]),
        stats(new[2], new[3]),
        stats(qr_theory, qz_theory),
    ))
    with (OUT / "fft_summary.csv").open(
        "w", encoding="utf-8"
    ) as stream:
        stream.write("case," + ",".join(columns) + "\n")
        for name, row in zip(("old_3d_100", "new_3d_100", "theory_nominal60_100"), rows):
            stream.write(name + "," + ",".join(f"{value:.9g}" for value in row) + "\n")

    sigma_r0 = np.sqrt(lattice["beta_r"][0] * EMITTANCE_RMS)
    sigma_z0 = np.sqrt(lattice["beta_z"][0] * EMITTANCE_RMS)
    np.savez_compressed(
        OUTPUT.with_suffix(".npz"),
        new_qr_phase=new[0], new_qz_phase=new[1],
        new_qr_fft=new[2], new_qz_fft=new[3],
        theory_qr_fft=qr_theory, theory_qz_fft=qz_theory,
        old_qr_fft=old["qr_fft"], old_qz_fft=old["qz_fft"],
        emittance_rms_m_rad=EMITTANCE_RMS,
        sigma_r0_m=sigma_r0, sigma_z0_m=sigma_z0,
        common_star=np.asarray(COMMON_STAR),
    )
    fig = plot_tune_comparison(
        new[2], new[3], qr_theory, qz_theory,
        endpoint_3d=COMMON_STAR, endpoint_toy=COMMON_STAR,
        qr_bare=lattice["qr"], qz_bare=lattice["qz"],
        xrange=(2.90, 3.30), yrange=(2.50, 2.90),
        title_3d="New 3-D PIC tracking (LSC on)",
        title_toy=r"Theory ($\epsilon_{r,z}=60\pi$ mm mrad, LSC on)",
        output_stem=OUTPUT,
        endpoint_label_3d="Common reference",
        endpoint_label_toy="Common reference",
    )
    plt.close(fig)
    print(f"Twiss sigma_r(0) = {sigma_r0*1e3:.9f} mm")
    print(f"Twiss sigma_z(0) = {sigma_z0*1e3:.9f} mm")
    for name, row in zip(("old 3-D", "new 3-D", "nominal-60 theory"), rows):
        print(name, dict(zip(columns, row)))
    print(f"saved: {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
