"""Compare corrected LSC-off 3-D tunes with a nominal 60-pi fixed toy."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import comparison_core as analysis
from dispersion import centered_dispersion_curves
from longitudinal_driver import EMITTANCE_RMS
from plotting import plot_tune_comparison


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "new_lsc_off_nominal60"
SOURCE = HERE / "results" / "new_lsc_off_kernel_fix" / "old_new_lsc_off_tunes.npz"
OUTPUT = OUT / "new_3d_vs_nominal60_toy_lsc_off"
N_TURNS = 90
COMMON_STAR = (3.024, 2.560)


def stats(qr, qz):
    return np.r_[qr.mean(), qr.std(), np.percentile(qr, (1, 5, 50, 95)),
                 qz.mean(), qz.std(), np.percentile(qz, (1, 5, 50, 95))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(SOURCE)
    lattice = analysis.build_lattice()
    phi, dispersion_x, dispersion_xp = centered_dispersion_curves()

    analysis.OUT = OUT
    analysis.TOY_TRACK_TURNS = 100
    analysis.TUNE_START = 10
    analysis.N_TURNS = N_TURNS
    analysis.N_TOY = 768
    analysis.TOY_STEPS_PER_CELL = 50
    analysis.TOY_USE_RESULT_CACHE = False
    analysis.TOY_SEED = None
    analysis.TOY_CASES = ("off",)
    analysis.TOY_PROGRESS_CHUNKS = 10
    analysis.FFT_HALF_WIDTH_R = 0.25
    analysis.FFT_HALF_WIDTH_Z = 0.32

    # Both planes start directly from the configured 60-pi RMS emittance.
    # No transverse emittance or sigma is measured from the 3-D solution.
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

    toy = analysis.run_toy_100(lattice)
    qr_3d, qz_3d = data["new_qr_fft"], data["new_qz_fft"]
    qr_toy, qz_toy = toy["qr_off_fft"], toy["qz_off_fft"]
    sigma_r0 = np.sqrt(lattice["beta_r"][0] * EMITTANCE_RMS)
    sigma_z0 = np.sqrt(lattice["beta_z"][0] * EMITTANCE_RMS)

    columns = ("Qr_mean", "Qr_std", "Qr_p01", "Qr_p05", "Qr_p50",
               "Qr_p95", "Qz_mean", "Qz_std", "Qz_p01", "Qz_p05",
               "Qz_p50", "Qz_p95")
    rows = np.vstack((stats(qr_3d, qz_3d), stats(qr_toy, qz_toy)))
    with (OUT / "fft_summary.csv").open(
        "w", encoding="utf-8"
    ) as stream:
        stream.write("case," + ",".join(columns) + "\n")
        for name, row in zip(("new_3d_90", "toy_nominal60_90"), rows):
            stream.write(name + "," + ",".join(f"{x:.9g}" for x in row) + "\n")

    np.savez_compressed(
        OUTPUT.with_suffix(".npz"), qr_3d=qr_3d, qz_3d=qz_3d,
        qr_toy=qr_toy, qz_toy=qz_toy,
        emittance_rms_m_rad=EMITTANCE_RMS,
        sigma_r0_m=sigma_r0, sigma_z0_m=sigma_z0,
        common_star=np.asarray(COMMON_STAR),
    )
    fig = plot_tune_comparison(
        qr_3d, qz_3d, qr_toy, qz_toy,
        endpoint_3d=COMMON_STAR, endpoint_toy=COMMON_STAR,
        qr_bare=lattice["qr"], qz_bare=lattice["qz"],
        xrange=(2.90, 3.30), yrange=(2.50, 2.90),
        title_3d="New 3-D PIC tracking (LSC off)",
        title_toy=r"Fixed toy ($\epsilon_{r,z}=60\pi$ mm mrad)",
        output_stem=OUTPUT,
        endpoint_label_3d="Common reference",
        endpoint_label_toy="Common reference",
    )
    plt.close(fig)

    print(f"epsilon_rms = {EMITTANCE_RMS:.9g} m rad")
    print(f"Twiss sigma_r(0) = {sigma_r0*1e3:.9f} mm")
    print(f"Twiss sigma_z(0) = {sigma_z0*1e3:.9f} mm")
    for name, row in zip(("new 3-D", "nominal-60 toy"), rows):
        print(name, dict(zip(columns, row)))
    print(f"saved: {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
