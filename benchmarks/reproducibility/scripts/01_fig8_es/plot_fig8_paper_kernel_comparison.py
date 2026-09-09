#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the Gaussian-only longitudinal-field benchmark for the paper.

The numerical curve is the final NS=256 result from
``fig8_gaussian_kernel_test.py``: the Gaussian source-slice-averaged kernel is
applied to the sampled line density after 8 cm Gaussian smoothing.  No
continuous model line density is used for the numerical curve.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import set_num_threads

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
os.chdir(REPO_ROOT)

from fig8_gaussian_kernel_test import gauss_kernel_1d_slice_avg
from plot_fig8_paper_uniform_disk import gaussian_smooth
from test_sc_260508_Es_formula import (
    A_SMOOTH,
    BUNCH_CENTER_FRAC,
    EK0_MEV,
    EPS0,
    N_PARTICLES,
    NX,
    NZ,
    Q_MACRO,
    SEED,
    SEO_PR_FILE,
    SEO_R_FILE,
    SIGMA_S,
    SIGMA_X,
    SIGMA_Z,
    S_CLIP_SIGMA,
    attach_seo_tables_to_bunch,
    axisymmetric_gaussian_longitudinal_field_integral,
    build_bunch_from_cyl,
    build_seo_s_matrix,
    convert_seo_pr_to_pr0_geom,
    generate_long_bunch_gaussian_s_on_closed_orbit,
    load_seo_pr_table,
    load_seo_r_table,
    prepare_fft_plan,
    unpack_solver_output,
)


# ---------------------------------------------------------------------------
# Figure parameters
# ---------------------------------------------------------------------------
NS_SOLVER = 256
SMOOTH_SIGMA = 0.08  # m, as in fig8_gaussian_kernel_test.py
SIGMA_PERP = float(np.sqrt(SIGMA_X * SIGMA_Z))
SLICE_AVERAGE_SUBSAMPLES = 64
SHOW_FIGURE = False

MANUSCRIPT_DIR = HERE / "outputs"
OUT_PNG = MANUSCRIPT_DIR / "benchmark_E_longitudinal_gaussian.png"
OUT_PDF = MANUSCRIPT_DIR / "benchmark_E_longitudinal_gaussian.pdf"
OUT_CSV = HERE / "fig8_gaussian_only_data.csv"
OUT_TXT = HERE / "fig8_gaussian_only_metrics.txt"


def main():
    set_num_threads(1)

    ek_axis, phi_axis, r_matrix = load_seo_r_table(SEO_R_FILE)
    pr_matrix = load_seo_pr_table(SEO_PR_FILE)
    pr0_matrix = convert_seo_pr_to_pr0_geom(pr_matrix)
    s_matrix, perimeter = build_seo_s_matrix(phi_axis, r_matrix, pr0_matrix)
    energy_index = int(np.argmin(np.abs(ek_axis - EK0_MEV)))

    coords_cyl, ek_array, _ = generate_long_bunch_gaussian_s_on_closed_orbit(
        seo_phi_axis=phi_axis,
        r0_phi=r_matrix[energy_index].copy(),
        pr0_phi=pr0_matrix[energy_index].copy(),
        s_phi=s_matrix[energy_index].copy(),
        perimeter=float(perimeter[energy_index]),
        n_particles=N_PARTICLES,
        bunch_center_frac=BUNCH_CENTER_FRAC,
        sigma_s=SIGMA_S,
        s_clip_sigma=S_CLIP_SIGMA,
        sigma_x=SIGMA_X,
        sigma_z=SIGMA_Z,
        ek0_mev=EK0_MEV,
        seed=SEED,
    )
    bunch = build_bunch_from_cyl(
        coords_cyl, ek_array, Q_MACRO, NX, NZ, NS_SOLVER
    )
    attach_seo_tables_to_bunch(
        bunch,
        ek_axis,
        phi_axis,
        r_matrix,
        pr_matrix,
        pr0_matrix,
        s_matrix,
        perimeter,
    )
    fft_forward, fft_inverse = prepare_fft_plan(NX, NZ, threads=1)
    solver_result = unpack_solver_output(
        bunch.Build_2_5D_SliceGrid_sbs_new_fixProjection(
            fft_forward, fft_inverse
        )
    )

    s_grid = np.asarray(solver_result["s_grid"], dtype=np.float64)
    slice_charge = np.asarray(solver_result["rho_long"], dtype=np.float64)
    total_charge = float(np.sum(slice_charge))
    weights = slice_charge / total_charge
    s_center = float(np.sum(s_grid * weights))
    sigma_s_measured = float(
        np.sqrt(np.sum(weights * (s_grid - s_center) ** 2))
    )
    s_relative = s_grid - s_center
    ds = float(s_grid[1] - s_grid[0])
    slice_charge_smooth = gaussian_smooth(
        slice_charge, s_grid, SMOOTH_SIGMA
    )
    separation = s_grid[:, None] - s_grid[None, :]
    field_gaussian = gauss_kernel_1d_slice_avg(
        separation,
        SIGMA_PERP,
        ds,
        n_int=SLICE_AVERAGE_SUBSAMPLES,
    ) @ slice_charge_smooth
    field_theory = axisymmetric_gaussian_longitudinal_field_integral(
        s_eval=s_grid,
        s_center=s_center,
        sigma_s=sigma_s_measured,
        sigma_perp=SIGMA_PERP,
        total_charge=total_charge,
        eps0=EPS0,
        n_tau=800,
    )

    peak_theory = float(np.max(np.abs(field_theory)))
    in_bunch = np.abs(s_relative) < 3.5 * sigma_s_measured

    def rms_percent(field):
        return float(
            np.sqrt(np.mean((field[in_bunch] - field_theory[in_bunch]) ** 2))
            / peak_theory
            * 100.0
        )

    def peak_percent(field):
        return float(
            (np.max(np.abs(field)) - peak_theory) / peak_theory * 100.0
        )

    metrics = (
        f"N={N_PARTICLES}\n"
        f"NS={NS_SOLVER}\n"
        f"slice_width_m={ds:.9f}\n"
        f"smoothing_sigma_m={SMOOTH_SIGMA:.6f}\n"
        "line_density=sampled_Gaussian_bunch\n"
        f"gaussian=source_slice_average_{SLICE_AVERAGE_SUBSAMPLES}\n"
        f"measured_sigma_s_m={sigma_s_measured:.9f}\n"
        f"gaussian_rms_percent_of_theory_peak={rms_percent(field_gaussian):.6f}\n"
        f"gaussian_peak_difference_percent={peak_percent(field_gaussian):.6f}\n"
    )
    OUT_TXT.write_text(metrics, encoding="utf-8")
    print(metrics, end="")

    np.savetxt(
        OUT_CSV,
        np.column_stack(
            (s_relative, field_theory, field_gaussian)
        ),
        delimiter=",",
        header=(
            "s_minus_sc_m,Es_3D_Gaussian_theory_V_per_m,"
            "Es_Gaussian_V_per_m"
        ),
        comments="",
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 18.0,
            "axes.labelsize": 22.0,
            "xtick.labelsize": 18.0,
            "ytick.labelsize": 18.0,
            "legend.fontsize": 14.0,
            "axes.linewidth": 0.9,
        }
    )
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(
        s_relative,
        field_theory / 1.0e3,
        color="#2453ff",
        lw=2.5,
        label="3D Gaussian theory",
        zorder=1,
    )
    axis.plot(
        s_relative,
        field_gaussian / 1.0e3,
        color="#d62728",
        linestyle="none",
        marker=".",
        ms=3.2,
        alpha=0.9,
        label="Gaussian-profile calculation",
        zorder=3,
    )
    axis.set_xlabel(r"$s-s_c$ [m]")
    axis.set_ylabel(r"$E_s$ [kV/m]")
    axis.legend(loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.98, bottom=0.15)
    fig.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)

    print(f"saved: {OUT_PNG}")
    print(f"saved: {OUT_PDF}")
    print(f"saved: {OUT_CSV}")
    print(f"saved: {OUT_TXT}")


if __name__ == "__main__":
    main()
