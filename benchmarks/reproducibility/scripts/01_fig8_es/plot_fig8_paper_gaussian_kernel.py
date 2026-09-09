#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-style Fig. 8 using the Gaussian transverse-profile kernel.

This is a separate benchmark implementation; it does not modify the production
uniform-disk kernel.  The longitudinal charge histogram is obtained from the
same 2.5D bunch construction, smoothed on a configurable scale, and convolved
with the source-slice-averaged Gaussian kernel.  The final figure contains only
the smoothed numerical curve and the full 3D Gaussian theory.
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
from test_sc_260508_Es_formula import (
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
# User parameters
# ---------------------------------------------------------------------------
NS_SOLVER = 256
SIGMA_PERP = float(np.sqrt(SIGMA_X * SIGMA_Z))
SMOOTH_SIGMA = 0.12  # m; minimum RMS in the scan and a smooth visual result
SMOOTH_SCAN = (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16)
SLICE_AVERAGE_SUBSAMPLES = 64
SHOW_FIGURE = True

OUT_PNG = HERE / "fig8_gaussian_kernel_paper.png"
OUT_PDF = HERE / "fig8_gaussian_kernel_paper.pdf"
OUT_CSV = HERE / "fig8_gaussian_kernel_paper_data.csv"


def gaussian_smooth(values, s_grid, sigma):
    """Smooth a uniformly sampled array with a normalized Gaussian window."""
    ds = float(s_grid[1] - s_grid[0])
    half_width = max(1, int(np.ceil(5.0 * sigma / ds)))
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64) * ds
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= np.sum(weights)
    return np.convolve(values, weights, mode="same")


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
    charge_weights = slice_charge / total_charge
    s_center = float(np.sum(s_grid * charge_weights))
    sigma_s_measured = float(
        np.sqrt(np.sum(charge_weights * (s_grid - s_center) ** 2))
    )
    s_relative = s_grid - s_center
    ds = float(s_grid[1] - s_grid[0])

    separation = s_grid[:, None] - s_grid[None, :]
    kernel_matrix = gauss_kernel_1d_slice_avg(
        separation,
        SIGMA_PERP,
        ds,
        n_int=SLICE_AVERAGE_SUBSAMPLES,
    )

    field_gaussian_theory = axisymmetric_gaussian_longitudinal_field_integral(
        s_eval=s_grid,
        s_center=s_center,
        sigma_s=sigma_s_measured,
        sigma_perp=SIGMA_PERP,
        total_charge=total_charge,
        eps0=EPS0,
        n_tau=800,
    )
    peak_theory = float(np.max(np.abs(field_gaussian_theory)))
    in_bunch = np.abs(s_relative) < 3.5 * sigma_s_measured

    print("Gaussian-kernel smoothing scan (relative to 3D Gaussian theory):")
    print(" sigma [cm]   RMS [% peak]   roughness [% peak]   peak diff [%]")
    scan_results = []
    for sigma_test in SMOOTH_SCAN:
        charge_test = gaussian_smooth(slice_charge, s_grid, sigma_test)
        field_test = kernel_matrix @ charge_test
        rms_test = float(
            np.sqrt(
                np.mean(
                    (
                        (field_test[in_bunch] - field_gaussian_theory[in_bunch])
                        / peak_theory
                    )
                    ** 2
                )
            )
            * 100.0
        )
        roughness_test = float(
            np.sqrt(np.mean(np.diff(field_test[in_bunch], n=2) ** 2))
            / peak_theory
            * 100.0
        )
        peak_difference_test = float(
            (np.max(np.abs(field_test)) - peak_theory) / peak_theory * 100.0
        )
        scan_results.append((sigma_test, rms_test, roughness_test,
                             peak_difference_test))
        print(
            f" {sigma_test * 100:8.1f}   {rms_test:12.3f}"
            f"   {roughness_test:18.3f}   {peak_difference_test:13.3f}"
        )

    slice_charge_smooth = gaussian_smooth(
        slice_charge, s_grid, SMOOTH_SIGMA
    )
    field_gaussian_kernel = kernel_matrix @ slice_charge_smooth
    rms_selected = float(
        np.sqrt(
            np.mean(
                (
                    (
                        field_gaussian_kernel[in_bunch]
                        - field_gaussian_theory[in_bunch]
                    )
                    / peak_theory
                )
                ** 2
            )
        )
        * 100.0
    )

    np.savetxt(
        OUT_CSV,
        np.column_stack(
            (s_relative, field_gaussian_kernel, field_gaussian_theory)
        ),
        delimiter=",",
        header=(
            "s_minus_sc_m,Es_smoothed_Gaussian_kernel_V_per_m,"
            "Es_3D_Gaussian_theory_V_per_m"
        ),
        comments="",
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.9,
        }
    )

    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(
        s_relative,
        field_gaussian_kernel / 1.0e3,
        color="#d62728",
        lw=1.8,
        label="Gaussian-kernel PIC (smoothed)",
        zorder=2,
    )
    axis.plot(
        s_relative,
        field_gaussian_theory / 1.0e3,
        color="#2453ff",
        lw=2.0,
        label="3D Gaussian theory",
        zorder=1,
    )
    axis.axhline(0.0, color="0.35", lw=0.65, alpha=0.65)
    axis.set_xlabel(r"$s-s_c$ [m]")
    axis.set_ylabel(r"$E_s$ [kV/m]")
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.legend(loc="lower right", frameon=False)

    fig.subplots_adjust(left=0.135, right=0.985, top=0.98, bottom=0.15)
    fig.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    print(
        f"N={N_PARTICLES}, NS={NS_SOLVER}, "
        f"measured sigma_s={sigma_s_measured:.6f} m"
    )
    print(
        f"selected smoothing sigma={SMOOTH_SIGMA * 100:.1f} cm, "
        f"RMS={rms_selected:.3f}% of theory peak"
    )
    print(f"saved: {OUT_PNG}")
    print(f"saved: {OUT_PDF}")
    print(f"saved: {OUT_CSV}")
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
