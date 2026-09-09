#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fig8_improvement_steps.py
==============================
Step-by-step visualization of how the systematic low bias of the PIC
scatter in the original Fig. 8 (Referee A, Sec. VI-B) is removed.

Same bunch (seed 42) is sliced by the 2.5D solver at NS=256 (original
Fig. 8) and NS=512 (improved).  Five panels apply the corrections one
by one, each with Es(s) on the left and the deviation from theory
[% of theory peak] on the right:

  (a) original Fig. 8:   NS=256, theory at NOMINAL sigma_s, 3D Gaussian theory
  (b) + theory at MEASURED sigma_s (removes fake offset from nominal width)
  (c) + NS=512 slices    (discretization /4, ds: 4.1 -> 2.05 cm ~ a/2)
  (d) + disk-kernel theory (same transverse model as the solver kernel:
      removes the intrinsic disk-vs-Gaussian ~3.2% peak difference)
  (e) + 8 cm Gaussian smoothing of lambda(s) (removes the shot noise)

Output:
  fig8_improvement_steps.png / fig8_improvement_steps.txt next to this script.

Run from anywhere:
    python -u benchmarks/reproducibility/scripts/01_fig8_es/plot_fig8_improvement_steps.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
os.chdir(REPO_ROOT)  # SEO files are referenced relative to repo root

from test_sc_260508_Es_formula import (
    load_seo_r_table, load_seo_pr_table,
    convert_seo_pr_to_pr0_geom, build_seo_s_matrix,
    generate_long_bunch_gaussian_s_on_closed_orbit,
    build_bunch_from_cyl, attach_seo_tables_to_bunch,
    prepare_fft_plan, unpack_solver_output,
    axisymmetric_gaussian_longitudinal_field_integral,
    SEO_R_FILE, SEO_PR_FILE,
    EK0_MEV, N_PARTICLES, Q_MACRO, NX, NZ,
    SIGMA_X, SIGMA_Z, BUNCH_CENTER_FRAC, SIGMA_S, S_CLIP_SIGMA,
    SEED, A_SMOOTH, EPS0,
)
from numba import set_num_threads

# ================================================================
# CONFIG — edit here
# ================================================================

NS_ORIG = 256           # original Fig. 8 slice count
NS_NEW = 512            # improved slice count
SMOOTH_LAM = 0.08       # [m] Gaussian smoothing of lambda(s)
OUT_PNG = Path(__file__).resolve().parent / "fig8_improvement_steps.png"
OUT_TXT = Path(__file__).resolve().parent / "fig8_improvement_steps.txt"

_LINES = []


def out(msg=""):
    print(msg, flush=True)
    _LINES.append(str(msg))


def rms(y):
    return float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)))


def disk_kernel(z, a):
    return (np.sign(z) / (2.0 * np.pi * EPS0 * a ** 2)
            * (1.0 - np.abs(z) / np.sqrt(z ** 2 + a ** 2)))


def gaussian_lambda_disk_theory(s_grid, s_center, sigma_s, total_charge,
                                a_radius, n_fine=40001):
    s_half = max(6.5 * sigma_s, np.max(np.abs(s_grid - s_center)) + 1.0)
    s_fine = np.linspace(-s_half, s_half, n_fine)
    ds = s_fine[1] - s_fine[0]
    lam = (total_charge / (np.sqrt(2.0 * np.pi) * sigma_s)
           * np.exp(-s_fine ** 2 / (2.0 * sigma_s ** 2)))
    K = disk_kernel(s_fine, a_radius)
    E_fine = np.convolve(lam, K * ds, mode="same")
    return np.interp(s_grid - s_center, s_fine, E_fine)


def run_solver(ns, bunch_data, seo_tables):
    """Run the fixProjection solver on the shared bunch with ns slices."""
    coords_cyl, ek_arr = bunch_data
    (seo_ek_axis, seo_phi_axis, seo_r_matrix, seo_pr_matrix,
     seo_pr0_geom_matrix, seo_s_matrix, seo_perimeter) = seo_tables
    bunch = build_bunch_from_cyl(coords_cyl, ek_arr, Q_MACRO, NX, NZ, ns)
    attach_seo_tables_to_bunch(
        bunch, seo_ek_axis, seo_phi_axis, seo_r_matrix, seo_pr_matrix,
        seo_pr0_geom_matrix, seo_s_matrix, seo_perimeter)
    fft_fwd, fft_inv = prepare_fft_plan(NX, NZ, threads=1)
    ret = bunch.Build_2_5D_SliceGrid_sbs_new_fixProjection(fft_fwd, fft_inv)
    sol = unpack_solver_output(ret)
    return (np.asarray(sol["s_grid"]), np.asarray(sol["rho_long"]),
            np.asarray(sol["Ef_map"]))


def main():
    set_num_threads(1)

    # ---- shared SEO tables and bunch (identical particles for both NS) ----
    seo_ek_axis, seo_phi_axis, seo_r_matrix = load_seo_r_table(SEO_R_FILE)
    seo_pr_matrix = load_seo_pr_table(SEO_PR_FILE)
    seo_pr0_geom_matrix = convert_seo_pr_to_pr0_geom(seo_pr_matrix)
    seo_s_matrix, seo_perimeter = build_seo_s_matrix(
        seo_phi_axis, seo_r_matrix, seo_pr0_geom_matrix)
    iE = int(np.argmin(np.abs(seo_ek_axis - EK0_MEV)))
    seo_tables = (seo_ek_axis, seo_phi_axis, seo_r_matrix, seo_pr_matrix,
                  seo_pr0_geom_matrix, seo_s_matrix, seo_perimeter)

    coords_cyl, ek_arr, _ = generate_long_bunch_gaussian_s_on_closed_orbit(
        seo_phi_axis=seo_phi_axis,
        r0_phi=seo_r_matrix[iE].copy(),
        pr0_phi=seo_pr0_geom_matrix[iE].copy(),
        s_phi=seo_s_matrix[iE].copy(),
        perimeter=float(seo_perimeter[iE]),
        n_particles=N_PARTICLES,
        bunch_center_frac=BUNCH_CENTER_FRAC,
        sigma_s=SIGMA_S, s_clip_sigma=S_CLIP_SIGMA,
        sigma_x=SIGMA_X, sigma_z=SIGMA_Z,
        ek0_mev=EK0_MEV, seed=SEED,
    )

    out("running solver at NS=256 (original Fig. 8) ...")
    s_o, q_o, E_o = run_solver(NS_ORIG, (coords_cyl, ek_arr), seo_tables)
    out("running solver at NS=512 (improved) ...")
    s_n, q_n, E_n = run_solver(NS_NEW, (coords_cyl, ek_arr), seo_tables)

    # ---- measured bunch parameters from the NS=512 lambda(s) ----
    s_center = float(np.sum(s_n * q_n) / np.sum(q_n))
    Q_tot = float(np.sum(q_n))
    w = q_n / np.sum(q_n)
    sigma_s_meas = float(np.sqrt(np.sum(w * (s_n - s_center) ** 2)))
    out(f"measured sigma_s = {sigma_s_meas:.4f} m (nominal {SIGMA_S}), "
        f"Q = {Q_tot:.6e} C")

    sigma_perp = float(np.sqrt(SIGMA_X * SIGMA_Z))

    def gauss_theory(s_grid, sigma_s):
        return axisymmetric_gaussian_longitudinal_field_integral(
            s_eval=s_grid, s_center=s_center, sigma_s=sigma_s,
            sigma_perp=sigma_perp, total_charge=Q_tot, eps0=EPS0,
            n_tau=800)

    # theories on each grid, at nominal and measured sigma_s
    Eg_o_nom = gauss_theory(s_o, SIGMA_S)
    Eg_o = gauss_theory(s_o, sigma_s_meas)
    Eg_n = gauss_theory(s_n, sigma_s_meas)
    Ed_n = gaussian_lambda_disk_theory(s_n, s_center, sigma_s_meas,
                                       Q_tot, A_SMOOTH)
    pk = float(np.max(np.abs(Eg_n)))   # common normalization

    # denoised PIC at NS=512: smooth lambda, direct-sum disk kernel
    s_rel_n = s_n - s_center
    kg = np.exp(-s_rel_n ** 2 / (2.0 * SMOOTH_LAM ** 2))
    kg /= kg.sum()
    q_sm = np.convolve(q_n, kg, mode="same")
    Kmat = disk_kernel(s_n[:, None] - s_n[None, :], A_SMOOTH)
    E_n_sm = Kmat @ q_sm

    # ---- five steps: (grid, PIC field, theory, label, note) ----
    steps = [
        (s_o, E_o, Eg_o_nom, NS_ORIG, "3D Gaussian theory",
         "(a) original Fig. 8: NS=256, theory at nominal "
         r"$\sigma_s$=1.5 m, Gaussian theory"),
        (s_o, E_o, Eg_o, NS_ORIG, "3D Gaussian theory",
         "(b) + theory at measured $\\sigma_s$="
         f"{sigma_s_meas:.4f} m"),
        (s_n, E_n, Eg_n, NS_NEW, "3D Gaussian theory",
         "(c) + NS=512 slices (ds: 4.1 -> 2.05 cm)"),
        (s_n, E_n, Ed_n, NS_NEW, "disk-kernel theory",
         "(d) + disk-kernel theory (same transverse model as solver)"),
        (s_n, E_n_sm, Ed_n, NS_NEW, "disk-kernel theory",
         "(e) + 8 cm smoothing of $\\lambda(s)$ (denoise)"),
    ]

    # ---- figure: 5 rows x 2 cols (Es | deviation) ----
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(5, 2, figsize=(11.0, 13.0), sharex="col")

    out("\nstep | rms dev | coherent (systematic) part | noise part "
        "[% of theory peak]")
    for i, (s_g, E_pic, E_th, ns, th_label, title) in enumerate(steps):
        s_rel = s_g - s_center
        m_in = np.abs(s_rel) < 3.5 * sigma_s_meas
        dev = (E_pic - E_th) / pk * 100
        r = rms(dev[m_in])
        # coherent part: smooth the deviation itself (kills the random
        # shot noise, keeps model/discretization/shape offsets)
        kg_i = np.exp(-s_rel ** 2 / (2.0 * SMOOTH_LAM ** 2))
        kg_i /= kg_i.sum()
        dev_sm = np.convolve(dev, kg_i, mode="same")
        r_coh = rms(dev_sm[m_in])
        r_noise = float(np.sqrt(max(r ** 2 - r_coh ** 2, 0.0)))
        out(f"    {title.split(':')[0]:5s}  rms = {r:5.2f}%   "
            f"coherent = {r_coh:5.2f}%   noise = {r_noise:5.2f}%")

        axL, axR = axes[i]
        axL.plot(s_rel, E_th, "-", color="#1f4aff", lw=1.8,
                 label=th_label, zorder=3)
        axL.plot(s_rel, E_pic, ".", ms=2.0, color="#d62728", alpha=0.25,
                 zorder=2, label=f"2.5D PIC ({ns} slices)")
        axL.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        axL.set_ylabel(r"$E_s$ [V/m]")
        axL.set_title(title, loc="left")
        axL.legend(loc="lower left", frameon=False)
        axL.grid(alpha=0.25)

        axR.plot(s_rel[m_in], dev[m_in], ".", ms=1.8, color="#999999",
                 alpha=0.30, zorder=2)
        axR.plot(s_rel[m_in], dev_sm[m_in], "-", color="#d62728", lw=1.6,
                 zorder=3, label="coherent part")
        axR.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        axR.set_ylabel("dev [% of peak]")
        axR.set_ylim(-14, 14)
        axR.grid(alpha=0.25)
        axR.text(0.03, 0.95,
                 f"rms = {r:.2f}%\ncoherent = {r_coh:.2f}%\n"
                 f"noise = {r_noise:.2f}%",
                 transform=axR.transAxes, va="top", ha="left",
                 fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec="0.7", alpha=0.9))
        if i == 0:
            axR.legend(loc="lower right", frameon=False, fontsize=8)

    axes[-1, 0].set_xlabel(r"$s-s_c$ [m]")
    axes[-1, 1].set_xlabel(r"$s-s_c$ [m]")
    fig.align_ylabels()
    fig.subplots_adjust(left=0.08, right=0.985, top=0.975, bottom=0.05,
                        hspace=0.35, wspace=0.22)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    out(f"\n[saved] {OUT_PNG}")
    plt.close(fig)

    OUT_TXT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"[saved] {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
