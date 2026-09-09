#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig8_gaussian_kernel_test.py
============================
Gaussian transverse-profile longitudinal kernel: implementation,
validation, and benchmark at production-like slice counts.

Motivation (Referee A, Sec. VI-B):
  * The bunch IS a 3D Gaussian, so the reference theory must remain the
    3D Gaussian integral (paper Eq. (48)) -- we cannot simply compare
    against the disk-kernel theory.
  * NS=512 slices is not acceptable for production; the original figure
    used NS=256 and the tracking chain uses NS=128.
  => The correct fix is to upgrade the solver's longitudinal kernel from
     a uniform disk (a=40 mm) to a Gaussian transverse profile
     (sigma_perp=20 mm), removing the ~3.2% model difference at its
     source, and to verify the residual at NS=256 and NS=128.

Gaussian 1D kernel (per unit line charge), derived by integrating the
on-axis field of a transverse Gaussian sheet rho_perp(r) =
1/(2*pi*sigma^2) * exp(-r^2/2 sigma^2):

    K_G(z) = sgn(z)/(4*pi*eps0*sigma^2)
             * [1 - 2*x*exp(x^2) * integral_x^inf exp(-t^2) dt]
           = sgn(z)/(4*pi*eps0*sigma^2)
             * [1 - sqrt(pi)*x*exp(x^2)*erfc(x)],
    x = |z|/(sqrt(2)*sigma),

i.e. only elementary functions plus the standard complementary error
function erfc (defined by the integral above).

Checks: K_G(0+/-) = +/-1/(4*pi*eps0*sigma^2) (finite cusp);
K_G -> sgn(z)/(4*pi*eps0*z^2) for |z| >> sigma (point-charge far field).
Numerical note: the product exp(x^2)*erfc(x) overflows in double
precision for x >~ 26 (here |z| >~ 0.75 m), so it is evaluated stably
as erfcx for x <= 6 and by its truncated asymptotic series
1 - 1/(2x^2) + 3/(4x^4) - 15/(8x^6) + 105/(16x^8) - 945/(32x^10)
for x > 6.  This is only an implementation detail; the formula above
is the definition.

Experiments:
  (0) kernel sanity: cusp value, far-field limit, erfcx branch continuity.
  (1) pure-theory validation: ideal Gaussian lambda convolved with K_G
      (fine grid) vs the 3D Gaussian integral theory -> ~0 if the kernel
      is the exact 1D reduction of the 3D Gaussian model.
  (2) noise-free discretization scan (midpoint vs slice-averaged kernel)
      for both kernels at NS = 128 / 256 / 512.
  (3) full benchmark on the real PIC bunch (seed 42, 1e6 macro):
      solver lambda-hat at NS = 128 and 256, direct-summed with the disk
      kernel and with the Gaussian kernel, compared to the 3D Gaussian
      theory, raw and after 8 cm smoothing.

Output: printed table + fig8_gaussian_kernel_test.txt next to script.

Run from anywhere:
    python -u benchmarks/reproducibility/scripts/01_fig8_es/fig8_gaussian_kernel_test.py
"""

import os
import sys
from math import erfc, exp, pi, sqrt
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

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

NS_LIST = (128, 256)     # production tracking uses 128; figure uses <=256
NS_SCAN = (128, 256, 512)
SMOOTH_LAM = 0.08        # [m] Gaussian smoothing width for lambda(s)
OUT_TXT = Path(__file__).resolve().parent / "fig8_gaussian_kernel_test.txt"
OUT_FIG_KERNEL = Path(__file__).resolve().parent / "fig8_gaussian_kernel_shape.png"
OUT_FIG_BENCH = Path(__file__).resolve().parent / "fig8_gaussian_kernel_benchmark.png"

SIGMA_PERP = float(np.sqrt(SIGMA_X * SIGMA_Z))   # 20 mm

_LINES = []


def out(msg=""):
    print(msg, flush=True)
    _LINES.append(str(msg))


def rms(y):
    return float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)))


# ================================================================
# kernels (per unit charge)
# ================================================================

def _exp_x2_erfc(x):
    """Stable evaluation of exp(x^2) * erfc(x)  (== erfcx(x)).

    Direct exp(x^2)*erfc(x) overflows in double precision for x >~ 26;
    use the direct product for x <= 6 and its truncated asymptotic
    series for x > 6."""
    if x <= 6.0:
        return exp(x * x) * erfc(x)
    x2 = x * x
    return ((1.0 - 0.5 / x2 + 0.75 / (x2 * x2) - 1.875 / (x2 ** 3)
             + 6.5625 / (x2 ** 4) - 29.53125 / (x2 ** 5))
            / (sqrt(pi) * x))


def gauss_kernel_1d(z, sigma):
    """Gaussian transverse-profile longitudinal kernel, vectorized:

        K_G(z) = sgn(z)/(4 pi eps0 sigma^2)
                 * [1 - sqrt(pi) x exp(x^2) erfc(x)],
        x = |z|/(sqrt(2) sigma).
    """
    z = np.asarray(z, dtype=np.float64)
    flat = np.abs(z).ravel() / (np.sqrt(2.0) * sigma)
    ex = np.array([_exp_x2_erfc(float(v)) for v in flat])
    x = flat.reshape(z.shape)
    exe = ex.reshape(z.shape)
    return (np.sign(z) / (4.0 * np.pi * EPS0 * sigma ** 2)
            * (1.0 - np.sqrt(np.pi) * x * exe))


def gauss_kernel_1d_scalar(z, sigma):
    """Scalar version of K_G (numba-friendly: only math.erfc + branches)."""
    x = abs(z) / (sqrt(2.0) * sigma)
    exe = _exp_x2_erfc(x)
    sgn = 1.0 if z > 0.0 else (-1.0 if z < 0.0 else 0.0)
    return (sgn / (4.0 * pi * EPS0 * sigma * sigma)
            * (1.0 - sqrt(pi) * x * exe))


def gauss_kernel_1d_slice_avg(z, sigma, ds, n_int=64):
    """Slice-averaged Gaussian kernel: mean of K_G(z - u) over the source
    slice u in [-ds/2, ds/2] (midpoint sub-samples)."""
    z = np.asarray(z, dtype=np.float64)
    u = (np.arange(n_int) + 0.5) / n_int * ds - ds / 2.0
    tot = np.zeros_like(z)
    for uu in u:
        tot += gauss_kernel_1d(z - uu, sigma)
    return tot / n_int


def disk_kernel(z, a):
    z = np.asarray(z, dtype=np.float64)
    return (np.sign(z) / (2.0 * np.pi * EPS0 * a ** 2)
            * (1.0 - np.abs(z) / np.sqrt(z ** 2 + a ** 2)))


def direct_sum(s_grid, q_s, kern):
    Kmat = kern(s_grid[:, None] - s_grid[None, :])
    return Kmat @ q_s





# ================================================================

def gaussian_lambda_disk_theory(s_grid, s_center, sigma_s, total_charge,
                                a_radius, n_fine=40001):
    s_half = max(6.5 * sigma_s, np.max(np.abs(s_grid - s_center)) + 1.0)
    s_fine = np.linspace(-s_half, s_half, n_fine)
    ds = s_fine[1] - s_fine[0]
    lam = (total_charge / (np.sqrt(2.0 * np.pi) * sigma_s)
           * np.exp(-s_fine ** 2 / (2.0 * sigma_s ** 2)))
    E_fine = np.convolve(lam, disk_kernel(s_fine, a_radius) * ds,
                         mode="same")
    return np.interp(s_grid - s_center, s_fine, E_fine)


def continuous_convolution(s_grid, s_center, sigma_s, total_charge, kern,
                           n_fine=40001):
    """Continuous ideal-Gaussian lambda convolved with kernel `kern`."""
    s_half = max(6.5 * sigma_s, np.max(np.abs(s_grid - s_center)) + 1.0)
    s_fine = np.linspace(-s_half, s_half, n_fine)
    ds = s_fine[1] - s_fine[0]
    lam = (total_charge / (np.sqrt(2.0 * np.pi) * sigma_s)
           * np.exp(-s_fine ** 2 / (2.0 * sigma_s ** 2)))
    E_fine = np.convolve(lam, kern(s_fine) * ds, mode="same")
    return np.interp(s_grid - s_center, s_fine, E_fine)


def discretized_field(s_grid, sigma_s, total_charge, s_center, kern,
                      slice_averaged=False):
    """Noise-free midpoint sampling of ideal Gaussian lambda on s_grid,
    direct-summed with kern (optionally slice-averaged over ds)."""
    ds = float(s_grid[1] - s_grid[0])
    lam = (total_charge / (np.sqrt(2.0 * np.pi) * sigma_s)
           * np.exp(-(s_grid - s_center) ** 2 / (2.0 * sigma_s ** 2)))
    q = lam * ds
    if not slice_averaged:
        return direct_sum(s_grid, q, kern)
    # slice-averaged kernel: K_bar(r) = (1/ds) int_{-ds/2}^{ds/2} K(r-u) du
    n_int = 64
    u = (np.arange(n_int) + 0.5) / n_int * ds - ds / 2.0
    r = s_grid[:, None] - s_grid[None, :]
    Kbar = np.mean(kern(r[..., None] - u[None, None, :]), axis=2)
    return Kbar @ q


def main():
    set_num_threads(1)
    out("=" * 72)
    out("Gaussian longitudinal kernel: validation + benchmark at NS=128/256")
    out("=" * 72)
    out(f"sigma_perp = {SIGMA_PERP*1e3:.1f} mm (Gaussian kernel), "
        f"a = {A_SMOOTH*1e3:.0f} mm (disk kernel)")

    # -----------------------------------------------------------
    # (0) kernel sanity
    # -----------------------------------------------------------
    cusp = 1.0 / (4.0 * np.pi * EPS0 * SIGMA_PERP ** 2)
    zt = np.array([0.5, 1.0, 2.0])
    far = gauss_kernel_1d(zt, SIGMA_PERP)
    out("\n[(0) kernel sanity]")
    out(f"    cusp 1/(4 pi eps0 sigma^2) = {cusp:.4e} V/m/C  "
        f"(disk cusp = {1.0/(2.0*np.pi*EPS0*A_SMOOTH**2):.4e})")
    out(f"    K_G(0+) computed = {gauss_kernel_1d(np.array([1e-12]), SIGMA_PERP)[0]:.4e}")
    for zz, ff in zip(zt, far):
        pt = 1.0 / (4.0 * np.pi * EPS0 * zz ** 2)
        out(f"    z={zz:4.1f} m: K_G = {ff:.4e}, point-charge = {pt:.4e}, "
            f"ratio = {ff/pt:.5f}")
    xs = np.linspace(5.999, 6.001, 5)
    vals = gauss_kernel_1d(xs * np.sqrt(2.0) * SIGMA_PERP, SIGMA_PERP)
    out(f"    erfcx branch continuity at x=6: max jump = "
        f"{np.max(np.abs(np.diff(vals))):.3e} "
        "(~1e-4 relative, from the truncated asymptotic series; "
        "sufficient for percent-level budgets)")

    # -----------------------------------------------------------
    # (1) pure-theory validation of K_G vs 3D Gaussian integral
    # -----------------------------------------------------------
    out("\n[(1) pure-theory validation: ideal Gaussian lambda, "
        "K_G convolution vs 3D Gaussian integral (continuous, fine grid)]")
    sigma_s = SIGMA_S
    Q_tot = 3.204353e-06
    s_center = 0.0
    s_fine = np.linspace(-6.5 * sigma_s, 6.5 * sigma_s, 40001)
    E_kg = continuous_convolution(s_fine, s_center, sigma_s, Q_tot,
                                  lambda z: gauss_kernel_1d(z, SIGMA_PERP))
    E_3d = axisymmetric_gaussian_longitudinal_field_integral(
        s_eval=s_fine, s_center=s_center, sigma_s=sigma_s,
        sigma_perp=SIGMA_PERP, total_charge=Q_tot, eps0=EPS0, n_tau=800)
    mm = np.abs(s_fine) < 3.5 * sigma_s
    pk = float(np.max(np.abs(E_3d)))
    out(f"    rms diff = {rms((E_kg - E_3d)[mm])/pk*100:.3f}% of peak, "
        f"max = {np.max(np.abs((E_kg - E_3d)[mm]))/pk*100:.3f}%")
    out("    -> if ~0, K_G is the exact 1D reduction of the 3D Gaussian "
        "model; the disk-vs-Gaussian model gap disappears by construction.")

    # -----------------------------------------------------------
    # (2) noise-free discretization scan, midpoint vs slice-averaged
    # -----------------------------------------------------------
    out("\n[(2) noise-free discretization, ideal Gaussian lambda on the "
        "SOLVER-like grid (span 10.57 m as in the production runs): "
        "rms deviation from the 3D Gaussian theory, % of peak]")
    header = "    NS    ds[cm]  ds/sig  | disk mid | gauss mid | gauss avg"
    out(header)
    for ns in NS_SCAN:
        g = np.linspace(-5.283, 5.283, ns)   # production grid span
        ds = g[1] - g[0]
        E_ref = axisymmetric_gaussian_longitudinal_field_integral(
            s_eval=g, s_center=0.0, sigma_s=sigma_s,
            sigma_perp=SIGMA_PERP, total_charge=Q_tot, eps0=EPS0,
            n_tau=800)
        mm = np.abs(g) < 3.5 * sigma_s
        E_dm = discretized_field(g, sigma_s, Q_tot, 0.0,
                                 lambda z: disk_kernel(z, A_SMOOTH))
        E_gm = discretized_field(g, sigma_s, Q_tot, 0.0,
                                 lambda z: gauss_kernel_1d(z, SIGMA_PERP))
        E_ga = discretized_field(
            g, sigma_s, Q_tot, 0.0,
            lambda z: gauss_kernel_1d_slice_avg(z, SIGMA_PERP, ds))
        out(f"    {ns:4d}  {ds*100:6.2f}  {ds/SIGMA_PERP:6.2f} | "
            f"{rms((E_dm-E_ref)[mm])/pk*100:7.2f}% | "
            f"{rms((E_gm-E_ref)[mm])/pk*100:8.2f}% | "
            f"{rms((E_ga-E_ref)[mm])/pk*100:8.2f}%")

    # -----------------------------------------------------------
    # (3) full benchmark on the real PIC bunch, NS = 128 / 256
    # -----------------------------------------------------------
    out("\n[(3) real PIC bunch (seed 42, 1e6 macro): solver lambda-hat "
        "direct-summed with disk vs Gaussian kernel, vs 3D Gaussian theory]")
    seo_ek_axis, seo_phi_axis, seo_r_matrix = load_seo_r_table(SEO_R_FILE)
    seo_pr_matrix = load_seo_pr_table(SEO_PR_FILE)
    seo_pr0_geom_matrix = convert_seo_pr_to_pr0_geom(seo_pr_matrix)
    seo_s_matrix, seo_perimeter = build_seo_s_matrix(
        seo_phi_axis, seo_r_matrix, seo_pr0_geom_matrix)
    iE = int(np.argmin(np.abs(seo_ek_axis - EK0_MEV)))
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
    seo_tables = (seo_ek_axis, seo_phi_axis, seo_r_matrix, seo_pr_matrix,
                  seo_pr0_geom_matrix, seo_s_matrix, seo_perimeter)

    bench = {}
    for ns in NS_LIST:
        out(f"\n    --- NS = {ns} ---")
        bunch = build_bunch_from_cyl(coords_cyl, ek_arr, Q_MACRO, NX, NZ,
                                     ns)
        attach_seo_tables_to_bunch(bunch, *seo_tables)
        fft_fwd, fft_inv = prepare_fft_plan(NX, NZ, threads=1)
        ret = bunch.Build_2_5D_SliceGrid_sbs_new_fixProjection(
            fft_fwd, fft_inv)
        sol = unpack_solver_output(ret)
        s_grid = np.asarray(sol["s_grid"])
        q_s = np.asarray(sol["rho_long"])
        Ef_solver = np.asarray(sol["Ef_map"])   # disk kernel inside solver

        s_c = float(np.sum(s_grid * q_s) / np.sum(q_s))
        Q_t = float(np.sum(q_s))
        w = q_s / np.sum(q_s)
        sig_meas = float(np.sqrt(np.sum(w * (s_grid - s_c) ** 2)))
        s_rel = s_grid - s_c
        m_in = np.abs(s_rel) < 3.5 * sig_meas

        E_ref = axisymmetric_gaussian_longitudinal_field_integral(
            s_eval=s_grid, s_center=s_c, sigma_s=sig_meas,
            sigma_perp=SIGMA_PERP, total_charge=Q_t, eps0=EPS0, n_tau=800)
        pk_i = float(np.max(np.abs(E_ref)))

        E_disk = direct_sum(s_grid, q_s, lambda z: disk_kernel(z, A_SMOOTH))
        E_gaus = direct_sum(s_grid, q_s,
                            lambda z: gauss_kernel_1d(z, SIGMA_PERP))
        ds_i = float(s_grid[1] - s_grid[0])
        E_gavg = direct_sum(
            s_grid, q_s,
            lambda z: gauss_kernel_1d_slice_avg(z, SIGMA_PERP, ds_i))

        kg = np.exp(-s_rel ** 2 / (2.0 * SMOOTH_LAM ** 2))
        kg /= kg.sum()
        q_sm = np.convolve(q_s, kg, mode="same")
        E_gaus_sm = direct_sum(s_grid, q_sm,
                               lambda z: gauss_kernel_1d(z, SIGMA_PERP))
        E_gavg_sm = direct_sum(
            s_grid, q_sm,
            lambda z: gauss_kernel_1d_slice_avg(z, SIGMA_PERP, ds_i))
        E_disk_sm = direct_sum(s_grid, q_sm,
                               lambda z: disk_kernel(z, A_SMOOTH))

        r_disk = rms((E_disk - E_ref)[m_in]) / pk_i * 100
        r_gm = rms((E_gaus - E_ref)[m_in]) / pk_i * 100
        r_ga = rms((E_gavg - E_ref)[m_in]) / pk_i * 100
        r_disk_sm = rms((E_disk_sm - E_ref)[m_in]) / pk_i * 100
        r_gm_sm = rms((E_gaus_sm - E_ref)[m_in]) / pk_i * 100
        r_ga_sm = rms((E_gavg_sm - E_ref)[m_in]) / pk_i * 100
        bench[ns] = dict(s_rel=s_rel, m_in=m_in, E_ref=E_ref, pk=pk_i,
                         E_disk=E_disk, E_gavg=E_gavg,
                         E_disk_sm=E_disk_sm, E_gavg_sm=E_gavg_sm,
                         r_ga=r_ga, r_ga_sm=r_ga_sm,
                         r_disk_sm=r_disk_sm)

        out(f"    measured sigma_s = {sig_meas:.4f} m, "
            f"ds = {ds_i*100:.2f} cm")
        out(f"    raw  lambda: disk rms = {r_disk:.2f}%  "
            f"(solver output itself: "
            f"{rms((Ef_solver-E_ref)[m_in])/pk_i*100:.2f}%)")
        out(f"    raw  lambda: gauss mid rms = {r_gm:.2f}% | "
            f"gauss slice-avg rms = {r_ga:.2f}%")
        out(f"    smoothed:    disk rms = {r_disk_sm:.2f}% | "
            f"gauss mid rms = {r_gm_sm:.2f}% | "
            f"gauss slice-avg rms = {r_ga_sm:.2f}%")

    out("\n[bottom line] with the Gaussian kernel the comparison against "
        "the 3D Gaussian theory contains NO model gap; residual = shot "
        "noise (removable by smoothing) + discretization at the chosen NS.")

    # -----------------------------------------------------------
    # (4) figures
    # -----------------------------------------------------------
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    # ---- Fig A: kernel shapes -------------------------------------
    zK = np.linspace(-3.0, 3.0, 6001)
    K_d = disk_kernel(zK, A_SMOOTH)
    K_g = gauss_kernel_1d(zK, SIGMA_PERP)
    K_pc = np.sign(zK) / (4.0 * np.pi * EPS0
                          * np.maximum(np.abs(zK), 1e-3) ** 2)
    figA, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    a1.plot(zK, K_d, "-", color="#0a7a30", lw=1.6,
            label=f"disk kernel, $a$={A_SMOOTH*1e3:.0f} mm")
    a1.plot(zK, K_g, "-", color="#1f4aff", lw=1.6,
            label=rf"Gaussian kernel, $\sigma_\perp$="
                  f"{SIGMA_PERP*1e3:.0f} mm")
    for ns_i, cc in ((256, "#d62728"), (128, "#9467bd")):
        ds_i = 10.566 / ns_i
        K_avg = gauss_kernel_1d_slice_avg(zK, SIGMA_PERP, ds_i)
        a1.plot(zK, K_avg, "--", color=cc, lw=1.3,
                label=f"Gaussian slice-avg, NS={ns_i} "
                      f"(ds={ds_i*100:.1f} cm)")
    a1.set_xlim(-0.5, 0.5)
    a1.set_xlabel("z [m]")
    a1.set_ylabel(r"$K(z)$ [V/m/C]")
    a1.legend(loc="upper right", frameon=False, fontsize=7.5)
    a1.grid(alpha=0.25)
    a1.set_title("(a) near field: cusp at $z=0$", loc="left")
    a2.semilogy(np.abs(zK), np.abs(K_d), "-", color="#0a7a30", lw=1.4,
                label="disk")
    a2.semilogy(np.abs(zK), np.abs(K_g), "-", color="#1f4aff", lw=1.4,
                label="Gaussian")
    a2.semilogy(np.abs(zK), np.abs(K_pc), ":", color="k", lw=1.2,
                label=r"point charge $1/(4\pi\varepsilon_0 z^2)$")
    a2.set_xlim(1e-2, 3.0)
    a2.set_xscale("log")
    a2.set_xlabel("|z| [m]")
    a2.set_ylabel(r"$|K(z)|$ [V/m/C]")
    a2.legend(loc="lower left", frameon=False, fontsize=8)
    a2.grid(alpha=0.25, which="both")
    a2.set_title("(b) far field: both kernels -> point charge", loc="left")
    figA.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.14,
                         wspace=0.28)
    figA.savefig(OUT_FIG_KERNEL, dpi=300, bbox_inches="tight")
    plt.close(figA)
    out(f"\n[saved] {OUT_FIG_KERNEL}")

    # ---- Fig B: separate panels, one data set per panel -------------
    # All at NS=256 (the slice count of the published Fig. 8).
    # Each panel: 3D Gaussian theory (blue line) + one PIC result.
    d = bench[256]
    panels = [
        ("(a) original Fig. 8: disk kernel, NS=256",
         d["E_disk"], "#888888", ".", 2.6, 0.35),
        ("(b) Gaussian kernel, NS=256",
         d["E_gavg"], "#2ca02c", ".", 3.2, 0.45),
        ("(c) Gaussian kernel + smoothing, NS=256",
         d["E_gavg_sm"], "#d62728", ".", 2.8, 0.85),
    ]
    s_rel = d["s_rel"]
    figB, axesB = plt.subplots(1, 3, figsize=(15.5, 4.4), sharex=True,
                               sharey=True)
    for axB, (title, E_pic, color, mkr, ms, alpha) in \
            zip(axesB, panels):
        axB.plot(s_rel, d["E_ref"], "-", color="#1f4aff", lw=2.0,
                 label=r"3D Gaussian theory, $\sigma_\perp$=20 mm",
                 zorder=5)
        axB.plot(s_rel, E_pic, mkr, ms=ms, color=color, alpha=alpha,
                 zorder=3, label="2.5D PIC")
        axB.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        axB.set_title(title, loc="left", fontsize=9.5)
        axB.grid(alpha=0.25)
        if axB is axesB[0]:
            axB.set_ylabel(r"$E_s$ [V/m]")
        axB.set_xlabel(r"$s-s_c$ [m]")
        axB.legend(loc="lower left", frameon=False, fontsize=8)
    figB.align_ylabels()
    figB.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.12,
                         wspace=0.10)
    figB.savefig(OUT_FIG_BENCH, dpi=300, bbox_inches="tight")
    plt.close(figB)
    out(f"[saved] {OUT_FIG_BENCH}")

    OUT_TXT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
