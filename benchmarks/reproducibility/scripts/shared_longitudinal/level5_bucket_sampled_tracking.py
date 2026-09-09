#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Level-5: bucket-sampled initial distribution + adaptive-frozen tracking.

Instead of a matched Gaussian, the initial distribution is created by
uniform rejection sampling inside the bare-RF bucket (H(zd, dEk) < H_sep).
This is closer to the physics of a painted bunch whose phase-space area
is filled by multi-turn injection.

Two modes are run side-by-side on the SAME initial particles:
    bare RF  : SC off (Es = 0),  particles follow the bare-RF Hamiltonian
    adaptive : SC on,  Es(s) rebuilt every ES_UPDATE_INTERVAL steps from the
               particle histogram (Gaussian-smoothed) convolved with the
               analytic uniform-disk kernel — NO shape assumption (1D theory).
               Es grid is full-intensity; f_sc scales the force only.

Outputs (./outputs/):
    level5_bucket_initial.png        : initial distribution on H contour
    level5_evolution.png             : sigma_s, sigma_E, Es_max vs turn
    level5_phasespace_bareRF.png     : phase-space snapshots (bare RF)
    level5_phasespace_adaptive.png   : phase-space snapshots (adaptive SC)
    level5_phasespace_overlay.png    : bare RF + adaptive overlaid at key turns
    level5_summary.txt               : key numbers

All parameters are set in the CONFIG section (before main); no command-line
arguments. Figures pop up with plt.show() when SHOW_FIGS = True and are
saved to ./outputs/ when SAVE_FIGS = True.
"""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.signal import fftconvolve

# ================================================================
# Physical constants (CSNS/FFA 300 MeV injection)
# ================================================================

E_CHARGE = 1.602176634e-19
M0C2_MEV = 938.2720813
C_LIGHT = 2.99792458e8
EPS0 = 8.8541878128e-12

E0_MEV = 300.0
GAMMA = 1.0 + E0_MEV / M0C2_MEV
BETA = np.sqrt(1.0 - 1.0 / GAMMA**2)
BETA_C = BETA * C_LIGHT

N_PROTONS = 2.08e13
Q_TOTAL = N_PROTONS * E_CHARGE

A_DISK = 0.04          # uniform-disk kernel radius [m] (matches solver)

# RF (painting stage, stationary bucket)
N_GAPS = 2
U0_GAP = 100e3
V_TURN = N_GAPS * U0_GAP
PHI_S = 0.0

# SC grid
SC_S_MAX = 12.0
SC_NS = 512

# ================================================================
# SEO frequency table
# ================================================================

def load_seo_freq_table(seo_ini_path):
    raw = np.loadtxt(seo_ini_path, skiprows=2)
    Ek = raw[:, 1].astype(np.float64)
    Freq = raw[:, 4].astype(np.float64)
    order = np.argsort(Ek)
    return np.ascontiguousarray(Ek[order]), np.ascontiguousarray(Freq[order])


# ================================================================
# Bare-RF dynamics (from level1_bucket_toy.py)
# ================================================================

def dz_dt(zd, dEk, E0, Tc, Ek_grid, T_grid):
    Ek = E0 + dEk
    T_Ek = np.interp(Ek, Ek_grid, T_grid)
    return -(Tc - T_Ek) / Tc


def dE_dt(zd, dEk, E0, Tc, fi0, w0, V_turn, Ek_grid, T_grid):
    sin_val = np.sin(fi0 + w0 * zd) - np.sin(fi0)
    return (V_turn * sin_val / Tc) / 1e6    # [MeV/s]


def dynamic_rhs(zd, dEk, E0, Tc, fi0, w0, V_turn, Ek_grid, T_grid):
    return dz_dt(zd, dEk, E0, Tc, Ek_grid, T_grid), \
           dE_dt(zd, dEk, E0, Tc, fi0, w0, V_turn, Ek_grid, T_grid)


# ================================================================
# Bucket Hamiltonian + sampling
# ================================================================

def compute_separatrix_zd(G):
    phi_u = np.pi - G["fi0"]
    return (phi_u - G["fi0"]) / G["w0"]


def build_hamiltonian(G, zd_lims, dEk_lims, Nq=401, Np=401):
    q_axis = np.linspace(zd_lims[0], zd_lims[1], Nq)
    p_axis = np.linspace(dEk_lims[0], dEk_lims[1], Np)

    # A(p) = zd_dot at zd=0
    zd_zero = np.zeros(1, dtype=np.float64)
    A_vals = np.empty(Np, dtype=np.float64)
    for j in range(Np):
        A_vals[j] = dz_dt(zd_zero, p_axis[j:j+1],
                          G["E0"], G["Tc"], G["Ek_grid"], G["T_grid"])[0]

    # B(q) = dEk_dot at dEk=0
    dEk_zero = np.zeros(1, dtype=np.float64)
    B_vals = np.empty(Nq, dtype=np.float64)
    for i in range(Nq):
        B_vals[i] = dE_dt(q_axis[i:i+1], dEk_zero,
                          G["E0"], G["Tc"], G["fi0"], G["w0"],
                          G["V_turn"], G["Ek_grid"], G["T_grid"])[0]

    dq = q_axis[1] - q_axis[0]
    dp = p_axis[1] - p_axis[0]
    acc_q = np.zeros(Nq, dtype=np.float64)
    acc_p = np.zeros(Np, dtype=np.float64)
    for i in range(1, Nq):
        acc_q[i] = acc_q[i-1] + 0.5 * (B_vals[i-1] + B_vals[i]) * dq
    for j in range(1, Np):
        acc_p[j] = acc_p[j-1] + 0.5 * (A_vals[j-1] + A_vals[j]) * dp

    Hq, Hp = -acc_q, acc_p
    H_map = Hq[:, None] + Hp[None, :]
    H00 = float(np.interp(0.0, q_axis, Hq) + np.interp(0.0, p_axis, Hp))
    H_map -= H00

    sepx = compute_separatrix_zd(G)
    H_sep = float(np.interp(sepx, q_axis, Hq) + np.interp(0.0, p_axis, Hp) - H00)

    return q_axis, p_axis, H_map, H_sep, Hq, Hp


def sample_bucket_uniform(n_part, G, seed=42, fill_frac=1.0, Nq=301, Np=301):
    """
    Uniform sampling inside the bucket (H < fill_frac * H_sep).

    Builds a Hamiltonian grid, uses its own H_sep, then rejection-samples:
    candidate positions are drawn uniformly from trapped grid cells, and
    bilinear interpolation of H_map rejects any particle outside the bucket.
    Each accepted particle is forced strictly inside via H verification.

    Returns:
        zd, dEk : particle arrays (all guaranteed H < fill_frac * H_sep)
        bucket_grid : dict with q_axis, p_axis, H_map, H_sep
    """
    rng = np.random.default_rng(seed)
    zd_sep = compute_separatrix_zd(G)

    # Bucket height (linear eta — used only to set grid bounds)
    beta2 = 1.0 - (M0C2_MEV / (E0_MEV + M0C2_MEV))**2
    Ek_grid_loc = G["Ek_grid"]
    T_grid_loc = G["T_grid"]
    dT_dEk_vals = np.gradient(T_grid_loc, Ek_grid_loc)
    dT_dEk_at_E0 = float(np.interp(G["E0"], Ek_grid_loc, dT_dEk_vals))
    eta_lin = dT_dEk_at_E0 * beta2 * (G["E0"] + M0C2_MEV) / G["Tc"]
    V_turn = G["V_turn"]
    Fsep = np.cos(G["fi0"]) - 0.5*(np.pi - 2.0*abs(G["fi0"]))*np.sin(abs(G["fi0"]))
    E_tot_MeV = G["E0"] + M0C2_MEV
    dEk_bucket = np.sqrt(max(2.0 * V_turn * beta2 * E_tot_MeV * 1e6
                              / (np.pi * abs(eta_lin)) * Fsep, 0.0)) / 1e6

    # Build ONE Hamiltonian grid — used for sampling, H-interpolation, and plotting
    q_axis, p_axis, H_map, H_sep, Hq, Hp = build_hamiltonian(
        G,
        (-zd_sep * 1.05, zd_sep * 1.05),
        (-dEk_bucket * 1.05, dEk_bucket * 1.05),
        Nq=Nq, Np=Np,
    )

    dq = float(q_axis[1] - q_axis[0])
    dp = float(p_axis[1] - p_axis[0])

    # Trapped-cell mask — use THIS grid's own H_sep
    # H_sep < 0, H=0 at SFP.  fill_frac * H_sep sets the inner boundary:
    #   fill_frac=1.0 -> H >= H_sep  (full bucket)
    #   fill_frac=0.1 -> H >= 0.1*H_sep  (inner 10%)
    # Small epsilon guards against roundoff at the separatrix for fill_frac=1.0.
    H_threshold = fill_frac * H_sep + 1e-12   # negative, closer to 0 for small fill_frac
    trapped = (H_map >= H_threshold) & (H_map < 0.0)
    trapped_flat = trapped.ravel()
    trapped_indices = np.where(trapped_flat)[0]
    n_trapped_cells = len(trapped_indices)

    if n_trapped_cells == 0:
        raise RuntimeError("No trapped cells found in bucket sampling grid.")

    # Bilinear interpolation helper for fast H lookup
    iq_idx = np.arange(Nq)
    ip_idx = np.arange(Np)

    def H_at_points(zd_arr, dEk_arr):
        """Bilinear interpolation of H_map at arbitrary (zd, dEk)."""
        # Find fractional indices
        fq = np.clip((zd_arr - q_axis[0]) / dq, 0.0, Nq - 1.001)
        fp = np.clip((dEk_arr - p_axis[0]) / dp, 0.0, Np - 1.001)
        iq0 = np.floor(fq).astype(int)
        ip0 = np.floor(fp).astype(int)
        iq1 = np.minimum(iq0 + 1, Nq - 1)
        ip1 = np.minimum(ip0 + 1, Np - 1)
        wq = fq - iq0
        wp = fp - ip0
        H00 = H_map[iq0, ip0]
        H10 = H_map[iq1, ip0]
        H01 = H_map[iq0, ip1]
        H11 = H_map[iq1, ip1]
        return (H00 * (1 - wq) * (1 - wp) +
                H10 * wq * (1 - wp) +
                H01 * (1 - wq) * wp +
                H11 * wq * wp)

    # Rejection loop: generate batches, interpolate H, keep those inside
    zd_out = np.empty(n_part, dtype=np.float64)
    dEk_out = np.empty(n_part, dtype=np.float64)
    n_found = 0
    batch_size = max(4 * n_part, 20000)

    while n_found < n_part:
        # Randomly select trapped cells for the batch
        cell_idx = rng.choice(trapped_indices, size=batch_size)
        iq, ip = divmod(cell_idx, Np)
        z_trial = q_axis[iq] + rng.uniform(0.0, dq, size=batch_size)
        e_trial = p_axis[ip] + rng.uniform(0.0, dp, size=batch_size)

        H_trial = H_at_points(z_trial, e_trial)
        accept = (H_trial >= H_threshold) & (H_trial < 0.0)
        n_accept = int(np.sum(accept))
        need = n_part - n_found
        take = min(n_accept, need)
        idx_accept = np.where(accept)[0][:take]
        zd_out[n_found:n_found + take] = z_trial[idx_accept]
        dEk_out[n_found:n_found + take] = e_trial[idx_accept]
        n_found += take

    bucket_grid = {"q_axis": q_axis, "p_axis": p_axis,
                   "H_map": H_map, "H_sep": H_sep}

    return zd_out, dEk_out, bucket_grid


# ================================================================
# SC kernel (numba-accelerated)
# ================================================================

@njit
def disk_kernel_numba(z):
    """Uniform-disk kernel E_z at axial distance z [m], radius a=0.04 m."""
    a = 0.04
    abs_z = np.abs(z)
    return (1.0 / (2.0 * np.pi * EPS0 * a * a)
            * np.sign(z) * (1.0 - abs_z / np.sqrt(z * z + a * a)))


@njit
def histogram_numba(s_part, s_min, ds, counts_out):
    """
    Fill *pre-allocated* counts_out with histogram of s_part.
    Uniform bins: bin i = [s_min + i*ds, s_min + (i+1)*ds).
    Particles outside are discarded.

    counts_out is zeroed in-place and reused — no allocation per call.
    """
    n_parts = len(s_part)
    n_bins = len(counts_out)
    s_max = s_min + n_bins * ds
    inv_ds = 1.0 / ds

    # Zero the buffer
    counts_out[:] = 0.0

    # Binning loop — single bounds check, then idx is guaranteed valid
    for i in range(n_parts):
        s = s_part[i]
        if s < s_min or s >= s_max:
            continue
        idx = int((s - s_min) * inv_ds)
        counts_out[idx] += 1.0

    return counts_out


# ================================================================
# RHS and RK4 with precomputed uniform lookup table (numba)
# ================================================================

@njit
def interp_T_uniform(Ek_arr, E_min, dE, T_table):
    """
    Fast T(Ek) lookup: linear interp on uniform precomputed table.
    Replaces np.interp which is slow inside numba.
    """
    n = len(Ek_arr)
    out = np.empty(n, dtype=np.float64)
    n_table = len(T_table)
    inv_dE = 1.0 / dE
    for i in range(n):
        idx_float = (Ek_arr[i] - E_min) * inv_dE
        idx = int(idx_float)
        if idx < 0:
            out[i] = T_table[0]
        elif idx >= n_table - 1:
            out[i] = T_table[n_table - 1]
        else:
            frac = idx_float - idx
            out[i] = T_table[idx] * (1.0 - frac) + T_table[idx + 1] * frac
    return out


@njit
def interp_Es_uniform(s_part, s_min, ds, Es_grid):
    """
    Fast Es(s) lookup: linear interp on uniform SC grid.
    Like interp_T_uniform — avoids np.interp overhead in numba.
    """
    n = len(s_part)
    n_bins = len(Es_grid)
    inv_ds = 1.0 / ds
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx_float = (s_part[i] - s_min) * inv_ds
        idx = int(idx_float)
        if idx < 0:
            out[i] = Es_grid[0]
        elif idx >= n_bins - 1:
            out[i] = Es_grid[n_bins - 1]
        else:
            frac = idx_float - idx
            out[i] = Es_grid[idx] * (1.0 - frac) + Es_grid[idx + 1] * frac
    return out


@njit
def rhs_numba(zd, dEk, Tc, w0, fis, V_turn,
              E_min, dE_table, T_table,
              f_sc, BETA_C, Es_part):
    """RHS: fast uniform-table interpolation instead of np.interp."""
    Ek_all = E0_MEV + dEk
    T_Ek_vec = interp_T_uniform(Ek_all, E_min, dE_table, T_table)
    dzd = -(Tc - T_Ek_vec) / Tc

    sin_rf = np.sin(fis + w0 * zd) - np.sin(fis)
    ddek_rf = V_turn * sin_rf / Tc / 1e6       # [MeV/s]
    ddek = ddek_rf + f_sc * Es_part * BETA_C / 1e6

    return dzd, ddek


@njit
def rk4_step_numba(zd, dEk, dt, Tc, w0, fis, V_turn,
                   E_min, dE_table, T_table,
                   f_sc, BETA_C, Es_part):
    """One full RK4 step, Es held constant.  Uses fast uniform-table interp."""
    k1z, k1e = rhs_numba(zd, dEk, Tc, w0, fis, V_turn,
                         E_min, dE_table, T_table, f_sc, BETA_C, Es_part)
    k2z, k2e = rhs_numba(zd + 0.5 * dt * k1z, dEk + 0.5 * dt * k1e,
                         Tc, w0, fis, V_turn,
                         E_min, dE_table, T_table, f_sc, BETA_C, Es_part)
    k3z, k3e = rhs_numba(zd + 0.5 * dt * k2z, dEk + 0.5 * dt * k2e,
                         Tc, w0, fis, V_turn,
                         E_min, dE_table, T_table, f_sc, BETA_C, Es_part)
    k4z, k4e = rhs_numba(zd + dt * k3z, dEk + dt * k3e,
                         Tc, w0, fis, V_turn,
                         E_min, dE_table, T_table, f_sc, BETA_C, Es_part)

    zd_new = zd + dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
    dEk_new = dEk + dt / 6.0 * (k1e + 2.0 * k2e + 2.0 * k3e + k4e)

    return zd_new, dEk_new


# ================================================================
# Es from particle histogram + numba convolution + smooth
# ================================================================

def compute_Es_from_particles(s_part, s_edges, q_macro, smooth_sigma,
                              precomputed_kernel, counts_buffer):
    """
    Per-step Es: numba histogram -> smooth -> FFT-convolve with kernel -> interp.

    precomputed_kernel = disk_kernel_numba(s_centers) * ds  (size = N_bins)
    counts_buffer: pre-allocated float64 array of len = N_bins, reused.
    Returns (s_centers, lam, Es_grid).
    """
    ds = s_edges[1] - s_edges[0]
    centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    # Numba zero-fill histogram into pre-allocated buffer
    histogram_numba(s_part, s_edges[0], ds, counts_buffer)
    lam = counts_buffer * q_macro / ds

    if smooth_sigma and smooth_sigma > 0:
        n = len(centers)
        half = min(int(4.0 * smooth_sigma / ds), n // 2)
        if half > 1:
            w = np.exp(-0.5 * (np.arange(-half, half + 1) * ds / smooth_sigma) ** 2)
            w /= w.sum()
            lam = np.convolve(lam, w, mode="same")

    Es_grid = fftconvolve(lam, precomputed_kernel, mode="same")
    return centers, lam, Es_grid


# ================================================================
# Tracking loop (numba-accelerated inner step)
# ================================================================

def track_bunch(zd0, dEk0, mode, n_turns, n_steps_per_turn,
                P, s_edges, snap_turns, smooth_sigma,
                precomputed_kernel, counts_buffer, es_update_interval=1,
                t_lookup=None):
    """
    mode: 'bare' (Es=0) | 'adaptive' (Es rebuilt via
           numba histogram + FFT convolution + disk kernel)

    counts_buffer: pre-allocated float64 array (len = N_bins), reused.
    es_update_interval: recompute Es every N steps (1 = every step).
    t_lookup: (E_min, dE_table, T_table) precomputed uniform lookup.
    Returns dict with turns, sig_s, sig_E, Es_max, snapshots, final Es_grid.
    """
    zd = zd0.copy()
    dEk = dEk0.copy()
    dt = P["Tc"] / n_steps_per_turn
    f_sc = P["f_sc"]
    # Es grid is built at FULL intensity; f_sc scales the FORCE only
    # (same convention as level2).  Do NOT multiply q_macro by f_sc here,
    # otherwise the SC force becomes proportional to f_sc**2.
    q_macro = Q_TOTAL / len(zd0)

    s_centers_arr = 0.5 * (s_edges[:-1] + s_edges[1:])
    ds_grid = s_edges[1] - s_edges[0]
    s_min_grid = s_edges[0]
    # Pre-extract for numba calls
    Tc = P["Tc"]
    w0 = P["w0"]
    fis = P["fis"]
    V_turn = P["V_turn"]
    E_min, dE_table, T_table = t_lookup

    hist_turn, hist_sig_s, hist_sig_E = [], [], []
    hist_Es_max = []

    snap_set = set(snap_turns)
    snapshots = {}
    Es_grid_final = np.zeros_like(s_centers_arr)

    for turn in range(n_turns + 1):
        if turn % 10 == 0:
            print(f"  turn {turn:>4d}/{n_turns} ...", flush=True)
        s_part = -BETA_C * zd
        hist_turn.append(turn)
        hist_sig_s.append(np.std(s_part))
        hist_sig_E.append(np.std(dEk))

        # Per-turn Es record
        if mode != "bare" and f_sc > 0:
            _, _, Es_turn = compute_Es_from_particles(
                s_part, s_edges, q_macro, smooth_sigma, precomputed_kernel,
                counts_buffer)
            hist_Es_max.append(np.max(np.abs(Es_turn)) * f_sc)  # effective field
        else:
            hist_Es_max.append(0.0)

        if turn in snap_set:
            snapshots[turn] = (zd.copy(), dEk.copy(), s_part.copy())

        if turn == n_turns:
            if mode != "bare" and f_sc > 0:
                _, _, Es_grid_final = compute_Es_from_particles(
                    s_part, s_edges, q_macro, smooth_sigma, precomputed_kernel,
                    counts_buffer)
                Es_grid_final = Es_grid_final * f_sc   # effective field
            break

        # ---- Step loop: Es recomputed every es_update_interval steps ----
        n_steps = n_steps_per_turn
        for step_i in range(n_steps):
            if mode != "bare" and f_sc > 0 and step_i % es_update_interval == 0:
                s_part = -BETA_C * zd
                centers, _, Es_grid = compute_Es_from_particles(
                    s_part, s_edges, q_macro, smooth_sigma, precomputed_kernel,
                    counts_buffer)
                Es_part = interp_Es_uniform(
                    s_part, s_min_grid, ds_grid, Es_grid)
            elif mode == "bare" or f_sc == 0:
                if step_i == 0:
                    Es_part = np.zeros(len(zd), dtype=np.float64)

            zd, dEk = rk4_step_numba(
                zd, dEk, dt,
                Tc, w0, fis, V_turn,
                E_min, dE_table, T_table,
                f_sc, BETA_C, Es_part)

    return {
        "mode": mode,
        "turns": np.array(hist_turn),
        "sig_s": np.array(hist_sig_s),
        "sig_E": np.array(hist_sig_E),
        "Es_max": np.array(hist_Es_max),
        "snapshots": snapshots,
        "Es_grid_final": Es_grid_final,
        "s_grid": s_centers_arr,
    }


# ================================================================
# Plotting
# ================================================================

def sample_gaussian_matched(n_part, sigma_s0, sigma_E0, seed=42):
    """
    Gaussian initial distribution: zd ~ N(0, σs₀/βc), dEk ~ N(0, σE₀).
    Returns (zd, dEk) arrays.
    """
    rng = np.random.default_rng(seed)
    sigma_zd = sigma_s0 / BETA_C
    zd = rng.normal(0.0, sigma_zd, n_part).astype(np.float64)
    dEk = rng.normal(0.0, sigma_E0, n_part).astype(np.float64)
    return zd, dEk


def sample_gaussian_truncated(n_part, sigma_s0, sigma_E0, G, seed=42):
    """
    Gaussian inside bucket: reject particles with H(zd,dEk) >= H_sep.
    Falls back to plain Gaussian if bucket grid not buildable.
    Returns (zd, dEk, bucket_grid).
    """
    zd_sep = compute_separatrix_zd(G)
    beta2 = 1.0 - (M0C2_MEV / (E0_MEV + M0C2_MEV))**2
    Ek_grid_loc = G["Ek_grid"]
    T_grid_loc = G["T_grid"]
    dTdEk_grad = np.gradient(T_grid_loc, Ek_grid_loc)
    dT_dEk_at_E0 = float(np.interp(G["E0"], Ek_grid_loc, dTdEk_grad))
    eta_lin = dT_dEk_at_E0 * beta2 * (G["E0"] + M0C2_MEV) / G["Tc"]
    Fsep = np.cos(G["fi0"]) - 0.5*(np.pi-2.0*abs(G["fi0"]))*np.sin(abs(G["fi0"]))
    dEk_bucket = np.sqrt(max(2.0*G["V_turn"]*beta2*(G["E0"]+M0C2_MEV)*1e6
                              / (np.pi*abs(eta_lin))*Fsep, 0.0)) / 1e6

    q_axis, p_axis, H_map, H_sep, _, _ = build_hamiltonian(
        G, (-zd_sep*1.05, zd_sep*1.05), (-dEk_bucket*1.05, dEk_bucket*1.05),
        Nq=301, Np=301)
    dq = q_axis[1] - q_axis[0]
    dp = p_axis[1] - p_axis[0]

    rng = np.random.default_rng(seed)
    sigma_zd = sigma_s0 / BETA_C
    batch = max(4 * n_part, 20000)
    zd_out = np.empty(n_part, dtype=np.float64)
    dEk_out = np.empty(n_part, dtype=np.float64)
    n_found = 0

    while n_found < n_part:
        z_trial = rng.normal(0.0, sigma_zd, batch)
        e_trial = rng.normal(0.0, sigma_E0, batch)

        # bilinear interp H
        fq = np.clip((z_trial - q_axis[0]) / dq, 0.0, len(q_axis) - 1.001)
        fp = np.clip((e_trial - p_axis[0]) / dp, 0.0, len(p_axis) - 1.001)
        iq0 = np.floor(fq).astype(int)
        ip0 = np.floor(fp).astype(int)
        iq1 = np.minimum(iq0 + 1, len(q_axis) - 1)
        ip1 = np.minimum(ip0 + 1, len(p_axis) - 1)
        wq, wp = fq - iq0, fp - ip0
        H_trial = (H_map[iq0, ip0] * (1-wq)*(1-wp) + H_map[iq1, ip0] * wq*(1-wp)
                   + H_map[iq0, ip1] * (1-wq)*wp + H_map[iq1, ip1] * wq*wp)

        accept = (H_trial >= H_sep + 1e-12) & (H_trial < 0.0)
        n_accept = int(np.sum(accept))
        need = n_part - n_found
        take = min(n_accept, need)
        idx = np.where(accept)[0][:take]
        zd_out[n_found:n_found + take] = z_trial[idx]
        dEk_out[n_found:n_found + take] = e_trial[idx]
        n_found += take

    bucket_grid = {"q_axis": q_axis, "p_axis": p_axis,
                   "H_map": H_map, "H_sep": H_sep}
    return zd_out, dEk_out, bucket_grid


def sample_waterbag_truncated(n_part, sigma_s0, sigma_E0, G, seed=42):
    """
    Waterbag (uniform ellipse) in (zd, dEk), truncated to bucket interior.

    Ellipse: (zd/a_zd)² + (dEk/b_dEk)² ≤ 1, uniform filling in area.
    For 2D uniform ellipse: sigma = semi-axis / 2, so
      a_zd = 2 * sigma_s0 / beta_c,  b_dEk = 2 * sigma_E0.

    Rejects particles outside the separatrix.
    Returns (zd, dEk, bucket_grid).
    """
    rng = np.random.default_rng(seed)
    a_zd  = 2.0 * sigma_s0 / BETA_C   # semi-axis zd [s]
    b_dEk = 2.0 * sigma_E0            # semi-axis dEk [MeV]

    # Build bucket grid for H interpolation
    zd_sep = compute_separatrix_zd(G)
    beta2 = 1.0 - (M0C2_MEV / (E0_MEV + M0C2_MEV))**2
    Ek_grid_loc = G["Ek_grid"]
    T_grid_loc = G["T_grid"]
    dTdEk_grad = np.gradient(T_grid_loc, Ek_grid_loc)
    dT_dEk_at_E0 = float(np.interp(G["E0"], Ek_grid_loc, dTdEk_grad))
    eta_lin = dT_dEk_at_E0 * beta2 * (G["E0"] + M0C2_MEV) / G["Tc"]
    Fsep = np.cos(G["fi0"]) - 0.5*(np.pi - 2.0*abs(G["fi0"]))*np.sin(abs(G["fi0"]))
    dEk_bucket = np.sqrt(max(2.0*G["V_turn"]*beta2*(G["E0"]+M0C2_MEV)*1e6
                              / (np.pi*abs(eta_lin))*Fsep, 0.0)) / 1e6

    q_axis, p_axis, H_map, H_sep, _, _ = build_hamiltonian(
        G, (-zd_sep*1.05, zd_sep*1.05), (-dEk_bucket*1.05, dEk_bucket*1.05),
        Nq=301, Np=301)
    dq = q_axis[1] - q_axis[0]
    dp = p_axis[1] - p_axis[0]

    batch = max(4 * n_part, 20000)
    zd_out = np.empty(n_part, dtype=np.float64)
    dEk_out = np.empty(n_part, dtype=np.float64)
    n_found = 0

    while n_found < n_part:
        # Uniform ellipse sampling: radial scaling of uniform circle
        # g ~ sqrt(U[0,1]) gives uniform area filling of 2D ellipse
        g = np.sqrt(rng.uniform(0.0, 1.0, batch))
        theta = rng.uniform(0.0, 2.0 * np.pi, batch)
        z_trial = a_zd * g * np.cos(theta)
        e_trial = b_dEk * g * np.sin(theta)

        # bilinear interp H for bucket truncation
        fq = np.clip((z_trial - q_axis[0]) / dq, 0.0, len(q_axis) - 1.001)
        fp = np.clip((e_trial - p_axis[0]) / dp, 0.0, len(p_axis) - 1.001)
        iq0 = np.floor(fq).astype(int)
        ip0 = np.floor(fp).astype(int)
        iq1 = np.minimum(iq0 + 1, len(q_axis) - 1)
        ip1 = np.minimum(ip0 + 1, len(p_axis) - 1)
        wq, wp = fq - iq0, fp - ip0
        H_trial = (H_map[iq0, ip0] * (1-wq)*(1-wp) + H_map[iq1, ip0] * wq*(1-wp)
                   + H_map[iq0, ip1] * (1-wq)*wp + H_map[iq1, ip1] * wq*wp)

        accept = (H_trial >= H_sep + 1e-12) & (H_trial < 0.0)
        n_accept = int(np.sum(accept))
        need = n_part - n_found
        take = min(n_accept, need)
        idx = np.where(accept)[0][:take]
        zd_out[n_found:n_found + take] = z_trial[idx]
        dEk_out[n_found:n_found + take] = e_trial[idx]
        n_found += take

    bucket_grid = {"q_axis": q_axis, "p_axis": p_axis,
                   "H_map": H_map, "H_sep": H_sep}
    return zd_out, dEk_out, bucket_grid


def plot_initial_distribution(zd0, dEk0, G, H_map, H_sep,
                              q_axis, p_axis, out_dir):
    """Show initial particles on top of the bucket H contour."""
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    # H contours (trapped region only)
    vmax = 0.0
    vmin = np.nanmin(H_map)
    levels = np.linspace(vmin * 0.9, vmax, 15)
    CS = ax.contour(q_axis * 1e9, p_axis, H_map.T, levels=levels,
                     linewidths=0.5, cmap="viridis", alpha=0.4)
    # Separatrix — thick red solid
    ax.contour(q_axis * 1e9, p_axis, H_map.T, levels=[float(H_sep)],
               colors="#cc0000", linewidths=2.2, linestyles="-")
    ax.plot([], [], "-", color="#cc0000", lw=2.0, label="separatrix")

    # Particles
    ax.plot(zd0 * 1e9, dEk0, ".", ms=0.8, color="#1f77b4", alpha=0.5,
            label=f"N = {len(zd0)}")

    # Fixed points
    fi_s = G["fi0"]
    w0 = G["w0"]
    zd_ufp = (np.pi - fi_s) / w0
    zd_ufp_left = -(np.pi + fi_s) / w0
    ax.plot(0.0, 0.0, "k+", ms=12, mew=2.5)
    ax.plot(zd_ufp * 1e9, 0.0, "kx", ms=10, mew=2.0)
    ax.plot(zd_ufp_left * 1e9, 0.0, "kx", ms=10, mew=2.0)

    sig_s0 = np.std(-BETA_C * zd0)
    ax.set_xlabel(r"$z_d$ [ns]")
    ax.set_ylabel(r"$\Delta E_k$ [MeV]")
    ax.set_title(f"Initial bucket-sampled distribution  |  "
                 rf"$\sigma_s$ = {sig_s0:.2f} m,  N = {len(zd0)}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    f = out_dir / "level5_bucket_initial.png"
    if SAVE_FIGS:
        fig.savefig(f, dpi=300, bbox_inches="tight")
        print(f"[saved] {f}")
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


def plot_evolution(res_bare, res_adaptive, out_dir, f_sc):
    """sigma_s, sigma_E, Es_max vs turn for both modes."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    for res, c, lab in [(res_bare, "#2ca02c", "bare RF"),
                         (res_adaptive, "#1f77b4", "adaptive SC")]:
        axes[0].plot(res["turns"], res["sig_s"], color=c, lw=1.8, label=lab)
        axes[1].plot(res["turns"], res["sig_E"], color=c, lw=1.8, label=lab)
        axes[2].plot(res["turns"], np.array(res["Es_max"]) / 1e3,
                     color=c, lw=1.8, label=lab)

    axes[0].set_xlabel("turn")
    axes[0].set_ylabel(r"$\sigma_s$ [m]")
    axes[0].set_title("(a) bunch length")
    axes[1].set_xlabel("turn")
    axes[1].set_ylabel(r"$\sigma_{\Delta E_k}$ [MeV]")
    axes[1].set_title("(b) energy spread")
    axes[2].set_xlabel("turn")
    axes[2].set_ylabel(r"$E_{s,\mathrm{max}}$ [kV/m]")
    axes[2].set_title("(c) peak SC field  (per-step histogram)")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    n_prot = f_sc * N_PROTONS
    fig.suptitle(rf"Bucket-sampled initial distribution: "
                 rf"bare RF vs adaptive SC  ($f_{{sc}}$={f_sc}, "
                 rf"N = {n_prot:.2e} p)", y=1.01)
    fig.tight_layout()
    f = out_dir / f"level5_evolution_fsc{f_sc:.2f}.png"
    if SAVE_FIGS:
        fig.savefig(f, dpi=300, bbox_inches="tight")
        print(f"[saved] {f}")
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


def _plot_single_phasespace_snaps(res, out_dir, fname, bucket_grid=None):
    """Phase-space snapshots for one mode, with auto-computed uniform axes."""
    snaps = res["snapshots"]
    turns = sorted(snaps.keys())
    n = len(turns)

    # ---- axis limits from bucket_grid (ensures full separatrix visible) ----
    if bucket_grid is not None:
        zd_ns_lo = bucket_grid["q_axis"][0]  * 1e9
        zd_ns_hi = bucket_grid["q_axis"][-1] * 1e9
        dek_lim  = max(abs(bucket_grid["p_axis"][0]),
                       abs(bucket_grid["p_axis"][-1]))

    ncols = min(n, 5)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.0 * ncols, 4.0 * nrows),
                             squeeze=False)
    for idx, t in enumerate(turns):
        ax = axes[idx // ncols][idx % ncols]

        # ---- bucket reference ----
        if bucket_grid is not None:
            q_axis = bucket_grid["q_axis"]
            p_axis = bucket_grid["p_axis"]
            H_map  = bucket_grid["H_map"]
            H_sep  = bucket_grid["H_sep"]
            # inner H contours (light grey)
            levels = np.linspace(H_sep, 0.0, 10)
            ax.contour(q_axis * 1e9, p_axis, H_map.T, levels=levels,
                       linewidths=0.3, colors="#cccccc", alpha=0.4)
            # full bucket boundary — thick red solid separatrix
            ax.contour(q_axis * 1e9, p_axis, H_map.T, levels=[float(H_sep)],
                       colors="#cc0000", linewidths=2.5, linestyles="-")

        zd, dEk, _ = snaps[t]
        ax.plot(zd * 1e9, dEk, ".", ms=0.8, color="#1f77b4", alpha=0.5)
        if bucket_grid is not None:
            ax.set_xlim(zd_ns_lo, zd_ns_hi)
            ax.set_ylim(-dek_lim, dek_lim)
        ax.set_xlabel(r"$z_d$ [ns]")
        ax.set_title(f"turn {t}")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.set_ylabel(r"$\Delta E_k$ [MeV]")
    # hide unused subplots
    for idx in range(len(turns), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(f"Phase space: {res['mode']}", y=1.01)
    fig.tight_layout()
    f = out_dir / fname
    if SAVE_FIGS:
        fig.savefig(f, dpi=300, bbox_inches="tight")
        print(f"[saved] {f}")
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


def plot_phasespace_overlay(res_bare, res_adaptive, out_dir, f_sc,
                            bucket_grid=None, overlay_turns=(0, 50, 200, 500)):
    """Overlay bare RF (green) and adaptive SC (blue) at key turns,
       with common x/y limits and bucket separatrix for reference."""
    turns = sorted(set(overlay_turns) & set(res_bare["snapshots"].keys())
                   & set(res_adaptive["snapshots"].keys()))
    if not turns:
        return

    # ---- axis limits from bucket_grid (ensures full separatrix visible) ----
    if bucket_grid is not None:
        zd_lo = bucket_grid["q_axis"][0]  * 1e9
        zd_hi = bucket_grid["q_axis"][-1] * 1e9
        dek_hi = max(abs(bucket_grid["p_axis"][0]),
                     abs(bucket_grid["p_axis"][-1]))
    else:
        zd_lo, zd_hi, dek_hi = -200, 200, 14

    n = len(turns)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 5.0), squeeze=False)

    for idx, t in enumerate(turns):
        ax = axes[0][idx]

        # ---- bucket separatrix & H contours as reference ----
        if bucket_grid is not None:
            q_axis = bucket_grid["q_axis"]   # zd [s]
            p_axis = bucket_grid["p_axis"]   # dEk [MeV]
            H_map  = bucket_grid["H_map"]    # shape (Nq, Np)
            H_sep  = bucket_grid["H_sep"]
            # inner H contours (light grey, thin)
            levels = np.linspace(H_sep, 0.0, 10)
            ax.contour(q_axis * 1e9, p_axis, H_map.T, levels=levels,
                       linewidths=0.3, colors="#cccccc", alpha=0.35)
            # full bucket boundary — thick red solid separatrix
            ax.contour(q_axis * 1e9, p_axis, H_map.T, levels=[float(H_sep)],
                       colors="#cc0000", linewidths=2.2, linestyles="-")

        # ---- particles ----
        zd_b, dEk_b, _ = res_bare["snapshots"][t]
        zd_a, dEk_a, _ = res_adaptive["snapshots"][t]
        ax.plot(zd_b * 1e9, dEk_b, ".", ms=0.5, color="#2ca02c", alpha=0.35,
                label="bare RF")
        ax.plot(zd_a * 1e9, dEk_a, ".", ms=0.5, color="#1f77b4", alpha=0.35,
                label="SC")

        # ---- common axis limits ----
        ax.set_xlim(zd_lo, zd_hi)
        ax.set_ylim(-dek_hi, dek_hi)

        ax.set_xlabel(r"$z_d$ [ns]")
        ax.set_title(f"turn {t}")
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.set_ylabel(r"$\Delta E_k$ [MeV]")
            ax.legend(fontsize=7, loc="upper right")

    n_prot = f_sc * N_PROTONS
    fig.suptitle(rf"Bare RF vs SC  |  "
                 rf"$f_{{sc}}$={f_sc}, $N$={n_prot:.2e} p", y=1.005)
    fig.tight_layout()
    f = out_dir / f"level5_phasespace_overlay_fsc{f_sc:.2f}.png"
    if SAVE_FIGS:
        fig.savefig(f, dpi=300, bbox_inches="tight")
        print(f"[saved] {f}")
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


# ================================================================
# CONFIG — edit here
# ================================================================

N_PART = 20000         # macro-particles (more for cleaner histogram Es)
N_TURNS = 200          # tracking turns
N_STEPS_PER_TURN = 1000
F_SC = 0.5             # SC strength factor (N = F_SC * N_PROTONS)
# fill_frac mapping:  1.0->15.7m, 0.5->8.8m, 0.1->3.6m, 0.05->2.5m, 0.02->1.6m
FILL_FRAC = 0.10       # fraction of bucket H-area filled → σs₀≈3.6 m
SMOOTH_SIGMA = 0.08    # Gaussian smooth of histogram lambda [m] (None=raw)
ES_UPDATE_INTERVAL = 10  # update Es every N steps (1=every step, 10=good balance)
                        #  0 = no smoothing (shot noise inherent)
                        #  0.05-0.1 = light smooth, preserves shape
                        #  0.3-0.5 = heavy smooth, effectively Gaussian
SEED = 42
DIST_TYPE = "waterbag"  # "bucket" / "gaussian" / "waterbag" (uniform ellipse in zd-dEk)
GAUSS_SIGMA_S0 = 3.62   # target initial σs [m] for Gaussian
WATERBAG_SIGMA_S0 = 3.54  # target σs [m] for waterbag (matches 3D tracking turn-1)
WATERBAG_SIGMA_E0 = 2.01  # target σE [MeV] for waterbag

# Snap turns for phase-space plots
SNAP_TURNS = (0, 50, 100, 140)
OVERLAY_TURNS = (0, 50, 100, 140)

SHOW_FIGS = True       # plt.show() popup
SAVE_FIGS = True

# ================================================================
# Main
# ================================================================

def main():
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load SEO frequency table ----
    seo_path = REPO_ROOT / "Bmap_FD16" / "resultsSEO_noError" / "SEO_ini.txt"
    Ek_grid, Freq_grid = load_seo_freq_table(str(seo_path))
    T_grid = 1.0 / Freq_grid
    Tc = float(np.interp(E0_MEV, Ek_grid, T_grid))
    f0 = 1.0 / Tc
    w0 = 2.0 * np.pi * f0

    # ---- eta from frequency table ----
    beta2 = BETA**2
    dT_dEk = np.gradient(T_grid, Ek_grid)
    eta = float(np.interp(E0_MEV, Ek_grid, dT_dEk)) * beta2 \
        * (E0_MEV + M0C2_MEV) / Tc
    eta_abs = abs(eta)

    # ---- linear synchrotron frequency ----
    f_s0 = f0 * np.sqrt(eta_abs * V_TURN
                        / (2.0 * np.pi * beta2 * (E0_MEV + M0C2_MEV) * 1e6))

    # ---- bucket parameters ----
    G = {"E0": E0_MEV, "Tc": Tc, "fi0": PHI_S, "w0": w0,
         "V_turn": V_TURN, "Ek_grid": Ek_grid, "T_grid": T_grid}

    zd_sep = compute_separatrix_zd(G)
    # bucket height (linear eta, for reporting only)
    dT_dEk_vals = np.gradient(T_grid, Ek_grid)
    dT_dEk_at_E0 = float(np.interp(E0_MEV, Ek_grid, dT_dEk_vals))
    eta_lin = dT_dEk_at_E0 * beta2 * (E0_MEV + M0C2_MEV) / Tc
    Fsep = np.cos(G["fi0"]) - 0.5*(np.pi - 2.0*abs(G["fi0"]))*np.sin(abs(G["fi0"]))
    dEk_bucket = np.sqrt(max(2.0 * V_TURN * beta2 * (E0_MEV + M0C2_MEV) * 1e6
                              / (np.pi * abs(eta_lin)) * Fsep, 0.0)) / 1e6

    # ---- sample initial distribution ----
    if DIST_TYPE == "bucket":
        print(f"Building bucket Hamiltonian + sampling {N_PART} particles "
              f"(fill fraction = {FILL_FRAC}) ...")
        zd0, dEk0, bucket_grid = sample_bucket_uniform(
            N_PART, G, seed=SEED, fill_frac=FILL_FRAC, Nq=301, Np=301)
    elif DIST_TYPE == "gaussian":
        print(f"Sampling {N_PART} particles "
              f"(Gaussian, sig_s0={GAUSS_SIGMA_S0:.2f} m, "
              f"sig_E0={dEk_bucket*0.16:.2f} MeV, truncated to bucket) ...")
        sigma_E0_gauss = dEk_bucket * 0.16
        zd0, dEk0, bucket_grid = sample_gaussian_truncated(
            N_PART, GAUSS_SIGMA_S0, sigma_E0_gauss, G, seed=SEED)
        if bucket_grid is None:
            raise RuntimeError("Gaussian truncation requires bucket Hamiltonian.")
    elif DIST_TYPE == "waterbag":
        print(f"Sampling {N_PART} particles "
              f"(waterbag, target sig_s0={WATERBAG_SIGMA_S0:.2f} m, "
              f"sig_E0={WATERBAG_SIGMA_E0:.2f} MeV, truncated to bucket) ...")
        zd0, dEk0, bucket_grid = sample_waterbag_truncated(
            N_PART, WATERBAG_SIGMA_S0, WATERBAG_SIGMA_E0, G, seed=SEED)
    else:
        raise ValueError(f"Unknown DIST_TYPE: {DIST_TYPE}")

    s0 = -BETA_C * zd0
    sigma_s0 = float(np.std(s0))
    sigma_E0 = float(np.std(dEk0))
    span_ns = (zd0.max() - zd0.min()) * 1e9

    # ---- SC grid ----
    s_half = max(SC_S_MAX, 6.5 * sigma_s0)
    s_grid = np.linspace(-s_half, s_half, SC_NS + 1)

    # ---- tracking parameters ----
    P = {"Tc": Tc, "w0": w0, "fis": PHI_S, "V_turn": V_TURN,
         "Ek_grid": Ek_grid, "T_grid": T_grid, "f_sc": F_SC}
    n_prot = F_SC * N_PROTONS

    print()
    print("=" * 64)
    print("Level-5: bucket-sampled initial distribution tracking")
    print("=" * 64)
    print(f"E0 = {E0_MEV:.0f} MeV    f0 = {f0/1e6:.4f} MHz")
    print(f"V_turn = {V_TURN/1e3:.0f} kV    phi_s = {np.rad2deg(PHI_S):.0f} deg")
    print(f"f_s0 = {f_s0/1e3:.2f} kHz (T_s = {f0/f_s0:.0f} turns)")
    print(f"Bucket: zd_sep = {zd_sep*1e9:.1f} ns,  "
          f"dEk_bucket = {dEk_bucket:.2f} MeV")
    print(f"Initial distribution: N_part = {N_PART},  "
          f"sigma_s0 = {sigma_s0:.3f} m,  sigma_E0 = {sigma_E0:.3f} MeV,  "
          f"zd span = {span_ns:.1f} ns")
    print(f"SC: f_sc = {F_SC}  ->  N = {n_prot:.2e} protons")
    print(f"SC grid: +/- {s_half:.1f} m,  {SC_NS} bins")
    print(f"Tracking: {N_TURNS} turns,  {N_STEPS_PER_TURN} steps/turn")
    print()

    # ---- precompute convolution kernel (fixed, so once is enough) ----
    s_centers_k = 0.5 * (s_grid[:-1] + s_grid[1:])
    ds_k = s_centers_k[1] - s_centers_k[0]
    precomp_kernel = ds_k * disk_kernel_numba(s_centers_k)
    counts_buffer = np.zeros(SC_NS, dtype=np.float64)
    counts_buffer_summary = np.zeros(SC_NS, dtype=np.float64)

    # ---- precompute uniform lookup table for T(Ek) ----
    E_table_min = E0_MEV - 30.0
    E_table_max = E0_MEV + 30.0
    N_TABLE = 20000   # 0.003 MeV resolution
    dE_tab = (E_table_max - E_table_min) / N_TABLE
    E_tab_vals = np.linspace(E_table_min, E_table_max, N_TABLE + 1)
    T_tab_vals = np.interp(E_tab_vals, Ek_grid, T_grid).astype(np.float64)
    t_lookup = (E_table_min, dE_tab, T_tab_vals)

    # ---- run bare RF ----
    print("--- bare RF (SC off) ---")
    res_bare = track_bunch(
        zd0.copy(), dEk0.copy(), "bare",
        N_TURNS, N_STEPS_PER_TURN, P, s_grid, SNAP_TURNS,
        SMOOTH_SIGMA, precomp_kernel, counts_buffer,
        ES_UPDATE_INTERVAL, t_lookup)
    print(f"    final sigma_s = {res_bare['sig_s'][-1]:.3f} m")
    print(f"    final sigma_E = {res_bare['sig_E'][-1]:.3f} MeV")

    # ---- run adaptive ----
    print(f"--- adaptive SC (Es updated every {ES_UPDATE_INTERVAL} step(s), "
          f"smooth={SMOOTH_SIGMA} m) ---")
    res_adaptive = track_bunch(
        zd0.copy(), dEk0.copy(), "adaptive",
        N_TURNS, N_STEPS_PER_TURN, P, s_grid, SNAP_TURNS,
        SMOOTH_SIGMA, precomp_kernel, counts_buffer,
        ES_UPDATE_INTERVAL, t_lookup)
    print(f"    final sigma_s = {res_adaptive['sig_s'][-1]:.3f} m")
    print(f"    final sigma_E = {res_adaptive['sig_E'][-1]:.3f} MeV")
    print(f"    peak Es_max   = {res_adaptive['Es_max'].max()/1e3:.2f} kV/m")

    # ---- figures ----
    plot_initial_distribution(zd0, dEk0, G,
                              bucket_grid["H_map"], bucket_grid["H_sep"],
                              bucket_grid["q_axis"], bucket_grid["p_axis"],
                              out_dir)
    plot_evolution(res_bare, res_adaptive, out_dir, F_SC)
    _plot_single_phasespace_snaps(res_bare, out_dir,
                                   "level5_phasespace_bareRF.png",
                                   bucket_grid)
    _plot_single_phasespace_snaps(res_adaptive, out_dir,
                                   "level5_phasespace_adaptive.png",
                                   bucket_grid)
    plot_phasespace_overlay(res_bare, res_adaptive, out_dir, F_SC,
                            bucket_grid, OVERLAY_TURNS)

    # ---- summary text ----
    lines = []
    add = lines.append
    add("=" * 60)
    add("Level-5: Bucket-sampled longitudinal tracking summary")
    add("=" * 60)
    add(f"initial sigma_s = {sigma_s0:.3f} m")
    add(f"initial sigma_E = {sigma_E0:.3f} MeV")
    add(f"N_part = {N_PART},  N_protons = {n_prot:.2e} (f_sc = {F_SC})")
    add(f"bucket half-height = {dEk_bucket:.2f} MeV")
    add("")
    add(f"{'':>12} {'bare RF':>12} {'adaptive SC':>12}")
    add(f"{'final sigma_s [m]':>12} {res_bare['sig_s'][-1]:>12.3f} "
        f"{res_adaptive['sig_s'][-1]:>12.3f}")
    add(f"{'final sigma_E [MeV]':>12} {res_bare['sig_E'][-1]:>12.3f} "
        f"{res_adaptive['sig_E'][-1]:>12.3f}")
    add("")
    # slope ratio from actual final Es (smoothed)
    add("")
    C_ref = BETA_C / f0    # circumference [m]
    for lab, res in [("initial", None), ("final", res_adaptive)]:
        if lab == "initial":
            s_part = s0
        else:
            s_part = -BETA_C * res["snapshots"][max(res["snapshots"].keys())][0]
        centers, _, Es_tmp = compute_Es_from_particles(
            s_part, s_grid, Q_TOTAL / N_PART * F_SC, SMOOTH_SIGMA,
            precomp_kernel, counts_buffer_summary)
        i0 = len(centers) // 2
        dEs_ds_0 = np.gradient(Es_tmp, centers)[i0]
        dVsc_dphi = abs(dEs_ds_0) * C_ref**2 / (2.0 * np.pi)
        sr = dVsc_dphi / V_TURN
        sig = float(np.std(s_part))
        add(f"slope ratio ({lab} sigma_s={sig:.2f} m) = {sr:.1f}")

    summary = "\n".join(lines)
    print("\n" + summary)
    summary_file = out_dir / "level5_summary.txt"
    summary_file.write_text(summary, encoding="utf-8")
    print(f"\n[saved] {summary_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
