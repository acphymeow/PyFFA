#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark: 纵向 Gaussian 长弧形束团的纵向场对比

前提：
    求解器中的 total_field_1d_numba 已经改为 uniform-disk kernel，
    且纵向核半径 a_radius = 0.04 m。

本脚本对比：
    1. solver E_s:
        求解器返回的 Ef_map。

    2. direct-sum reference:
        使用求解器返回的 rho_long 和 s_grid，
        按 uniform-disk kernel 重新直接求和。

    3. high-precision theory:
        横向均匀圆盘 + 纵向截断 Gaussian 线密度的高精度积分理论曲线。

注意：
    - 本脚本只画图，不保存图片。
    - 若 solver 中 a_radius 不是 0.04 m，需要同步修改 A_SMOOTH 和 DISK_RADIUS。
"""

import os
from math import erf

import numpy as np
import matplotlib.pyplot as plt
import pyfftw
from numba import set_num_threads

from FFAG_Bunch import FFAG_Bunch
from FFAG_ParasAndConversion import FFAG_ConversionTools


# ============================================================
# 用户参数
# ============================================================

SEO_R_FILE = "./Bmap_FD16/resultsSEO_noError/SEO_r.txt"
SEO_PR_FILE = "./Bmap_FD16/resultsSEO_noError/SEO_pr.txt"

OUT_DIR = "./benchmark_longitudinal_field_uniform_disk"

M0C2_MEV = 938.2720813
EPS0 = 8.8541878128e-12

EK0_MEV = 300.0

N_PARTICLES = 1000000
Q_MACRO = 2e7

NX = 160
NZ = 160
NS = 256

# 横向 Gaussian rms 尺寸
SIGMA_X = 0.020
SIGMA_Z = 0.020

# 纵向 Gaussian 参数
BUNCH_CENTER_FRAC = 0.11
SIGMA_S = 1.5
S_CLIP_SIGMA = 3.5

SEED = 42


# ============================================================
# 纵向核参数：必须和求解器一致
# ============================================================

LONG_KERNEL_MODE = "uniform_disk"

# 求解器中 total_field_1d_numba 的 a_radius 已设为 0.04 m
A_SMOOTH = 0.04
DISK_RADIUS = 0.04

SIGN_CONVENTION = 1.0
THEORY_SIGN = 1.0

# 高精度积分设置
N_REF_INTEGRAL = 200000
INTEGRAL_CHUNK_SIZE = 32


# ============================================================
# Matplotlib
# ============================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# SEO 表读取与几何工具
# ============================================================

def load_seo_r_table(filename):
    raw = np.loadtxt(filename, skiprows=1)
    seo_ek_axis = raw[1:, 0]
    seo_phi_axis = raw[0, 1:]
    seo_r_matrix = raw[1:, 1:]
    return seo_ek_axis, seo_phi_axis, seo_r_matrix


def load_seo_pr_table(filename):
    raw = np.loadtxt(filename, skiprows=1)
    return raw[1:, 1:]


def convert_seo_pr_to_pr0_geom(seo_pr_matrix):
    """
    将 SEO_pr_matrix 从 pr = vr / v 转成几何方向比 pr0 = r'/r。

    中平面近似下：
        pr0 = pr / sqrt(1 - pr^2)
    """
    pr = np.asarray(seo_pr_matrix, dtype=np.float64)
    denom = np.sqrt(np.maximum(1.0 - pr * pr, 1.0e-30))
    return pr / denom


def build_seo_s_matrix(seo_phi_axis, seo_r_matrix, seo_pr0_geom_matrix):
    """
    构造 s(phi) 表：
        h(phi) = sqrt(r^2 + r'^2) = r * sqrt(1 + pr0^2)
        s(phi) = integral h(phi) dphi
    """
    phi = np.asarray(seo_phi_axis, dtype=np.float64)
    rmat = np.asarray(seo_r_matrix, dtype=np.float64)
    pr0mat = np.asarray(seo_pr0_geom_matrix, dtype=np.float64)

    nE, nphi = rmat.shape
    s_matrix = np.zeros((nE, nphi), dtype=np.float64)
    perimeter = np.zeros(nE, dtype=np.float64)

    dphi = np.mean(np.diff(phi))

    for iE in range(nE):
        r = rmat[iE]
        pr0 = pr0mat[iE]
        h = r * np.sqrt(1.0 + pr0 * pr0)

        s = np.zeros(nphi, dtype=np.float64)
        for k in range(1, nphi):
            s[k] = s[k - 1] + 0.5 * (h[k - 1] + h[k]) * dphi

        s_matrix[iE] = s
        perimeter[iE] = np.trapz(h, phi)

    return s_matrix, perimeter


def periodic_interp_phi(phi_query, phi_axis, values):
    phi_axis = np.asarray(phi_axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    phi0 = phi_axis[0]
    period = 2.0 * np.pi

    q = (np.asarray(phi_query) - phi0) % period + phi0

    phi_ext = np.concatenate([phi_axis, [phi_axis[0] + period]])
    val_ext = np.concatenate([values, [values[0]]])

    return np.interp(q, phi_ext, val_ext)


def phi_from_s_mod(s_mod, phi_axis, s_axis, perimeter):
    s_mod = np.asarray(s_mod, dtype=np.float64) % perimeter

    s_ext = np.concatenate([s_axis, [perimeter]])
    phi_ext = np.concatenate([phi_axis, [phi_axis[0] + 2.0 * np.pi]])

    phi = np.interp(s_mod, s_ext, phi_ext)
    return phi % (2.0 * np.pi)


# ============================================================
# 纵向 Gaussian 长弧形束团生成
# ============================================================

def sample_truncated_gaussian(rng, sigma, clip_sigma, size):
    """
    拒绝采样生成截断 Gaussian。
    截断范围为 ± clip_sigma * sigma。
    """
    out = np.empty(size, dtype=np.float64)
    n_done = 0
    limit = clip_sigma * sigma

    while n_done < size:
        n_need = size - n_done
        cand = rng.normal(0.0, sigma, size=max(2 * n_need, 1024))
        cand = cand[np.abs(cand) <= limit]

        n_take = min(len(cand), n_need)
        out[n_done:n_done + n_take] = cand[:n_take]
        n_done += n_take

    return out


def generate_long_bunch_gaussian_s_on_closed_orbit(
    seo_phi_axis,
    r0_phi,
    pr0_phi,
    s_phi,
    perimeter,
    n_particles,
    bunch_center_frac,
    sigma_s,
    s_clip_sigma,
    sigma_x,
    sigma_z,
    ek0_mev,
    seed=1234,
):
    """
    生成沿闭轨弧长 s 呈 Gaussian 分布的长弧形束团。
    横向偏移沿局部法向 e_x 放置。
    """
    rng = np.random.default_rng(seed)

    s_center = bunch_center_frac * perimeter

    s_offset = sample_truncated_gaussian(
        rng=rng,
        sigma=sigma_s,
        clip_sigma=s_clip_sigma,
        size=n_particles,
    )

    s_cont = s_center + s_offset
    s_mod = s_cont % perimeter

    phi0 = phi_from_s_mod(s_mod, seo_phi_axis, s_phi, perimeter)

    r0 = periodic_interp_phi(phi0, seo_phi_axis, r0_phi)
    pr0 = periodic_interp_phi(phi0, seo_phi_axis, pr0_phi)

    x_local = rng.normal(0.0, sigma_x, size=n_particles)
    z_local = rng.normal(0.0, sigma_z, size=n_particles)

    cosf = np.cos(phi0)
    sinf = np.sin(phi0)

    er_x = cosf
    er_y = sinf
    ephi_x = -sinf
    ephi_y = cosf

    # e_x = (e_r - pr0 e_phi) / sqrt(1 + pr0^2)
    norm = np.sqrt(1.0 + pr0 * pr0)
    ex_x = (er_x - pr0 * ephi_x) / norm
    ex_y = (er_y - pr0 * ephi_y) / norm

    x0_lab = r0 * cosf
    y0_lab = r0 * sinf

    xp_lab = x0_lab + x_local * ex_x
    yp_lab = y0_lab + x_local * ex_y

    r_part = np.sqrt(xp_lab * xp_lab + yp_lab * yp_lab)
    phi_part = np.arctan2(yp_lab, xp_lab) % (2.0 * np.pi)

    coords_cyl = np.column_stack([r_part, phi_part, z_local])
    ek_arr = np.full(n_particles, ek0_mev, dtype=np.float64)

    aux = {
        "s_center": s_center,
        "s_offset": s_offset,
        "s_cont": s_cont,
        "s_mod": s_mod,
        "phi0": phi0,
        "r0": r0,
        "x_local_true": x_local,
        "z_local_true": z_local,
        "x_lab": xp_lab,
        "y_lab": yp_lab,
    }

    return coords_cyl, ek_arr, aux


def build_bunch_from_cyl(coords_cyl, ek_arr, q_macro, nx, nz, ns):
    r = coords_cyl[:, 0]
    phi = coords_cyl[:, 1]
    z = coords_cyl[:, 2]

    n = len(r)

    arr = np.zeros((n, 16), dtype=np.float64)
    arr[:, 0] = r
    arr[:, 2] = z
    arr[:, 4] = phi
    arr[:, 5] = ek_arr
    arr[:, 7] = 1.0
    arr[:, 8] = 1.0
    arr[:, 15] = np.arange(n)

    coords_v_boris = FFAG_ConversionTools().ConvertPrzek2Vrzek_boris(arr)

    bunch = FFAG_Bunch(
        coords_v_boris,
        marcosize=q_macro,
        sc_grid_size=(ns, nz, nx, 12.0),
    )

    bunch.Step_SurviveFlag = np.ones(n, dtype=bool)
    return bunch


def attach_seo_tables_to_bunch(
    bunch,
    seo_ek_axis,
    seo_phi_axis,
    seo_r_matrix,
    seo_pr_matrix,
    seo_pr0_geom_matrix,
    seo_s_matrix,
    seo_perimeter,
):
    bunch.SEO_Ek_axis = seo_ek_axis
    bunch.SEO_fi_axis = seo_phi_axis
    bunch.SEO_r_matrix = seo_r_matrix
    bunch.SEO_pr_matrix = seo_pr_matrix
    bunch.SEO_pr0_geom_matrix = seo_pr0_geom_matrix
    bunch.SEO_s_matrix = seo_s_matrix
    bunch.SEO_perimeter = seo_perimeter


# ============================================================
# FFT plan
# ============================================================

def prepare_fft_plan(nx, nz, threads=1):
    """
    保留求解器接口所需的 FFT plan。
    当前 open-boundary padded FFT 通常在求解器内部缓存生成。
    """
    pyfftw.interfaces.cache.enable()

    a_buf = pyfftw.empty_aligned((nz, nx), dtype="complex64")
    b_buf = pyfftw.empty_aligned((nz, nx), dtype="complex64")

    fft_fwd = pyfftw.FFTW(
        a_buf,
        b_buf,
        axes=(0, 1),
        direction="FFTW_FORWARD",
        threads=threads,
        flags=("FFTW_MEASURE",),
    )

    fft_inv = pyfftw.FFTW(
        b_buf,
        a_buf,
        axes=(0, 1),
        direction="FFTW_BACKWARD",
        threads=threads,
        flags=("FFTW_MEASURE",),
    )

    return fft_fwd, fft_inv


# ============================================================
# 求解器返回值解析
# ============================================================

def unpack_solver_output(ret):
    if len(ret) < 18:
        raise RuntimeError(f"solver return length = {len(ret)} < 18，无法解析。")

    return {
        "rho_xz": ret[0],
        "phi": ret[1],
        "Ex": ret[2],
        "Ez": ret[3],
        "xmin": float(ret[4]),
        "xmax": float(ret[5]),
        "zmin": float(ret[6]),
        "zmax": float(ret[7]),
        "BunchLength": float(ret[8]),
        "Ef_map": np.asarray(ret[9], dtype=np.float64),
        "s_grid": np.asarray(ret[10], dtype=np.float64),
        "rho_long": np.asarray(ret[11], dtype=np.float64),
        "coords_r": ret[12],
        "coords_fi": ret[13],
        "coords_x": ret[14],
        "coords_r0": ret[15],
        "coords_z": ret[16],
        "extra_grid": ret[17],
    }


# ============================================================
# uniform-disk direct-sum reference
# ============================================================

def longitudinal_field_reference_uniform_disk(
    s_grid,
    q_s,
    a_radius,
    eps0=EPS0,
    sign_convention=1.0,
):
    """
    基于 slice 总电荷 q_s 直接求和计算纵向场参考值。

    使用横向均匀圆盘轴向场核：

        E_ij =
        q_j / (2*pi*eps0*a^2)
        * sign(r)
        * [1 - |r|/sqrt(r^2+a^2)]

    其中：
        r = s_i - s_j
        a = a_radius

    Parameters
    ----------
    s_grid : ndarray
        纵向 slice 位置 [m]。

    q_s : ndarray
        每个 slice 的总电荷 Q_k [C]。

    a_radius : float
        横向均匀圆盘半径 [m]。
    """
    s_grid = np.asarray(s_grid, dtype=np.float64)
    q_s = np.asarray(q_s, dtype=np.float64)

    ns = len(s_grid)
    Ef = np.zeros(ns, dtype=np.float64)

    if a_radius <= 0.0:
        raise ValueError("a_radius must be positive.")

    a2 = a_radius * a_radius
    coeff = 1.0 / (2.0 * np.pi * eps0 * a2)

    for i in range(ns):
        si = s_grid[i]
        acc = 0.0

        for j in range(ns):
            qj = q_s[j]
            if qj == 0.0:
                continue

            r = si - s_grid[j]
            abs_r = abs(r)

            if r > 0.0:
                sgn = 1.0
            elif r < 0.0:
                sgn = -1.0
            else:
                sgn = 0.0

            acc += qj * sgn * (
                1.0 - abs_r / np.sqrt(r * r + a2)
            )

        Ef[i] = sign_convention * coeff * acc

    return Ef


# ============================================================
# high-precision theory: uniform disk + truncated Gaussian
# ============================================================

def truncated_gaussian_lambda(
    s,
    s_center,
    sigma_s,
    total_charge,
    clip_sigma,
):
    """
    截断 Gaussian 线电荷密度。

    截断范围:
        |s - s_center| <= clip_sigma * sigma_s

    并重新归一化，使截断区间内总电荷为 total_charge。
    """
    s = np.asarray(s, dtype=np.float64)
    u = (s - s_center) / sigma_s

    lam = np.zeros_like(s)

    mask = np.abs(u) <= clip_sigma
    norm_cut = erf(clip_sigma / np.sqrt(2.0))

    prefactor = total_charge / (
        np.sqrt(2.0 * np.pi) * sigma_s * norm_cut
    )

    lam[mask] = prefactor * np.exp(-0.5 * u[mask] * u[mask])

    return lam


def uniform_disk_axis_kernel(zeta, disk_radius, eps0=EPS0):
    """
    横向均匀圆盘在轴线上的纵向电场核。

    K_a(zeta)
    =
    1/(2*pi*eps0*a^2)
    * sign(zeta)
    * [1 - |zeta|/sqrt(zeta^2+a^2)]
    """
    zeta = np.asarray(zeta, dtype=np.float64)
    a = float(disk_radius)

    if a <= 0.0:
        raise ValueError("disk_radius must be positive.")

    abs_z = np.abs(zeta)

    K = (
        1.0 / (2.0 * np.pi * eps0 * a * a)
        * np.sign(zeta)
        * (1.0 - abs_z / np.sqrt(zeta * zeta + a * a))
    )

    return K


def trapz_integral(y, x, axis=-1):
    """
    兼容不同 numpy 版本。
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def longitudinal_field_theory_high_precision(
    s_eval,
    s_center,
    sigma_s,
    total_charge,
    disk_radius,
    clip_sigma,
    n_ref=200000,
    eps0=EPS0,
    theory_sign=1.0,
    chunk_size=32,
):
    """
    高精度积分计算纵向理论场：

        E_s(s) = ∫ lambda(s') K_a(s - s') ds'

    其中 lambda(s') 为截断 Gaussian 线密度，
    K_a 为横向均匀圆盘轴向场核。
    """
    s_eval = np.asarray(s_eval, dtype=np.float64)

    s_min = s_center - clip_sigma * sigma_s
    s_max = s_center + clip_sigma * sigma_s

    s_ref = np.linspace(s_min, s_max, n_ref)

    lambda_ref = truncated_gaussian_lambda(
        s_ref,
        s_center=s_center,
        sigma_s=sigma_s,
        total_charge=total_charge,
        clip_sigma=clip_sigma,
    )

    E_eval = np.zeros_like(s_eval)

    for i0 in range(0, len(s_eval), chunk_size):
        i1 = min(i0 + chunk_size, len(s_eval))

        zeta = s_eval[i0:i1, None] - s_ref[None, :]

        K = uniform_disk_axis_kernel(
            zeta,
            disk_radius=disk_radius,
            eps0=eps0,
        )

        integrand = lambda_ref[None, :] * K

        E_eval[i0:i1] = trapz_integral(integrand, s_ref, axis=1)

    return theory_sign * E_eval, s_ref, lambda_ref


# ============================================================
# 误差与绘图
# ============================================================

def robust_relative_l2(num, ref, floor=1.0e-30):
    num = np.asarray(num, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return np.sqrt(np.sum((num - ref) ** 2) / max(np.sum(ref ** 2), floor))


def plot_longitudinal_field_high_precision_theory(
    sol,
    sigma_s,
    a_radius,
    clip_sigma,
    n_ref=200000,
    out_file=None,
):
    """
    单图对比：
        solver E_s
        high-precision theory

    绘图风格：
        PIC: 红色散点
        theory: 蓝色粗线
    """
    s_grid = np.asarray(sol["s_grid"], dtype=np.float64)
    q_s = np.asarray(sol["rho_long"], dtype=np.float64)
    Ef_num = np.asarray(sol["Ef_map"], dtype=np.float64)

    if np.sum(q_s) > 0.0:
        s_center = float(np.sum(s_grid * q_s) / np.sum(q_s))
    else:
        s_center = float(np.mean(s_grid))

    total_charge = float(np.sum(q_s))

    Ef_ref = longitudinal_field_reference_uniform_disk(
        s_grid=s_grid,
        q_s=q_s,
        a_radius=a_radius,
        eps0=EPS0,
        sign_convention=SIGN_CONVENTION,
    )

    Ef_theory, s_ref, lambda_ref = longitudinal_field_theory_high_precision(
        s_eval=s_grid,
        s_center=s_center,
        sigma_s=sigma_s,
        total_charge=total_charge,
        disk_radius=a_radius,
        clip_sigma=clip_sigma,
        n_ref=n_ref,
        eps0=EPS0,
        theory_sign=THEORY_SIGN,
        chunk_size=INTEGRAL_CHUNK_SIZE,
    )

    s_rel = s_grid - s_center
    mask = np.abs(s_rel) <= clip_sigma * sigma_s

    rel_err_solver_ref = robust_relative_l2(Ef_num[mask], Ef_ref[mask])
    rel_err_solver_theory = robust_relative_l2(Ef_num[mask], Ef_theory[mask])

    print("\n========== Longitudinal field: uniform-disk benchmark ==========")
    print(f"total_charge = {total_charge:.6e} C")
    print(f"s_center     = {s_center:.6e} m")
    print(f"sigma_s      = {sigma_s:.6e} m")
    print(f"clip_sigma   = {clip_sigma:.3f}")
    print(f"a_radius     = {a_radius:.6e} m")
    print(f"n_ref        = {n_ref}")
    print(f"THEORY_SIGN  = {THEORY_SIGN:+.1f}")
    print(f"max |solver E_s| = {np.max(np.abs(Ef_num)):.6e} V/m")
    print(f"max |direct ref| = {np.max(np.abs(Ef_ref)):.6e} V/m")
    print(f"max |theory|     = {np.max(np.abs(Ef_theory)):.6e} V/m")
    print(f"solver vs direct-sum relL2 = {rel_err_solver_ref:.6e}")
    print(f"solver vs high-precision theory relL2 = {rel_err_solver_theory:.6e}")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    # PIC 数值结果：小散点，避免遮挡理论曲线
    ax.scatter(
        s_rel,
        Ef_num,
        s=25.0,
        color="#d62728",
        alpha=0.78,
        edgecolors="none",
        label=r"2.5D PIC $E_s$",
        zorder=3,
    )

    # 理论结果：粗实线
    ax.plot(
        s_rel,
        Ef_theory,
        color="#1f4aff",
        lw=3.0,
        solid_capstyle="round",
        label=r"theory $E_s$",
        zorder=4,
    )

    ax.set_xlabel(r"$s-s_c$ [m]", fontsize=26)
    ax.set_ylabel(r"$E_s$ [V/m]", fontsize=26)

    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.grid(True, alpha=0.22, linewidth=0.8)

    # 坐标范围留一点边界
    x_pad = 0.04 * (np.max(s_rel) - np.min(s_rel))
    y_min = min(np.min(Ef_num), np.min(Ef_theory))
    y_max = max(np.max(Ef_num), np.max(Ef_theory))
    y_pad = 0.08 * (y_max - y_min)

    ax.set_xlim(np.min(s_rel) - x_pad, np.max(s_rel) + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # legend 去框、透明
    ax.legend(
        fontsize=22,
        loc="upper left",
        frameon=False,
        handlelength=2.2,
        borderpad=0.3,
        labelspacing=0.5,
    )

    plt.tight_layout()

    if out_file is not None:
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"    saved: {out_file}")

    plt.show()

    return Ef_ref, Ef_theory

# ============================================================
# 主程序
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    set_num_threads(1)

    print("[1] Load SEO tables...")
    seo_ek_axis, seo_phi_axis, seo_r_matrix = load_seo_r_table(SEO_R_FILE)
    seo_pr_matrix = load_seo_pr_table(SEO_PR_FILE)
    seo_pr0_geom_matrix = convert_seo_pr_to_pr0_geom(seo_pr_matrix)

    seo_s_matrix, seo_perimeter = build_seo_s_matrix(
        seo_phi_axis,
        seo_r_matrix,
        seo_pr0_geom_matrix,
    )

    iE = int(np.argmin(np.abs(seo_ek_axis - EK0_MEV)))
    ek_ref = float(seo_ek_axis[iE])
    gamma_rel = 1.0 + EK0_MEV / M0C2_MEV

    print(f"    target Ek0 = {EK0_MEV:.6f} MeV")
    print(f"    nearest SEO row = {iE}, Ek = {ek_ref:.6f} MeV")
    print(f"    gamma = {gamma_rel:.8f}")

    r0_phi = seo_r_matrix[iE].copy()
    pr0_phi = seo_pr0_geom_matrix[iE].copy()
    s_phi = seo_s_matrix[iE].copy()
    perimeter = float(seo_perimeter[iE])

    print("[2] Generate Gaussian long arc-shaped bunch...")
    coords_cyl, ek_arr, aux = generate_long_bunch_gaussian_s_on_closed_orbit(
        seo_phi_axis=seo_phi_axis,
        r0_phi=r0_phi,
        pr0_phi=pr0_phi,
        s_phi=s_phi,
        perimeter=perimeter,
        n_particles=N_PARTICLES,
        bunch_center_frac=BUNCH_CENTER_FRAC,
        sigma_s=SIGMA_S,
        s_clip_sigma=S_CLIP_SIGMA,
        sigma_x=SIGMA_X,
        sigma_z=SIGMA_Z,
        ek0_mev=EK0_MEV,
        seed=SEED,
    )

    print(f"    N particles = {N_PARTICLES}")
    print(f"    sigma_x = {SIGMA_X:.6e} m")
    print(f"    sigma_z = {SIGMA_Z:.6e} m")
    print(f"    sigma_s = {SIGMA_S:.6e} m")
    print(f"    s clip = ±{S_CLIP_SIGMA:.2f} sigma_s")
    print(f"    generated s span = {np.max(aux['s_cont']) - np.min(aux['s_cont']):.6e} m")
    print(f"    longitudinal kernel radius a = {A_SMOOTH:.6e} m")

    print("[3] Build FFAG_Bunch...")
    bunch = build_bunch_from_cyl(
        coords_cyl,
        ek_arr,
        q_macro=Q_MACRO,
        nx=NX,
        nz=NZ,
        ns=NS,
    )

    attach_seo_tables_to_bunch(
        bunch,
        seo_ek_axis=seo_ek_axis,
        seo_phi_axis=seo_phi_axis,
        seo_r_matrix=seo_r_matrix,
        seo_pr_matrix=seo_pr_matrix,
        seo_pr0_geom_matrix=seo_pr0_geom_matrix,
        seo_s_matrix=seo_s_matrix,
        seo_perimeter=seo_perimeter,
    )

    print("[4] Prepare FFT interface plan...")
    fft_fwd, fft_inv = prepare_fft_plan(NX, NZ, threads=1)

    print("[5] Run 2.5D SC solver...")
    ret = bunch.Build_2_5D_SliceGrid_sbs_new_fixProjection(fft_fwd, fft_inv)
    sol = unpack_solver_output(ret)

    print("[OK] Solver finished.")
    print(f"    rho_long shape = {sol['rho_long'].shape}")
    print(f"    Ef_map shape   = {sol['Ef_map'].shape}")
    print(f"    s_grid shape   = {sol['s_grid'].shape}")
    print(f"    BunchLength    = {sol['BunchLength']:.6e} m")

    print("[6] Plot longitudinal field with high-precision theory...")
    plot_longitudinal_field_high_precision_theory(
        sol=sol,
        sigma_s=SIGMA_S,
        a_radius=A_SMOOTH,
        clip_sigma=S_CLIP_SIGMA,
        n_ref=N_REF_INTEGRAL,
        outfile = OUT_DIR
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
