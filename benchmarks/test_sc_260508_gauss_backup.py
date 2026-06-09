#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark: 纵向 Gaussian 长弧形束团切片横向场 vs 解析二维 Gaussian 横向场

验证目标：
    1. 束团沿闭轨弧长 s 呈 Gaussian 线密度分布；
    2. 每个 slice 内横向分布为圆对称 Gaussian；
    3. 不同 slice 的横向场幅值应随线电荷密度 lambda_k 改变；
    4. 若将横向场除以 lambda_k，不同 slice 的归一化场应基本重合；
    5. 这用于验证：
       - 长弧形束团的局部坐标重建；
       - 沿 s 的切片；
       - 非固定闭轨几何下的局部横向场求解；
       - 不同纵向线密度下场幅值的正确缩放。
"""

import os
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

OUT_DIR = "./benchmark_long_arc_gaussian_s"

M0C2_MEV = 938.2720813
EPS0 = 8.8541878128e-12

EK0_MEV = 300.0

N_PARTICLES = 200000
Q_MACRO = 1.0e8

NX = 160
NZ = 160
NS = 180

SIGMA_X = 0.010
SIGMA_Z = 0.010

# 纵向 Gaussian 参数
BUNCH_CENTER_FRAC = 0.11
SIGMA_S = 1.5
S_CLIP_SIGMA = 3.5

SEED = 42

# 若 compute_field_from_potential_numba_SBS 已经对 Ex/Ez 加了 1/gamma^2，
# 则保持 True；若输出的是未修正静电场，则设为 False。
FIELD_INCLUDES_GAMMA_REDUCTION = True

# 抽样 slice 分位点，按非空 slice 的累计电荷分布选取
SLICE_FRACTIONS = [0.03, 0.20, 0.50]

# 保存图片设置
SAVE_DPI = 300


# ============================================================
# Matplotlib
# ============================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = SAVE_DPI
plt.rcParams["savefig.bbox"] = "tight"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ============================================================
# 统一绘图风格与 slice 配色
# ============================================================

SLICE_COLORS = [
    "#d62728",  # S1 red
    "#ff7f0e",  # S2 orange
    "#2ca02c",  # S3 green
    "#9467bd",  # S4 purple
    "#8c564b",  # S5 brown
    "#e377c2",  # S6 pink
    "#17becf",  # S7 cyan
]

PARTICLE_COLOR = "#1f77b4"
ORBIT_COLOR = "0.75"


def get_slice_color(m):
    return SLICE_COLORS[m % len(SLICE_COLORS)]


# ============================================================
# 曲线样式：颜色表示 slice，线型/marker 表示 num/theory
# ============================================================

NUM_LW = 3.0
THEORY_LW = 2.0
THEORY_MARKER_SIZE = 12
THEORY_MARK_EVERY = 10
THEORY_DASH = (0, (5.0, 2.5))


def save_png(fig, out_file, dpi=SAVE_DPI):
    """
    只保存 PNG。
    out_file 可以带 .png，也可以不带后缀。
    """
    if out_file is None:
        return

    root, ext = os.path.splitext(out_file)
    if ext.lower() != ".png":
        out_file = root + ".png"

    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    print(f"    saved: {out_file}")


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
    denom = np.sqrt(np.maximum(1.0 - pr * pr, 1e-30))
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
    简单拒绝采样，生成截断 Gaussian。
    截断范围为 +- clip_sigma * sigma。
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

    横向偏移沿局部法向 e_x 放置，而不是简单 r = r0 + x。
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
    保留旧接口需要的 FFT plan。
    当前 open boundary padded FFT 通常在 Build_2_5D_SliceGrid_sbs_new_fixProjection
    内部缓存生成；这里主要用于兼容函数形参。
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
        "Ef_map": ret[9],
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
# 理论公式
# ============================================================

def gaussian_round_beam_field_1d(axis, sigma, lambda_k, gamma_rel=1.0, apply_gamma=True):
    """
    圆对称二维 Gaussian 线电荷横向电场一维截线：

        E_u(u) = lambda / (2 pi eps0 u) * [1 - exp(-u^2/(2 sigma^2))]

    若 apply_gamma=True，则乘 1/gamma^2。
    """
    u = np.asarray(axis, dtype=np.float64)
    abs_u = np.abs(u)

    E = np.zeros_like(u)
    mask = abs_u > 1e-30

    rr = abs_u[mask]
    E_mag = lambda_k / (2.0 * np.pi * EPS0 * rr) * (
        1.0 - np.exp(-rr * rr / (2.0 * sigma * sigma))
    )

    if apply_gamma:
        E_mag = E_mag / (gamma_rel * gamma_rel)

    E[mask] = E_mag * np.sign(u[mask])
    E[~mask] = 0.0
    return E


def gaussian_lambda_shape(s, s_center, sigma_s, total_charge):
    """
    连续 Gaussian 线密度理论形状。
    """
    s = np.asarray(s, dtype=np.float64)
    lam = total_charge / (np.sqrt(2.0 * np.pi) * sigma_s) * np.exp(
        -0.5 * ((s - s_center) / sigma_s) ** 2
    )
    return lam


def robust_relative_l2(num, ref, floor=1e-30):
    num = np.asarray(num, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return np.sqrt(np.sum((num - ref) ** 2) / max(np.sum(ref ** 2), floor))


# ============================================================
# Slice 选择
# ============================================================

def choose_slices_by_charge_quantile(sol, fractions):
    """
    根据纵向电荷累计分布选择 slice。
    对 Gaussian 纵向分布更合理：可以自然选到头部、中心、尾部。
    """
    qk = np.asarray(sol["rho_long"], dtype=np.float64)
    positive = qk > 0.0
    if not np.any(positive):
        raise RuntimeError("所有 slice 的纵向电荷均为 0，无法选择对比 slice。")

    cdf = np.cumsum(qk)
    if cdf[-1] <= 0.0:
        raise RuntimeError("总纵向电荷 <= 0，无法选择对比 slice。")
    cdf = cdf / cdf[-1]

    out = []
    for f in fractions:
        k = int(np.searchsorted(cdf, f))
        k = max(0, min(len(qk) - 1, k))
        if qk[k] > 0.0 and k not in out:
            out.append(k)

    return out


# ============================================================
# 绘图
# ============================================================

def plot_global_bunch_and_slices(
    aux,
    seo_phi_axis,
    r0_phi,
    s_phi,
    perimeter,
    selected_s_values,
    out_file=None,
    max_scatter=16000,
    focus_on_bunch=True,
):
    """
    绘制全局平面中的长弧形束团，并标出选定的纵向 slice 位置。
    """
    x_lab = aux["x_lab"]
    y_lab = aux["y_lab"]

    n = len(x_lab)
    if n > max_scatter:
        idx = np.linspace(0, n - 1, max_scatter).astype(int)
    else:
        idx = np.arange(n)

    phi_plot = np.linspace(0.0, 2.0 * np.pi, 2500)
    r_plot = periodic_interp_phi(phi_plot, seo_phi_axis, r0_phi)
    x_orb = r_plot * np.cos(phi_plot)
    y_orb = r_plot * np.sin(phi_plot)

    fig, ax = plt.subplots(figsize=(6.0, 5.5/7*6))

    ax.plot(
        x_orb,
        y_orb,
        color=ORBIT_COLOR,
        lw=2.0,
        alpha=0.9,
        label="reference \nclosed orbit",
        zorder=1,
    )

    ax.scatter(
        x_lab[idx],
        y_lab[idx],
        s=5,
        color=PARTICLE_COLOR,
        alpha=0.38,
        edgecolors="none",
        # label="particles",
        zorder=2,
    )

    for m, s_val in enumerate(selected_s_values):
        phi_k = phi_from_s_mod(s_val, seo_phi_axis, s_phi, perimeter)
        r_k = periodic_interp_phi(phi_k, seo_phi_axis, r0_phi)

        xk = r_k * np.cos(phi_k)
        yk = r_k * np.sin(phi_k)

        c = get_slice_color(m)

        ax.plot(
            xk,
            yk,
            "o",
            ms=10,
            color=c,
            markeredgecolor="black",
            markeredgewidth=0.9,
            label=f"S{m + 1}",
            zorder=4,
        )

        ax.text(
            xk + 0.08,
            yk + 0.05,
            f"S{m + 1}",
            fontsize=20,
            color="black",
            zorder=5,
        )

    # ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]", fontsize=22)
    ax.set_ylabel("Y [m]", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=22)
    # ax.legend(fontsize=30)
    # ax.grid(True, alpha=0.25)

    if focus_on_bunch:
        x_min, x_max = np.min(x_lab[idx]), np.max(x_lab[idx])
        y_min, y_max = np.min(y_lab[idx]), np.max(y_lab[idx])

        dx_view = x_max - x_min
        dy_view = y_max - y_min

        margin_x = 0.15 * dx_view + 0.30
        margin_y = 0.15 * dy_view + 0.30

        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)

    ax.legend(fontsize=18, loc="best", frameon=False)
    plt.tight_layout()
    import os
    os.makedirs("./prabfig", exist_ok=True)
    # plt.savefig("./prabfig/benchmark_bunch.png", dpi=300, bbox_inches="tight")
    # save_png(fig, out_file)


def plot_multi_slice_Ex_comparison(
    sol,
    selected_slices,
    gamma_rel,
    sigma,
    out_file=None,
    xlim_sigma=6.0,
):
    """
    绘制多个 slice 的 Ex(x, z=0) 与解析 Gaussian 场对比。

    颜色表示 slice：
        S1, S2, S3, ...

    线型表示数据来源：
        num    : 粗实线
        theory : 虚线 + 空心圆 marker
    """
    Ex = np.asarray(sol["Ex"], dtype=np.float64)

    ns, nz, nx = Ex.shape

    dx = (sol["xmax"] - sol["xmin"]) / nx
    dz = (sol["zmax"] - sol["zmin"]) / nz

    x_axis = sol["xmin"] + np.arange(nx) * dx
    z_axis = sol["zmin"] + np.arange(nz) * dz

    iz0 = int(np.argmin(np.abs(z_axis)))

    qk = np.asarray(sol["rho_long"], dtype=np.float64)
    delta_s_eff = sol["BunchLength"] / ns

    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    print("\n========== Gaussian-s multi-slice Ex benchmark ==========")
    print(f"grid: ns={ns}, nz={nz}, nx={nx}")
    print(f"dx = {dx:.6e}, dz = {dz:.6e}")
    print(f"BunchLength = {sol['BunchLength']:.6e} m")
    print(f"delta_s_eff = {delta_s_eff:.6e} m")
    print(f"gamma_rel = {gamma_rel:.8f}")
    print(f"apply 1/gamma^2 in theory = {FIELD_INCLUDES_GAMMA_REDUCTION}")

    for m, k in enumerate(selected_slices):
        c = get_slice_color(m)

        lambda_k = qk[k] / delta_s_eff
        Ex_num = Ex[k, iz0, :]

        Ex_th = gaussian_round_beam_field_1d(
            x_axis,
            sigma=sigma,
            lambda_k=lambda_k,
            gamma_rel=gamma_rel,
            apply_gamma=FIELD_INCLUDES_GAMMA_REDUCTION,
        )

        mask_x = np.abs(x_axis) < 3.0 * sigma
        err_ex = robust_relative_l2(Ex_num[mask_x], Ex_th[mask_x])

        print(
            f"slice S{m+1}: k={k:4d}, "
            f"s={sol['s_grid'][k]: .6e} m, "
            f"Q_k={qk[k]: .6e} C, "
            f"lambda_k={lambda_k: .6e} C/m, "
            f"relL2 Ex={err_ex:.4e}"
        )

        ax.plot(
            x_axis * 1e3,
            Ex_num / 1e3,
            lw=NUM_LW,
            color=c,
            linestyle="-",
            solid_capstyle="round",
            label=f"S{m + 1}: 2.5D PIC",
            zorder=2,
        )

        ax.plot(
            x_axis * 1e3,
            Ex_th / 1e3,
            lw=THEORY_LW,
            color=c,
            linestyle=THEORY_DASH,
            marker="o",
            markersize=THEORY_MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=c,
            markeredgewidth=1.0,
            markevery=THEORY_MARK_EVERY,
            label=f"S{m + 1}: theory",
            zorder=3,
        )

    ax.set_xlabel("x [mm]", fontsize=22)
    ax.set_ylabel(r"$E_x$ [kV/m]", fontsize=22)
    # ax.set_title(r"Local $E_x$ along $z=0$")
    # ax.grid(True, alpha=0.30)

    if xlim_sigma is not None:
        ax.set_xlim(-xlim_sigma * sigma * 1e3, xlim_sigma * sigma * 1e3)

    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    import os
    os.makedirs("./prabfig", exist_ok=True)
    plt.savefig("./prabfig/benchmark_bunch_ex.png", dpi=300, bbox_inches="tight")
    # save_png(fig, out_file)


def plot_multi_slice_Ez_comparison(
    sol,
    selected_slices,
    gamma_rel,
    sigma,
    out_file=None,
    xlim_sigma=6.0,
):
    """
    绘制多个 slice 的 Ez(x=0, z) 与解析 Gaussian 场对比。

    颜色表示 slice：
        S1, S2, S3, ...

    线型表示数据来源：
        num    : 粗实线
        theory : 虚线 + 空心圆 marker
    """
    Ez = np.asarray(sol["Ez"], dtype=np.float64)

    ns, nz, nx = Ez.shape

    dx = (sol["xmax"] - sol["xmin"]) / nx
    dz = (sol["zmax"] - sol["zmin"]) / nz

    x_axis = sol["xmin"] + np.arange(nx) * dx
    z_axis = sol["zmin"] + np.arange(nz) * dz

    ix0 = int(np.argmin(np.abs(x_axis)))

    qk = np.asarray(sol["rho_long"], dtype=np.float64)
    delta_s_eff = sol["BunchLength"] / ns

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    print("\n========== Gaussian-s multi-slice Ez benchmark ==========")
    print(f"grid: ns={ns}, nz={nz}, nx={nx}")
    print(f"dx = {dx:.6e}, dz = {dz:.6e}")
    print(f"BunchLength = {sol['BunchLength']:.6e} m")
    print(f"delta_s_eff = {delta_s_eff:.6e} m")
    print(f"gamma_rel = {gamma_rel:.8f}")
    print(f"apply 1/gamma^2 in theory = {FIELD_INCLUDES_GAMMA_REDUCTION}")

    for m, k in enumerate(selected_slices):
        c = get_slice_color(m)

        lambda_k = qk[k] / delta_s_eff
        Ez_num = Ez[k, :, ix0]

        Ez_th = gaussian_round_beam_field_1d(
            z_axis,
            sigma=sigma,
            lambda_k=lambda_k,
            gamma_rel=gamma_rel,
            apply_gamma=FIELD_INCLUDES_GAMMA_REDUCTION,
        )

        mask_z = np.abs(z_axis) < 3.0 * sigma
        err_ez = robust_relative_l2(Ez_num[mask_z], Ez_th[mask_z])

        print(
            f"slice S{m+1}: k={k:4d}, "
            f"s={sol['s_grid'][k]: .6e} m, "
            f"Q_k={qk[k]: .6e} C, "
            f"lambda_k={lambda_k: .6e} C/m, "
            f"relL2 Ez={err_ez:.4e}"
        )

        ax.plot(
            z_axis * 1e3,
            Ez_num / 1e3,
            lw=NUM_LW,
            color=c,
            linestyle="-",
            solid_capstyle="round",
            label=f"S{m + 1}: 2.5D PIC",
            zorder=2,
        )

        ax.plot(
            z_axis * 1e3,
            Ez_th / 1e3,
            lw=THEORY_LW,
            color=c,
            linestyle=THEORY_DASH,
            marker="o",
            markersize=THEORY_MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=c,
            markeredgewidth=1.0,
            markevery=THEORY_MARK_EVERY,
            label=f"S{m + 1}: theory",
            zorder=3,
        )

    ax.set_xlabel("z [mm]", fontsize=22)
    ax.set_ylabel(r"$E_z$ [kV/m]", fontsize=22)
    # ax.set_title(r"Local $E_z$ along $x=0$")
    ax.grid(True, alpha=0.30)

    if xlim_sigma is not None:
        ax.set_xlim(-xlim_sigma * sigma * 1e3, xlim_sigma * sigma * 1e3)

    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    import os
    os.makedirs("./prabfig", exist_ok=True)
    plt.savefig("./prabfig/benchmark_bunch_ez.png", dpi=300, bbox_inches="tight")
    # save_png(fig, out_file)


def plot_lambda_distribution(sol, aux, sigma_s, q_macro, out_file=None):
    qk = np.asarray(sol["rho_long"], dtype=np.float64)
    ns = len(qk)
    s_grid = np.asarray(sol["s_grid"], dtype=np.float64)

    delta_s_eff = sol["BunchLength"] / ns
    lambda_num = qk / delta_s_eff

    s_center_est = np.mean(aux["s_cont"])
    total_charge = len(aux["s_cont"]) * q_macro

    lambda_th = gaussian_lambda_shape(
        s_grid,
        s_center=s_center_est,
        sigma_s=sigma_s,
        total_charge=total_charge,
    )

    if np.max(lambda_th) > 0.0 and np.max(lambda_num) > 0.0:
        lambda_th_scaled = lambda_th * (np.max(lambda_num) / np.max(lambda_th))
    else:
        lambda_th_scaled = lambda_th

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        s_grid,
        lambda_num,
        "o-",
        ms=3,
        lw=1.2,
        color=PARTICLE_COLOR,
        label="slice lambda, numerical",
    )
    ax.plot(
        s_grid,
        lambda_th_scaled,
        "--",
        lw=1.8,
        color="black",
        label="Gaussian shape, scaled",
    )
    ax.set_xlabel("s [m]")
    ax.set_ylabel("lambda [C/m]")
    ax.set_title("Longitudinal line-charge density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    save_png(fig, out_file)


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
    print(f"    sigma_s = {SIGMA_S:.6e} m")
    print(f"    s clip = ±{S_CLIP_SIGMA:.2f} sigma_s")
    print(f"    effective generated span ≈ {np.max(aux['s_cont']) - np.min(aux['s_cont']):.6e} m")
    print(f"    sigma_x = {SIGMA_X:.6e} m, sigma_z = {SIGMA_Z:.6e} m")

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

    print("[5] Run final 2.5D SC solver: open boundary + fixed projection...")
    ret = bunch.Build_2_5D_SliceGrid_sbs_new_fixProjection(fft_fwd, fft_inv)
    sol = unpack_solver_output(ret)

    print("[OK] Solver finished.")
    print("    rho_xz shape =", sol["rho_xz"].shape)
    print("    phi shape    =", sol["phi"].shape)
    print("    Ex shape     =", sol["Ex"].shape)
    print("    Ez shape     =", sol["Ez"].shape)
    print(f"    BunchLength  = {sol['BunchLength']:.6e} m")
    print(f"    x range      = [{sol['xmin']:.6e}, {sol['xmax']:.6e}]")
    print(f"    z range      = [{sol['zmin']:.6e}, {sol['zmax']:.6e}]")

    print("[6] Choose representative slices by charge quantile...")
    selected_slices = choose_slices_by_charge_quantile(sol, SLICE_FRACTIONS)
    selected_s_values = [sol["s_grid"][k] for k in selected_slices]
    print("    selected slices =", selected_slices)

    if abs(SIGMA_X - SIGMA_Z) > 1e-12:
        print(
            "[WARNING] 当前横向解析公式采用圆对称 Gaussian，建议 SIGMA_X = SIGMA_Z。"
        )

    # print("[7] Plot longitudinal lambda distribution...")
    # plot_lambda_distribution(
    #     sol=sol,
    #     aux=aux,
    #     sigma_s=SIGMA_S,
    #     q_macro=Q_MACRO,
    #     out_file=os.path.join(OUT_DIR, "01_lambda_distribution.png"),
    # )

    print("[8] Plot global bunch and selected slices...")
    plot_global_bunch_and_slices(
        aux=aux,
        seo_phi_axis=seo_phi_axis,
        r0_phi=r0_phi,
        s_phi=s_phi,
        perimeter=perimeter,
        selected_s_values=selected_s_values,
        out_file=os.path.join(OUT_DIR, "02_global_gaussian_bunch_selected_slices.png"),
    )

    print("[9] Plot multi-slice field comparison...")
    plot_multi_slice_Ex_comparison(
        sol=sol,
        selected_slices=selected_slices,
        gamma_rel=gamma_rel,
        sigma=SIGMA_X,
        out_file=os.path.join(OUT_DIR, "03a_multi_slice_Ex_vs_gaussian_theory.png"),
    )

    plot_multi_slice_Ez_comparison(
        sol=sol,
        selected_slices=selected_slices,
        gamma_rel=gamma_rel,
        sigma=SIGMA_Z,
        out_file=os.path.join(OUT_DIR, "03b_multi_slice_Ez_vs_gaussian_theory.png"),
    )

    print("\nDone.")
    print(f"Figures saved in: {OUT_DIR}")

    plt.show()


if __name__ == "__main__":
    main()
