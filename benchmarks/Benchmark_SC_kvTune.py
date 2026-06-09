#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm
from scipy.signal import savgol_filter
from FFAG_MathTools import Bilinear_interp_2D_vect_uniform

# ============================================================
# 全局字体设置
# ============================================================
# plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["figure.titlesize"] = 26
plt.rcParams["lines.linewidth"] = 2.5

# ============================================================
# 物理常数
# ============================================================
e_charge = 1.602176634e-19
m_p      = 1.67262192369e-27
c_light  = 2.99792458e8
eps0     = 8.8541878128e-12

# ============================================================
# 用户配置
# ============================================================

# ---------- 理论：横向 KV + 纵向 Gaussian ----------
betaR_filepath = "./Bmap_FD16/resultsSEO_noError_16000/BetaFuncR.txt"
betaZ_filepath = "./Bmap_FD16/resultsSEO_noError_16000/BetaFuncZ.txt"

C   = 70.0
T   = 300.0
# Qx0 = 2.68
# Qy0 = 2.11
Qx0 = 3.23
Qy0 = 2.80

N_tot = 2.0e13

# 这里仍沿用你原始变量名。
# 在本脚本中，它被解释为纵向高斯分布的 full length = ±nsigma sigma_s 总长度
L_long_uniform = 12.0

eps_x_rms = 70.0e-6
eps_y_rms = 70.0e-6
nsigma = 4.0

# ---------- FFT 实际 tune ----------
# DATA_DIR0 = "../output/benchmark_SC_tune_kvAna/kv90"
DATA_DIR0 = "../output/NewBenchmark/FFT_Tune_kv70"

# SEO_r_filepath  = "./Bmap/resultsSEO/SEO_r.txt"
# SEO_pr_filepath = "./Bmap/resultsSEO/SEO_pr.txt"
SEO_r_filepath  = "./Bmap_FD16/resultsSEO_noError_16000/SEO_r.txt"
SEO_pr_filepath = "./Bmap_FD16/resultsSEO_noError_16000/SEO_pr.txt"

COL_X, COL_XP = 0, 1
COL_Z, COL_ZP = 2, 3
COL_fi, COL_Ek = 4, 5

PAD_FACTOR = 8
USE_SAVGOL = False
SG_WINDOW, SG_ORDER = 18, 4

# ---------- FFT tune 显示到哪个整数胞元 ----------
QX_CELL_SHIFT = 3.0
QY_CELL_SHIFT = 2.0

# ---------- 理论分布采样 ----------
N_THEORY_MACRO = 80000
THEORY_SEED = 1234

# ---------- 作图范围 ----------
QX_RANGE = [2.3, 3.3]
QY_RANGE = [1.9, 2.9]

# ============================================================
# 理论部分：横向 KV + 纵向 Gaussian
# ============================================================

def beta_gamma_from_T(T_MeV, m0_MeV=938.2720813):
    E_tot = T_MeV + m0_MeV
    gamma = E_tot / m0_MeV
    beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
    return beta, gamma


def lambda_p_gaussian_peak_trunc(N_tot, sigma_z, nsigma=4.0):
    z_cut = nsigma * sigma_z
    norm_trunc = math.erf(z_cut / (math.sqrt(2.0) * sigma_z))
    lambda_p_peak = N_tot / (math.sqrt(2.0 * math.pi) * sigma_z * norm_trunc)
    return lambda_p_peak, norm_trunc


def ExEz_slope_KV_from_rms_lambda(lambda_q,
                                  beta_x_array,
                                  beta_y_array,
                                  eps_x_rms,
                                  eps_y_rms):
    beta_x_array = np.asarray(beta_x_array)
    beta_y_array = np.asarray(beta_y_array)

    sigma_x = np.sqrt(beta_x_array * eps_x_rms)
    sigma_y = np.sqrt(beta_y_array * eps_y_rms)

    sigma_x = np.maximum(sigma_x, 1e-12)
    sigma_y = np.maximum(sigma_y, 1e-12)

    Ex_over_x = lambda_q / (4.0 * math.pi * eps0 * sigma_x * (sigma_x + sigma_y))
    Ey_over_y = lambda_q / (4.0 * math.pi * eps0 * sigma_y * (sigma_x + sigma_y))

    return Ex_over_x, Ey_over_y


def k_sc_arrays_from_rms_lambda(lambda_q,
                                beta_x_array,
                                beta_y_array,
                                eps_x_rms,
                                eps_y_rms,
                                T_MeV,
                                q=e_charge,
                                m=m_p):
    beta_rel, gamma = beta_gamma_from_T(T_MeV)

    Ex_over_x, Ey_over_y = ExEz_slope_KV_from_rms_lambda(
        lambda_q,
        beta_x_array,
        beta_y_array,
        eps_x_rms,
        eps_y_rms
    )

    factor = -q / (gamma**3 * m * beta_rel * beta_rel * c_light * c_light)

    k_sc_x = factor * Ex_over_x
    k_sc_y = factor * Ey_over_y

    return k_sc_x, k_sc_y


def deltaQ_from_cell_k_sc(fi_deg_array, beta_array, k_sc_array, C, N_cells):
    fi_deg_array = np.asarray(fi_deg_array)
    beta_array   = np.asarray(beta_array)
    k_sc_array   = np.asarray(k_sc_array)

    fi_cell_min = fi_deg_array[0]
    fi_cell_max = fi_deg_array[-1]
    d_fi_deg = fi_cell_max - fi_cell_min

    L_cell = C * (d_fi_deg / 360.0)
    s_cell = (fi_deg_array - fi_cell_min) / d_fi_deg * L_cell

    integrand = beta_array * k_sc_array

    I_cell = np.trapz(integrand, s_cell)
    I_ring = N_cells * I_cell

    dQ = I_ring / (4.0 * math.pi)

    return dQ


def compute_theory_kv_gaussian_peak():
    betaR_data = np.loadtxt(betaR_filepath, skiprows=1)
    betaZ_data = np.loadtxt(betaZ_filepath, skiprows=1)

    fi_deg = betaR_data[0, 1:]
    beta_x = betaR_data[2, 1:]
    beta_y = betaZ_data[2, 1:]

    fi_span_deg = fi_deg[-1] - fi_deg[0]
    N_cells = int(round(360.0 / fi_span_deg))

    # 纵向高斯：L_long_uniform 被解释为 ±nsigma sigma_s 的总长度
    sigma_s = L_long_uniform / (2.0 * nsigma)

    lambda_p_peak, norm_trunc = lambda_p_gaussian_peak_trunc(
        N_tot,
        sigma_s,
        nsigma=nsigma
    )

    lambda_q_peak = lambda_p_peak * e_charge

    k_sc_x, k_sc_y = k_sc_arrays_from_rms_lambda(
        lambda_q_peak,
        beta_x,
        beta_y,
        eps_x_rms,
        eps_y_rms,
        T_MeV=T
    )

    dQ_x_max = deltaQ_from_cell_k_sc(
        fi_deg,
        beta_x,
        k_sc_x,
        C,
        N_cells
    )

    dQ_y_max = deltaQ_from_cell_k_sc(
        fi_deg,
        beta_y,
        k_sc_y,
        C,
        N_cells
    )

    Qx_max = Qx0 + dQ_x_max
    Qy_max = Qy0 + dQ_y_max

    print("=== 理论：横向 KV + 纵向 Gaussian 峰值频移 ===")
    print(f"sigma_s = {sigma_s:.6f} m")
    print(f"lambda_p_peak = {lambda_p_peak:.6e} particles/m")
    print(f"lambda_q_peak = {lambda_q_peak:.6e} C/m")
    print(f"dQx_max = {dQ_x_max:.8f}")
    print(f"dQy_max = {dQ_y_max:.8f}")
    print(f"Bare tune = ({Qx0:.8f}, {Qy0:.8f})")
    print(f"Max SC tune shift point = ({Qx_max:.8f}, {Qy_max:.8f})")

    return {
        "bare": (Qx0, Qy0),
        "gaussian_peak": (Qx_max, Qy_max),
        "dQ": (dQ_x_max, dQ_y_max),
        "sigma_s": sigma_s,
        "norm_trunc": norm_trunc,
        "lambda_p_peak": lambda_p_peak,
        "lambda_q_peak": lambda_q_peak,
    }


def sample_truncated_gaussian(n, sigma, nsigma=4.0, seed=1234):
    rng = np.random.default_rng(seed)

    s_cut = nsigma * sigma
    out = []
    n_left = n

    while n_left > 0:
        trial = rng.normal(0.0, sigma, size=max(4 * n_left, 2000))
        trial = trial[np.abs(trial) <= s_cut]

        if len(trial) == 0:
            continue

        take = trial[:n_left]
        out.append(take)
        n_left -= len(take)

    return np.concatenate(out)


def compute_theory_kv_gaussian_distribution(theory_dict,
                                            n_macro=80000,
                                            seed=1234):
    Qx0, Qy0 = theory_dict["bare"]
    dQx_max, dQy_max = theory_dict["dQ"]
    sigma_s = theory_dict["sigma_s"]

    # 按粒子纵向高斯分布抽样
    s = sample_truncated_gaussian(
        n_macro,
        sigma_s,
        nsigma=nsigma,
        seed=seed
    )

    # 线密度相对于峰值的比例
    u = np.exp(-0.5 * (s / sigma_s)**2)

    # KV 横向模型下，频移只随纵向线密度缩放
    qx_th = Qx0 + dQx_max * u
    qy_th = Qy0 + dQy_max * u

    return qx_th, qy_th, s, u


def theory_u_pdf(u, nsigma=4.0):
    """
    对纵向截断高斯分布，u = lambda(s)/lambda_peak = exp[-s^2/(2 sigma_s^2)]。

    理论概率密度为：
    p(u) = 1 / [erf(nsigma/sqrt(2)) * sqrt(pi * (-ln u))]

    定义域：
    u in [exp(-nsigma^2/2), 1]
    """
    u = np.asarray(u, dtype=float)

    umin = np.exp(-0.5 * nsigma * nsigma)
    norm_trunc = math.erf(nsigma / math.sqrt(2.0))

    pdf = np.zeros_like(u)

    mask = (u >= umin) & (u < 1.0)
    pdf[mask] = 1.0 / (
        norm_trunc * np.sqrt(np.pi * (-np.log(u[mask])))
    )

    return pdf


# ============================================================
# FFT tune 部分
# ============================================================

def parabolic_interpolate(amp, k):
    y0, y1, y2 = amp[k - 1], amp[k], amp[k + 1]
    denom = y0 - 2.0 * y1 + y2

    if denom == 0:
        return float(k)

    return k + 0.5 * (y0 - y2) / denom


def tune_from_xpx(x, px, pad_factor=PAD_FACTOR, ignore_dc=True, px_scale=None):
    x = np.asarray(x)
    px = np.asarray(px)

    N = len(x)

    if N != len(px):
        raise ValueError("x and px must have same length")

    if N < 8:
        raise ValueError("too few TBT points for FFT tune analysis")

    if px_scale is None:
        sx = x.std(ddof=0)
        sp = px.std(ddof=0)
        px_scale = sp / sx if sx > 0 and sp > 0 else 1.0

    zz = x + 1j * (px / px_scale)
    zz = zz - zz.mean()

    Nfft = int(pad_factor * N)

    Z = np.fft.fft(zz, n=Nfft)
    freqs = np.fft.fftfreq(Nfft, d=1.0)
    mag = np.abs(Z)

    if ignore_dc:
        mag[0] = 0.0

    k = int(np.argmax(mag))
    k_use = min(max(k, 1), Nfft - 2)

    k_peak = parabolic_interpolate(mag, k_use)

    df = freqs[1] - freqs[0]
    tune = (k_peak % Nfft) * df

    return tune


def fft_tune_to_fractional_plot(vx_raw, vy_raw):
    """
    保留你原脚本的 FFT tune 修正规则。
    """
    qx = 1.0 - vx_raw
    qy = 1.0 - vy_raw

    if qy > 0.25:
        qy -= 1.0

    return qx, qy


def compute_fft_tunes():
    pairs = []

    SEO_r_info  = np.loadtxt(SEO_r_filepath,  skiprows=1)
    SEO_pr_info = np.loadtxt(SEO_pr_filepath, skiprows=1)

    SEO_fi_axis = SEO_r_info[0, 1:]
    SEO_Ek_axis = SEO_r_info[1:, 0]

    SEO_r_matrix  = SEO_r_info[1:, 1:]
    SEO_pr_matrix = SEO_pr_info[1:, 1:]

    all_paths = sorted(Path(DATA_DIR0).glob("particle_*.npz"))

    print(f"读取到粒子文件数: {len(all_paths)}")

    for path in all_paths:
        try:
            arr = np.load(path)["particles"]
            arr = arr[arr[:, -1].argsort()]

            if len(arr) < max(16, SG_WINDOW):
                continue

            x  = arr[:, COL_X].copy()
            px = arr[:, COL_XP].copy()
            y  = arr[:, COL_Z].copy()
            py = arr[:, COL_ZP].copy()
            fi = arr[:, COL_fi]
            Ek_MeV = arr[:, COL_Ek]

            fi_mod = np.mod(fi, 2.0 * np.pi)

            r_seo  = np.zeros_like(x)
            pr_seo = np.zeros_like(x)
            flag1  = np.zeros_like(x)
            flag2  = np.zeros_like(x)

            Bilinear_interp_2D_vect_uniform(
                SEO_Ek_axis[0],
                SEO_Ek_axis[1] - SEO_Ek_axis[0],
                len(SEO_Ek_axis),
                SEO_fi_axis[0],
                SEO_fi_axis[1] - SEO_fi_axis[0],
                len(SEO_fi_axis),
                SEO_r_matrix,
                Ek_MeV,
                fi_mod,
                r_seo,
                flag1
            )

            Bilinear_interp_2D_vect_uniform(
                SEO_Ek_axis[0],
                SEO_Ek_axis[1] - SEO_Ek_axis[0],
                len(SEO_Ek_axis),
                SEO_fi_axis[0],
                SEO_fi_axis[1] - SEO_fi_axis[0],
                len(SEO_fi_axis),
                SEO_pr_matrix,
                Ek_MeV,
                fi_mod,
                pr_seo,
                flag2
            )

            if USE_SAVGOL:
                win = SG_WINDOW if SG_WINDOW % 2 else SG_WINDOW + 1

                x  = x  - savgol_filter(x,  win, SG_ORDER)
                px = px - savgol_filter(px, win, SG_ORDER)
                y  = y  - savgol_filter(y,  win, SG_ORDER)
                py = py - savgol_filter(py, win, SG_ORDER)

            # 水平方向减去能量相关闭轨
            x  -= r_seo
            px -= pr_seo

            vx_raw = tune_from_xpx(x, px)
            vy_raw = tune_from_xpx(y, py)

            qx_frac, qy_frac = fft_tune_to_fractional_plot(vx_raw, vy_raw)

            qx_plot = qx_frac + QX_CELL_SHIFT
            qy_plot = qy_frac + QY_CELL_SHIFT

            pairs.append((qx_plot, qy_plot))

        except Exception as e:
            print(f"FFT tune计算失败: {path.name}, {e}")

    if len(pairs) == 0:
        return np.array([]), np.array([])

    qx_all, qy_all = np.array(pairs).T

    print(f"成功提取 FFT tune 粒子数: {len(qx_all)}")
    print(f"Qx FFT min / mean / max = {qx_all.min():.8f}, {qx_all.mean():.8f}, {qx_all.max():.8f}")
    print(f"Qy FFT min / mean / max = {qy_all.min():.8f}, {qy_all.mean():.8f}, {qy_all.max():.8f}")

    return qx_all, qy_all


# ============================================================
# 实际工作点投影到理论直线
# ============================================================

def project_points_to_theory_line(qx, qy, theory_dict):
    Qx0, Qy0 = theory_dict["bare"]
    dQx, dQy = theory_dict["dQ"]

    qx = np.asarray(qx)
    qy = np.asarray(qy)

    denom = dQx * dQx + dQy * dQy

    if denom <= 0:
        raise ValueError("Invalid theory dQ direction.")

    # 沿理论线方向的归一化坐标。
    # 对理论点而言，t = u = lambda(s)/lambda_peak。
    t = ((qx - Qx0) * dQx + (qy - Qy0) * dQy) / denom

    # 垂直于理论线的偏移，用于检查实际 footprint 宽度。
    perp = (-(qx - Qx0) * dQy + (qy - Qy0) * dQx) / np.sqrt(denom)

    return t, perp


# ============================================================
# 作图
# ============================================================

def plot_compare_with_theory_distribution(theory_dict, qx_all, qy_all):
    Qx0, Qy0 = theory_dict["bare"]
    Qx_max, Qy_max = theory_dict["gaussian_peak"]
    dQx_max, dQy_max = theory_dict["dQ"]

    qx_th, qy_th, s_th, u_th = compute_theory_kv_gaussian_distribution(
        theory_dict,
        n_macro=N_THEORY_MACRO,
        seed=THEORY_SEED
    )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(7.0, 6.0),
        constrained_layout=True
    )

    # ========================================================
    # 左图：二维工作点分布
    # ========================================================
    # # ax = axs[0]
    # #
    # ax.scatter(qx_all,
    #         qy_all,s=2)

    if len(qx_all) > 0:
        h = ax.hist2d(
            qx_all,
            qy_all,
            bins=[260, 260],
            range=[QX_RANGE, QY_RANGE],
            norm=LogNorm(),
            cmap="viridis"
        )
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label("Counts", fontsize=19)
        cbar.ax.tick_params(labelsize=14)

    # # 理论解析工作点分布：沿直线分布
    # ax.scatter(
    #     qx_th,
    #     qy_th,
    #     s=5,
    #     c="red",
    #     alpha=0.08,
    #     linewidths=0,
    #     label="Analytical distribution",
    #     zorder=6
    # )

    # 理论解析直线
    umin = np.exp(-0.5 * nsigma * nsigma)
    u_line = np.linspace(umin, 1.0, 500)

    qx_line = Qx0 + dQx_max * u_line
    qy_line = Qy0 + dQy_max * u_line

    ax.plot(
        qx_line,
        qy_line,
        "--",
        color="red",
        lw=3.0,
        label="Analytical KV-Gaussian \ntune spread line",
        zorder=8
    )

    # bare tune
    ax.scatter(
        Qx0,
        Qy0,
        marker="o",
        s=260,
        facecolor="white",
        edgecolor="blue",
        linewidth=2.8,
        zorder=12,
        label="Bare tune"
    )

    # 最大空间电荷频移点
    ax.scatter(
        Qx_max,
        Qy_max,
        marker="*",
        s=500,
        color="red",
        edgecolor="black",
        linewidth=2.4,
        zorder=13,
        label="Theoretic max SC tune shift"
    )

    # ax.plot(
    #     [Qx0, Qx_max],
    #     [Qy0, Qy_max],
    #     "-",
    #     color="red",
    #     lw=1.8,
    #     alpha=0.75,
    #     zorder=7
    # )

    ax.set_xlabel("Qx")
    ax.set_ylabel("Qz")
    # ax.set_title("Working-point footprint")
    # ax.set_xlim(QX_RANGE)
    # ax.set_ylim(QY_RANGE)
    ax.grid(True, alpha=0.30)
    ax.legend(loc="lower left", framealpha=0.88)

    # # ========================================================
    # # 右图：沿理论线方向的一维分布比较
    # # ========================================================
    # ax = axs[1]
    #
    # umin = np.exp(-0.5 * nsigma * nsigma)
    #
    # if len(qx_all) > 0:
    #     t_act, perp_act = project_points_to_theory_line(
    #         qx_all,
    #         qy_all,
    #         theory_dict
    #     )
    #
    #     # 只保留理论上合理范围附近的点。
    #     # 这里允许一点数值偏差。
    #     mask = (t_act >= umin - 0.05) & (t_act <= 1.05)
    #     t_use = t_act[mask]
    #
    #     print("=== 实际工作点投影到理论线方向 ===")
    #     if len(t_use) > 0:
    #         print(f"有效投影点数: {len(t_use)} / {len(t_act)}")
    #         print(f"t min / mean / max = {t_use.min():.8f}, {t_use.mean():.8f}, {t_use.max():.8f}")
    #     else:
    #         print("没有点落在设定的投影范围内。")
    #
    #     ax.hist(
    #         t_use,
    #         bins=90,
    #         range=(umin, 1.0),
    #         density=True,
    #         alpha=0.50,
    #         label="FFT particles projected"
    #     )
    #
    # # 理论解析概率密度
    # t_grid = np.linspace(umin, 0.9995, 800)
    # pdf_grid = theory_u_pdf(t_grid, nsigma=nsigma)
    #
    # ax.plot(
    #     t_grid,
    #     pdf_grid,
    #     color="red",
    #     lw=3.0,
    #     label="Analytical Gaussian"
    # )
    #
    # ax.set_xlabel("u = lambda(s) / lambda_peak")
    # ax.set_ylabel("Probability density")
    # ax.set_title("Distribution along theory line")
    # ax.set_xlim(umin, 1.0)
    # ax.grid(True, alpha=0.30)
    # ax.legend(framealpha=0.88)
    import os
    os.makedirs("./prabfig", exist_ok=True)
    plt.savefig("./prabfig/fig_tune_kv_.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_compare_only_2d(theory_dict, qx_all, qy_all):
    """
    如果你只想要一张二维图，可以调用这个函数。
    """
    Qx0, Qy0 = theory_dict["bare"]
    Qx_max, Qy_max = theory_dict["gaussian_peak"]
    dQx_max, dQy_max = theory_dict["dQ"]

    qx_th, qy_th, s_th, u_th = compute_theory_kv_gaussian_distribution(
        theory_dict,
        n_macro=N_THEORY_MACRO,
        seed=THEORY_SEED
    )

    fig, ax = plt.subplots(figsize=(8.0, 6.5), constrained_layout=True)

    if len(qx_all) > 0:
        h = ax.hist2d(
            qx_all,
            qy_all,
            bins=[260, 260],
            range=[QX_RANGE, QY_RANGE],
            norm=LogNorm(),
            cmap="viridis"
        )
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label("FFT particle density")

    ax.scatter(
        qx_th,
        qy_th,
        s=5,
        c="red",
        alpha=0.08,
        linewidths=0,
        label="Analytical distribution",
        zorder=6
    )

    umin = np.exp(-0.5 * nsigma * nsigma)
    u_line = np.linspace(umin, 1.0, 500)

    qx_line = Qx0 + dQx_max * u_line
    qy_line = Qy0 + dQy_max * u_line

    ax.plot(
        qx_line,
        qy_line,
        "--",
        color="red",
        lw=3.0,
        label="Analytical KV-Gaussian line",
        zorder=8
    )

    ax.scatter(
        Qx0,
        Qy0,
        marker="o",
        s=260,
        facecolor="white",
        edgecolor="blue",
        linewidth=2.8,
        zorder=12,
        label="Bare tune"
    )

    ax.scatter(
        Qx_max,
        Qy_max,
        marker="*",
        s=500,
        color="red",
        edgecolor="black",
        linewidth=2.4,
        zorder=13,
        label="Max SC tune shift"
    )

    ax.set_xlabel("Qx")
    ax.set_ylabel("Qz")
    ax.set_title("Working-point footprint")
    ax.set_xlim(QX_RANGE)
    ax.set_ylim(QY_RANGE)
    ax.grid(True, alpha=0.30)
    ax.legend(loc="lower left", framealpha=0.88)

    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    theory_dict = compute_theory_kv_gaussian_peak()
    qx_all, qy_all = compute_fft_tunes()
    qx_all[qx_all>3.5] -= 1.0
    qy_all+=1.0

    # 推荐：二维 footprint + 一维投影分布
    plot_compare_with_theory_distribution(theory_dict, qx_all, qy_all)

    # 如果只想画二维图，注释上面一行，打开下面一行：
    # plot_compare_only_2d(theory_dict, qx_all, qy_all)