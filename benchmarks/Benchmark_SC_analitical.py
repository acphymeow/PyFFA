#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt

# ===================== 物理常数 =====================
e_charge = 1.602176634e-19       # [C]
m_p      = 1.67262192369e-27     # [kg]
c_light  = 2.99792458e8          # [m/s]
eps0     = 8.8541878128e-12      # [F/m]


# ===================== 相对论因子 =====================
def beta_gamma_from_T(T_MeV, m0_MeV=938.2720813):
    E_tot = T_MeV + m0_MeV
    gamma = E_tot / m0_MeV
    beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
    return beta, gamma


# ===================== 纵向线密度 =====================
def line_density_from_N(N_tot, L_long):
    """均匀纵向：粒子数 / 长度 -> 线密度 [粒子/m]"""
    return N_tot / L_long


def lambda_p_gaussian_peak_trunc(N_tot, sigma_z, nsigma=4.0):
    """
    Gaussian 纵向 (截断 ±nsigma*sigma_z) 的中心切片峰值线密度 λ_p(0)
    """
    z_cut = nsigma * sigma_z
    norm_trunc = math.erf(z_cut / (math.sqrt(2.0) * sigma_z))
    lambda_p_peak = N_tot / (math.sqrt(2.0 * math.pi) * sigma_z * norm_trunc)
    return lambda_p_peak, norm_trunc


def gaussian_trunc_peak_to_avg_factor(sigma_z, nsigma=4.0):
    """
    截断高斯：<ΔQ> = F * ΔQ_peak 的比例因子 F
    （本脚本里保留计算，但不打印平均频移）
    """
    z_cut = nsigma * sigma_z
    norm_trunc = math.erf(z_cut / (math.sqrt(2.0) * sigma_z))
    F = (1.0 / norm_trunc) * (1.0 / math.sqrt(2.0)) * math.erf(nsigma)
    return F, norm_trunc


# ===================== Ex/x, Ez/z，用 RMS 发射度 + beta(s) 表示 =====================
def ExEz_slope_KV_from_rms_lambda(lambda_q,
                                  beta_r_array, beta_z_array,
                                  eps_r_rms, eps_z_rms):
    """
    输入：
      lambda_q      : 线电荷密度 [C/m]
      beta_r_array  : 一 cell 上的 beta_r(theta_i)
      beta_z_array  : 一 cell 上的 beta_z(theta_i)
      eps_r_rms     : 几何 RMS 发射度 (radial) [m·rad]
      eps_z_rms     : 几何 RMS 发射度 (vertical) [m·rad]

    输出：
      Ex_over_x_array, Ez_over_z_array : 与 beta(theta) 一一对应的数组 [V/m^2]
    """
    beta_r_array = np.asarray(beta_r_array)
    beta_z_array = np.asarray(beta_z_array)

    sigma_r = np.sqrt(beta_r_array * eps_r_rms)
    sigma_z = np.sqrt(beta_z_array * eps_z_rms)

    # 保护一下，防止0
    sigma_r = np.maximum(sigma_r, 1e-12)
    sigma_z = np.maximum(sigma_z, 1e-12)

    Ex_over_x = lambda_q / (4.0 * math.pi * eps0 * sigma_r * (sigma_r + sigma_z))
    Ez_over_z = lambda_q / (4.0 * math.pi * eps0 * sigma_z * (sigma_r + sigma_z))
    return Ex_over_x, Ez_over_z


def k_sc_arrays_from_rms_lambda(lambda_q,
                                beta_r_array, beta_z_array,
                                eps_r_rms, eps_z_rms,
                                T_MeV,
                                q=e_charge, m=m_p):
    """
    给定 λ_q + beta(theta) + RMS 发射度，计算一 cell 上的：
      k_sc_r_array(theta), k_sc_z_array(theta)

    注意：这里只算一个 cell，后面用 “乘 N_cells” 还原整环积分。
    """
    beta_rel, gamma = beta_gamma_from_T(T_MeV)
    Ex_over_x, Ez_over_z = ExEz_slope_KV_from_rms_lambda(
        lambda_q, beta_r_array, beta_z_array, eps_r_rms, eps_z_rms
    )

    factor = - q / (gamma**3 * m * beta_rel * beta_rel * c_light * c_light)
    k_sc_r = factor * Ex_over_x
    k_sc_z = factor * Ez_over_z
    return k_sc_r, k_sc_z


def deltaQ_from_cell_k_sc(fi_deg_array,
                          beta_array, k_sc_array,
                          C, N_cells):
    """
    使用一个cell上的 beta(theta), k_sc(theta) 计算整环 ΔQ：

      ΔQ = (1/4π) ∮ β(s) k_sc(s) ds
          = (1/4π) * N_cells * ∫_cell β(s) k_sc(s) ds

    这里：
      fi_deg_array : 一个 cell 上的方位角（度），比如 0~30 deg
      C            : 全环周长 [m]
      N_cells      : cell 数
    """
    fi_deg_array = np.asarray(fi_deg_array)
    beta_array   = np.asarray(beta_array)
    k_sc_array   = np.asarray(k_sc_array)

    # 一个 cell 对应的角度跨度
    fi_cell_min = fi_deg_array[0]
    fi_cell_max = fi_deg_array[-1]
    d_fi_deg    = fi_cell_max - fi_cell_min
    L_cell      = C * (d_fi_deg / 360.0)  # 按角度份额计算 cell 弧长

    # 把 fi_deg 映射到 s_cell（这个 cell 内的弧长坐标）
    # 线性映射：fi 从 [fi_min, fi_max] -> s 从 [0, L_cell]
    s_cell = (fi_deg_array - fi_cell_min) / d_fi_deg * L_cell

    integrand = beta_array * k_sc_array
    I_cell = np.trapz(integrand, s_cell)      # ∫_cell β k_sc ds
    I_ring = N_cells * I_cell                 # ∮ β k_sc ds

    dQ = I_ring / (4.0 * math.pi)
    return dQ


# ===================== 主程序示例 =====================
if __name__ == "__main__":

    # ------------ 1. 读入一个 cell 的 beta(theta) ------------
    betaR_filepath = "./Bmap/resultsSEO/BetaFuncR.txt"
    betaZ_filepath = "./Bmap/resultsSEO/BetaFuncZ.txt"

    betaR_data = np.loadtxt(betaR_filepath, skiprows=1)
    betaZ_data = np.loadtxt(betaZ_filepath, skiprows=1)

    # 假设第 0 行是 fi(deg)，第 2 行是 300 MeV 对应的 beta
    beta_fiAxis_deg = betaR_data[0, 1:]
    betaR_300MeV = betaR_data[2, 1:]
    betaZ_300MeV = betaZ_data[2, 1:]

    # ------------ 2. 机器和束流参数 ------------
    C   = 70.0      # 周长 [m]，自己改
    T   = 300.0     # 动能 [MeV]
    Qr0 = 2.68      # 无 SC Qr
    Qz0 = 2.11      # 无 SC Qz

    N_tot = 2.0e13  # 总质子数
    L_long_uniform = 14.0   # 假设纵向均匀长度 [m]

    eps_r_rms = 60.0e-6    # RMS 几何发射度 [m·rad], 1倍RMS椭圆面积[pi.m.rad]
    eps_z_rms = 60.0e-6

    # 由一个 cell 的角度跨度推 cell 数（例如 fi 为 0~30 deg -> 12 cell）
    fi_span_deg = beta_fiAxis_deg[-1] - beta_fiAxis_deg[0]
    N_cells = int(round(360.0 / fi_span_deg))

    beta_rel, gamma = beta_gamma_from_T(T)
    print("=== 基本参数 ===")
    print(f"beta  = {beta_rel:.6f}")
    print(f"gamma = {gamma:.6f}")
    print(f"N_cells (估算) = {N_cells:d}, fi_span_deg = {fi_span_deg:.3f} deg")

    # ------------ 3. 画该 RMS 发射度下的 R/Z 包络 ------------
    # 这里包络定义为 KV 半轴：a = 2 * sigma = 2 * sqrt(beta * eps_rms)
    betaR_arr = np.asarray(betaR_300MeV)
    betaZ_arr = np.asarray(betaZ_300MeV)

    sigma_r = np.sqrt(betaR_arr * eps_r_rms)
    sigma_z = np.sqrt(betaZ_arr * eps_z_rms)
    a_r = 2.0 * sigma_r
    a_z = 2.0 * sigma_z

    plt.figure()
    plt.plot(beta_fiAxis_deg, a_r, label="Radial envelope: a_r = 2σ_r")
    plt.plot(beta_fiAxis_deg, a_z, label="Vertical envelope: a_z = 2σ_z")
    plt.xlabel("Azimuth φ [deg]")
    plt.ylabel("Beam envelope [m]")
    plt.title("R/Z KV envelope for given RMS emittance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()

    # ------------ 3b. 画 Ex/x 和 Ez/z 随 φ 的变化（Uniform 纵向） ------------
    lambda_p_uniform = line_density_from_N(N_tot, L_long_uniform)
    lambda_q_uniform = lambda_p_uniform * e_charge

    Ex_over_x_u, Ez_over_z_u = ExEz_slope_KV_from_rms_lambda(
        lambda_q_uniform,
        betaR_300MeV, betaZ_300MeV,
        eps_r_rms, eps_z_rms
    )

    plt.figure()
    plt.plot(beta_fiAxis_deg, Ex_over_x_u, label="Ex/x (uniform)")
    plt.plot(beta_fiAxis_deg, Ez_over_z_u, label="Ez/z (uniform)")
    plt.xlabel("Azimuth φ [deg]")
    plt.ylabel("Field slope [V/m$^2$]")
    plt.title("Near-axis field slopes Ex/x, Ez/z (uniform longitudinal)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()

    # ------------ 4. Uniform 纵向：理论 ΔQ ------------
    # （注意 lambda_q_uniform 已在 3b 中计算）
    k_sc_r_u, k_sc_z_u = k_sc_arrays_from_rms_lambda(
        lambda_q_uniform,
        betaR_300MeV, betaZ_300MeV,
        eps_r_rms, eps_z_rms,
        T_MeV=T
    )

    dQ_r_u = deltaQ_from_cell_k_sc(beta_fiAxis_deg, betaR_300MeV,
                                   k_sc_r_u, C, N_cells)
    dQ_z_u = deltaQ_from_cell_k_sc(beta_fiAxis_deg, betaZ_300MeV,
                                   k_sc_z_u, C, N_cells)

    print("\n=== Uniform 纵向 ===")
    print(f"lambda_p (uniform) = {lambda_p_uniform:.3e}  [1/m]")
    print(f"ΔQ_r,uniform = {dQ_r_u:.4f}")
    print(f"ΔQ_z,uniform = {dQ_z_u:.4f}")
    print(f"Qr_with_SC (uniform) ≈ {Qr0 + dQ_r_u:.4f}")
    print(f"Qz_with_SC (uniform) ≈ {Qz0 + dQ_z_u:.4f}")

    # ------------ 5. Gaussian 纵向：只打印峰值 ΔQ ------------
    nsigma = 4.0
    # 定义：8 sigma_z = L_long_uniform -> sigma_z = L_long_uniform / 8
    sigma_z_long = L_long_uniform / (2.0 * nsigma)

    lambda_p_peak, norm_trunc = lambda_p_gaussian_peak_trunc(
        N_tot, sigma_z_long, nsigma=nsigma
    )
    lambda_q_peak = lambda_p_peak * e_charge

    k_sc_r_g_peak, k_sc_z_g_peak = k_sc_arrays_from_rms_lambda(
        lambda_q_peak,
        betaR_300MeV, betaZ_300MeV,
        eps_r_rms, eps_z_rms,
        T_MeV=T
    )

    dQ_r_g_peak = deltaQ_from_cell_k_sc(beta_fiAxis_deg, betaR_300MeV,
                                        k_sc_r_g_peak, C, N_cells)
    dQ_z_g_peak = deltaQ_from_cell_k_sc(beta_fiAxis_deg, betaZ_300MeV,
                                        k_sc_z_g_peak, C, N_cells)

    # 仍然计算 F 和平均频移，但不打印
    F_gauss, norm_trunc_check = gaussian_trunc_peak_to_avg_factor(
        sigma_z_long, nsigma=nsigma
    )
    dQ_r_g_avg = F_gauss * dQ_r_g_peak
    dQ_z_g_avg = F_gauss * dQ_z_g_peak

    print("\n=== Gaussian 纵向 (只看峰值, 截断 ±{:.1f}σ) ===".format(nsigma))
    print(f"sigma_z (longitudinal) = {sigma_z_long:.3f} m")
    print(f"norm_trunc = {norm_trunc:.6f}")
    print(f"ΔQ_r,Gaussian_peak = {dQ_r_g_peak:.4f}")
    print(f"ΔQ_z,Gaussian_peak = {dQ_z_g_peak:.4f}")
    print(f"Qr_with_SC (Gaussian_peak) ≈ {Qr0 + dQ_r_g_peak:.4f}")
    print(f"Qz_with_SC (Gaussian_peak) ≈ {Qz0 + dQ_z_g_peak:.4f}")


    # ------------ 6. Ex vs x & Ez vs z 散点图（粒子采样验证线性） ------------

    print("\n=== 生成 Ex-x / Ez-z 散点图（线性验证） ===")

    # 每 1 度取一次（可改），每个角度采样 Np 粒子
    phi_list = beta_fiAxis_deg
    Np = 100

    # 颜色映射
    norm_color = plt.Normalize(vmin=min(phi_list), vmax=max(phi_list))

    # -------- Ex vs x --------
    plt.figure(figsize=(7,5))
    for phi, Exx in zip(phi_list, Ex_over_x_u):

        # 当前角度对应的包络半轴
        idx = np.where(beta_fiAxis_deg == phi)[0][0]
        ar = a_r[idx]   # KV 半轴 = 2 sigma_r

        # 采样 x
        x_samples = np.random.uniform(-ar, ar, Np)

        # 空间电荷力（线性模型）
        Ex_samples = Exx * x_samples

        plt.scatter(x_samples, Ex_samples, s=6, color='b', alpha=0.45)

    plt.xlabel("x [m]")
    plt.ylabel("Ex [V/m]")
    plt.title("Ex vs x")
    plt.grid(True)
    plt.tight_layout()

    # -------- Ez vs z --------
    plt.figure(figsize=(7,5))
    for phi, Ezz in zip(phi_list, Ez_over_z_u):

        idx = np.where(beta_fiAxis_deg == phi)[0][0]
        az = a_z[idx]

        z_samples = np.random.uniform(-az, az, Np)
        Ez_samples = Ezz * z_samples

        plt.scatter(z_samples, Ez_samples, s=6, color='b', alpha=0.45)

    plt.xlabel("z [m]")
    plt.ylabel("Ez [V/m]")
    plt.title("Ez vs z")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
