#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ---------- scipy 可选（没有也能跑，只是不用 SG 平滑） ----------
try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
# ============================================================
# 全局字体设置（适合 PPT 展示）
# ============================================================
plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] = 26
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["figure.titlesize"] = 28
plt.rcParams["lines.linewidth"] = 2.5

# ============================================================
#                 用户配置区（只改这里）
# ============================================================

# ------- 跟踪数据（TBT） -------
# TBT_DIR = "./TBT_Dump/merged_files"          # crossing_angle_*.npz 所在目录 _2e13
TBT_DIR = "./output/Benchmark_env_2e14/merged_files"          # crossing_angle_*.npz 所在目录 _2e13
TBT_KEY = "particles"           # npz 内数组 key
TURNS_TO_COMPARE = [0, ] # 你想看第 0~3 圈
OUT_TRACK_NPZ = "sigma_y_map.npz"

# ------- 理论包络（beta -> k -> envelope） -------
betaR_path = "./Bmap/resultsSEO/BetaFuncR.txt"
betaZ_path = "./Bmap/resultsSEO/BetaFuncZ.txt"
row_idx = 2

C = 72.0                # 周长 [m]
eps_x = 60e-6           # RMS 几何发射度 [m·rad]
eps_y = 60e-6

# 线电荷密度（示例：你按自己物理量填写）
e_charge = 1.602176634e-19
N_particles = 2.0e14
# N_particles = 0.0e14
cut_ratio = 0.2
bunch_length = C * cut_ratio
lambda_q = (N_particles / bunch_length) * e_charge  # [C/m]

gamma_rel = 1.32
beta_rel = np.sqrt(1-1/gamma_rel**2)
particle = "proton"

# 平滑设置（没有 scipy 会自动关闭）
smooth_beta = True
sg_win = 11
sg_poly = 3

# 输出图
FIG_COMPARE_Y = "compare_theory_vs_tracking_sigma_y_turn0_to_3.png"

# ============================================================
#                     理论部分函数
# ============================================================

def load_beta_from_txt(path: str, row_index_for_energy: int = 2):
    data = np.loadtxt(path, skiprows=1)
    phi_deg = data[0, 1:]
    beta = data[row_index_for_energy, 1:]
    return phi_deg, beta

def phi_to_s_cell(phi_deg, C_):
    phi_deg = np.asarray(phi_deg, dtype=float)
    phi0 = phi_deg[0]
    dphi = phi_deg[-1] - phi0
    if dphi <= 0:
        raise ValueError("phi_deg 的跨度不合法")
    L_cell = C_ * (dphi / 360.0)
    s = (phi_deg - phi0) / dphi * L_cell
    return s, L_cell, dphi

def _maybe_savgol(x, win, poly):
    if (not HAS_SCIPY) or (win is None) or (poly is None):
        return x
    win_eff = min(win, len(x) // 2 * 2 - 1)
    if win_eff < 5:
        win_eff = 5
    return savgol_filter(x, win_eff, poly)

def k_from_beta_cell(s, beta, smooth=True, win=11, poly=3):
    beta = np.asarray(beta, dtype=float)
    beta_used = beta.copy()

    if smooth and HAS_SCIPY:
        beta_used = _maybe_savgol(beta_used, win, poly)

    ds = s[1] - s[0]
    beta_p = np.gradient(beta_used, ds)
    beta_pp = np.gradient(beta_p, ds)

    k = 1.0 / beta_used**2 + beta_p**2 / (4.0 * beta_used**2) - beta_pp / (2.0 * beta_used)
    return k, beta_used

def envelope_zero_sc(beta_used, eps_rms):
    sigma = np.sqrt(np.maximum(beta_used, 0.0) * eps_rms)
    return 2.0 * sigma  # a = 2 sigma

def matched_initial_from_beta(s, beta_used, eps_rms, smooth_deriv=True, win=11, poly=3):
    ds = s[1] - s[0]
    b = np.asarray(beta_used, dtype=float)

    if smooth_deriv and HAS_SCIPY:
        b_s = _maybe_savgol(b, win, poly)
        b_p = np.gradient(b_s, ds)
    else:
        b_p = np.gradient(b, ds)

    b0 = b[0]
    a0 = 2.0 * np.sqrt(b0 * eps_rms)
    a0p = np.sqrt(eps_rms) * b_p[0] / np.sqrt(b0)
    return a0, a0p

def tile_array_to_nturns(s_cell, y_cell, n_tiles):
    s_cell = np.asarray(s_cell, dtype=float)
    y_cell = np.asarray(y_cell, dtype=float)
    if n_tiles < 1:
        raise ValueError("n_tiles 必须 >= 1")

    L_cell = s_cell[-1] - s_cell[0]
    s_list, y_list = [], []
    for it in range(n_tiles):
        sl = slice(None) if it == 0 else slice(1, None)
        s_list.append(s_cell[sl] + it * L_cell)
        y_list.append(y_cell[sl])
    return np.concatenate(s_list), np.concatenate(y_list)

def tile_phi_to_360n(phi_cell_deg, y_cell, n_turns):
    phi_cell_deg = np.asarray(phi_cell_deg, dtype=float)
    y_cell = np.asarray(y_cell, dtype=float)

    phi0 = phi_cell_deg[0]
    phi_rel = phi_cell_deg - phi0
    dphi_cell = phi_rel[-1]  # cell span in deg

    n_cells_per_turn = int(round(360.0 / dphi_cell))
    n_cells_total = n_cells_per_turn * n_turns

    phi_list, y_list = [], []
    for ic in range(n_cells_total):
        sl = slice(None) if ic == 0 else slice(1, None)
        phi_list.append(phi_rel[sl] + ic * dphi_cell)
        y_list.append(y_cell[sl])

    return np.concatenate(phi_list), np.concatenate(y_list), n_cells_per_turn

def integrate_envelope_long(s_long, kx_long, ky_long,
                            eps_x_rms, eps_y_rms,
                            lambda_q_,
                            beta_rel_, gamma_rel_,
                            a0x, a0x_p, a0y, a0y_p,
                            particle_="proton"):
    eps0 = 8.8541878128e-12
    c = 299792458.0
    qe = 1.602176634e-19

    if particle_ == "proton":
        m = 1.67262192369e-27
    elif particle_ == "electron":
        m = 9.1093837015e-31
    else:
        raise ValueError("particle 只能是 'proton' 或 'electron'")

    # a=2*sigma -> eps_kv = 4 eps_rms
    epsx_kv = 4.0 * eps_x_rms
    epsy_kv = 4.0 * eps_y_rms

    # SC 系数（你之前脚本的形式）
    C_sc = (qe / (m * c**2)) * (lambda_q_ / (4.0 * np.pi * eps0)) * (1.0 / (beta_rel_**2 * gamma_rel_**3))

    y = np.array([a0x, a0x_p, a0y, a0y_p], dtype=float)
    ax = np.empty_like(s_long)
    ay = np.empty_like(s_long)

    for i in range(len(s_long)):
        ax_i, axp_i, ay_i, ayp_i = y
        ax[i] = ax_i
        ay[i] = ay_i

        denom = ax_i + ay_i
        if denom <= 0:
            raise RuntimeError(f"Envelope collapsed at i={i}, s={s_long[i]:.6f} m")

        axpp = -kx_long[i] * ax_i + (epsx_kv**2) / (ax_i**3) + C_sc / denom * 4
        aypp = -ky_long[i] * ay_i + (epsy_kv**2) / (ay_i**3) + C_sc / denom * 4

        if i < len(s_long) - 1:
            ds = s_long[i + 1] - s_long[i]
            # Euler–Cromer
            y[1] += ds * axpp
            y[0] += ds * y[1]
            y[3] += ds * aypp
            y[2] += ds * y[3]

    return ax, ay


# ============================================================
#                     跟踪部分函数
# ============================================================

_angle_pat = re.compile(r"crossing_angle_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_start_")

def parse_angle_from_filename(fname: str):
    m = _angle_pat.search(fname)
    if not m:
        return None
    return float(m.group(1))

def rms_sigma(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    xm = np.mean(x)
    return float(np.sqrt(np.mean((x - xm) ** 2)))

def compute_tracking_sigma_y_map(tbt_dir, key):
    # files = sorted([f for f in os.listdir(tbt_dir) if f.endswith(".npz")])
    files = sorted([f for f in os.listdir(tbt_dir) if f.endswith(".csv")])
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files found in: {tbt_dir}")

    angles = []
    for f in files:
        ang = parse_angle_from_filename(f)
        if ang is None:
            raise ValueError(f"Cannot parse crossing_angle from filename: {f}")
        angles.append(ang)
    angles = np.array(angles, dtype=float)

    order = np.argsort(angles)
    angles = angles[order]
    files = [files[i] for i in order]

    global_max_turn = 0
    for f in files:
        # arr = np.load(os.path.join(tbt_dir, f))[key]
        arr = np.loadtxt(os.path.join(tbt_dir, f), delimiter=',')
        turn = np.floor(arr[:, 4] / (2.0 * np.pi)).astype(np.int64)
        if turn.size > 0:
            global_max_turn = max(global_max_turn, int(np.max(turn)))

    turns = np.arange(global_max_turn + 1, dtype=np.int64)
    sigma_y_map = np.full((turns.size, angles.size), np.nan, dtype=float)

    for iaz, f in enumerate(files):
        # data = np.load(os.path.join(tbt_dir, f))[key]
        data = np.loadtxt(os.path.join(tbt_dir, f), delimiter=',')
        fi = data[:, 4]
        y = data[:, 2]
        turn = np.floor(fi / (2.0 * np.pi)).astype(np.int64)

        for t in range(global_max_turn + 1):
            mask = (turn == t)
            if not np.any(mask):
                continue
            sigma_y_map[t, iaz] = rms_sigma(y[mask])

    return angles, turns, sigma_y_map, files


# ============================================================
#                     主程序：对比绘图
# ============================================================
def main():
    # ------------------ 1) 跟踪：sigma_y(turn, angle) ------------------
    angles, turns, sigma_y_map, files = compute_tracking_sigma_y_map(TBT_DIR, TBT_KEY)

    np.savez(
        OUT_TRACK_NPZ,
        angles=angles,
        turns=turns,
        sigma_y_map=sigma_y_map,
        files=np.array(files, dtype=object),
    )
    print(f"[OK] Tracking sigma_y_map saved: {OUT_TRACK_NPZ}  shape={sigma_y_map.shape}")

    # 需要理论覆盖到的圈数：至少覆盖 max(TURNS_TO_COMPARE)+1
    n_turns_need = int(max(TURNS_TO_COMPARE)) + 1

    # ------------------ 2) 理论：计算 a_y(phi) 并转成 sigma_y_theory ------------------
    phi_cell, beta_x_raw = load_beta_from_txt(betaR_path, row_idx)
    phi2, beta_y_raw = load_beta_from_txt(betaZ_path, row_idx)
    if len(phi_cell) != len(phi2) or np.max(np.abs(phi_cell - phi2)) > 1e-9:
        raise ValueError("BetaFuncR 与 BetaFuncZ 的 phi 轴不一致。")

    s_cell, L_cell, dphi_cell = phi_to_s_cell(phi_cell, C)
    kx_cell, beta_x_use = k_from_beta_cell(s_cell, beta_x_raw, smooth=smooth_beta, win=sg_win, poly=sg_poly)
    ky_cell, beta_y_use = k_from_beta_cell(s_cell, beta_y_raw, smooth=smooth_beta, win=sg_win, poly=sg_poly)

    # zero-SC（可选，不一定用来对比）
    a0y_cell = envelope_zero_sc(beta_y_use, eps_y)
    phi_0, a0y_0, n_cells_per_turn = tile_phi_to_360n(phi_cell, a0y_cell, n_turns_need)

    # with-SC：把 cell k(s) 延拓到 n_turns_need 个整圈
    n_cells_total = n_turns_need * n_cells_per_turn
    s_long, kx_long = tile_array_to_nturns(s_cell, kx_cell, n_cells_total)
    _,      ky_long = tile_array_to_nturns(s_cell, ky_cell, n_cells_total)

    # 匹配初值
    a0x0, a0x_p0 = matched_initial_from_beta(s_cell, beta_x_use, eps_x, smooth_deriv=True, win=sg_win, poly=sg_poly)
    a0y0, a0y_p0 = matched_initial_from_beta(s_cell, beta_y_use, eps_y, smooth_deriv=True, win=sg_win, poly=sg_poly)

    aSCx_long, aSCy_long = integrate_envelope_long(
        s_long, kx_long, ky_long,
        eps_x, eps_y,
        lambda_q,
        beta_rel, gamma_rel,
        a0x0, a0x_p0, a0y0, a0y_p0,
        particle_=particle
    )

    # 理论坐标：phi_global (deg)
    phi_long = 360.0 * s_long / C

    # 理论 RMS：sigma = a/2
    sigma_y_theory = 0.5 * aSCy_long
    sigma_y_zero   = 0.5 * a0y_0  # zero-SC（同样是 a/2）

    print(f"[info] theory cell span={dphi_cell:.6f} deg, cells/turn={n_cells_per_turn}, L_cell={L_cell:.6f} m")
    if not HAS_SCIPY and smooth_beta:
        print("[warn] scipy 不可用：smooth_beta 已自动变为“近似关闭（不做 SG 平滑）”。")

    # ------------------ 3) 合并对比：同一张图 ------------------
    # 把 tracking 的 (turn, angle) 映射到 phi_global = angle + 360*turn
    plt.figure(figsize=(10, 4))

    # 画理论曲线（with-SC）
    plt.plot(phi_long, sigma_y_theory*100, lw=2.4, color='blue',label="Theory (with SC)")

    # # 可选：画 zero-SC（虚线）
    # plt.plot(phi_long, sigma_y_zero*100, lw=2.4, color='blue',label="Theory (without SC)")

    # 叠加 tracking 点（turn 0~3）
    for t in TURNS_TO_COMPARE:
        if t < 0 or t >= sigma_y_map.shape[0]:
            print(f"[warn] tracking has no turn={t}, skip.")
            continue
        phi_pts = angles + 360.0 * t
        sig_pts = sigma_y_map[t, :]
        plt.scatter(phi_pts, sig_pts*100, marker="*",
                 s=280,
                 color='red',
                 edgecolor='black',
                 linewidth=1.5,
                 zorder=10,
                 linestyle="None", label=f" PyFFAG Tracking (with SC)")

    plt.xlabel(r"Azimuth $\phi$ [deg]")
    plt.ylabel(r"$\sigma_y$ [cm]")
    # plt.title("Vertical envelope: Theory vs Tracking")
    plt.xticks()
    plt.yticks()
    plt.grid(True)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.xlim([0, 200])
    plt.ylim([1.5, 3.5])
    plt.savefig(FIG_COMPARE_Y, dpi=300)
    print(f"[OK] Saved: {FIG_COMPARE_Y}")

    import os
    os.makedirs("./prabfig", exist_ok=True)
    plt.savefig("./prabfig/fig_rms_env_sc.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
