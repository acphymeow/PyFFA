# -*- coding: utf-8 -*-
"""
统一预处理脚本：
1) 生成频率曲线 FrequencyCurve_*.txt
2) 生成 spiral E-map: Er_coef.txt / Ez_coef.txt / Ephi_coef.txt

支持两种 gap 模型：
- fixed  : 固定方位角 gap（旧模型）
- spiral : SEO 闭轨与 spiral 线交点确定 gap 位置（新模型）

主数据区格式保持旧格式：
    Time (s)    Frequency (Hz)    V0 (V)    Energy (MeV)
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from scipy.interpolate import interp1d
from matplotlib.colors import TwoSlopeNorm
from FFAG_Field import FFAG_EField_spiral


# ============================================================
# 命令行参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified preprocessor: build frequency curve and/or spiral E-map."
    )
    parser.add_argument(
        "-j", "--json",
        default="./config_Emap.json",
        help="Path to the JSON configuration file."
    )
    return parser.parse_args()

# ============================================================
# 角度辅助函数
# ============================================================
def wrap_0_2pi(x):
    """
    将角度包装到 [0, 2pi)。
    支持标量或 numpy 数组。
    """
    twopi = 2.0 * np.pi
    return np.mod(x, twopi)


def wrap_pm_pi(x):
    """
    将角度包装到 (-pi, pi]。
    支持标量或 numpy 数组。
    """
    twopi = 2.0 * np.pi
    y = (x + np.pi) % twopi - np.pi
    # 可选：把 -pi 统一映射到 +pi，避免边界歧义
    y = np.where(y <= -np.pi, y + twopi, y)
    return y


def phi_spiral_of_r(r, r_ref, s_ref, tan_a):
    r = np.asarray(r, dtype=np.float64)
    return s_ref + tan_a * np.log(r / r_ref)


# ============================================================
# spiral 线与闭轨交点（用于频率曲线几何修正）
# ============================================================
def interp_periodic_phi(phi_query, phi_base, val_base):
    """
    周期角插值：
    1) phi_base wrap 到 [0, 2pi)
    2) 排序
    3) 去除重复点
    4) 扩展成三圈
    5) 用 np.interp 在线性轴上插值
    """
    phi_query = np.asarray(phi_query, dtype=np.float64)
    phi_base = np.asarray(phi_base, dtype=np.float64)
    val_base = np.asarray(val_base, dtype=np.float64)

    if phi_base.ndim != 1 or val_base.ndim != 1:
        raise ValueError("phi_base and val_base must be 1D.")
    if phi_base.size != val_base.size:
        raise ValueError("phi_base and val_base must have the same length.")
    if phi_base.size < 2:
        raise ValueError("Need at least 2 base points for interpolation.")

    twopi = 2.0 * np.pi

    phi_wrapped = wrap_0_2pi(phi_base)

    order = np.argsort(phi_wrapped)
    phi_sorted = phi_wrapped[order]
    val_sorted = val_base[order]

    # 去除重复角点（例如 0 和 2pi wrap 后重合）
    keep = np.ones(phi_sorted.size, dtype=bool)
    keep[1:] = np.diff(phi_sorted) > 1e-14
    phi_unique = phi_sorted[keep]
    val_unique = val_sorted[keep]

    if phi_unique.size < 2:
        raise ValueError("Too few unique phi points after periodic deduplication.")

    phi_ext = np.concatenate([phi_unique - twopi, phi_unique, phi_unique + twopi])
    val_ext = np.concatenate([val_unique, val_unique, val_unique])

    return np.interp(phi_query, phi_ext, val_ext)


def find_intersections_phi_theta(phi, r_co, phi_curve, rmin=None, rmax=None, atol=1e-12):
    g = wrap_pm_pi(phi_curve - phi)
    out = []

    for k in range(len(g) - 1):
        g0 = g[k]
        g1 = g[k + 1]

        # 避免 wrap 在 ±pi 附近造成伪跳变
        if abs(g1 - g0) > np.pi:
            continue

        if abs(g0) < atol:
            phi_star = phi[k]
            r_star = r_co[k]
            if (rmin is None or r_star >= rmin) and (rmax is None or r_star <= rmax):
                out.append((phi_star, r_star))
            continue

        if g0 * g1 < 0.0 or abs(g1) < atol:
            denom = g1 - g0
            if abs(denom) < 1e-15:
                continue

            t = (-g0) / denom
            phi_star = phi[k] + t * (phi[k + 1] - phi[k])
            r_star = r_co[k] + t * (r_co[k + 1] - r_co[k])

            if (rmin is None or r_star >= rmin) and (rmax is None or r_star <= rmax):
                out.append((phi_star, r_star))

    return out


def build_phi_gap_interpolator_from_SEO(
    SEO_r_path, alpha_deg, r_ref, s_ref_deg, rmin_valid, rmax_valid, Nphi,
    interp_kind="linear", verbose=True,
):
    data = np.loadtxt(SEO_r_path, skiprows=1)

    phi_seo = data[0, 1:].astype(np.float64)
    Ek_nodes = data[1:, 0].astype(np.float64)
    r_mat = data[1:, 1:].astype(np.float64)

    # 按能量升序排序
    eorder = np.argsort(Ek_nodes)
    Ek_nodes = Ek_nodes[eorder]
    r_mat = r_mat[eorder, :]

    alpha = np.deg2rad(alpha_deg)
    tan_a = np.tan(alpha)
    s_ref = np.deg2rad(s_ref_deg)

    # if np.isclose(tan_a, 0.0, atol=1e-14):
    #     raise RuntimeError("Invalid alpha_deg: tan(alpha) is too close to zero.")
    if r_ref <= 0.0:
        raise RuntimeError("Invalid r_ref: must be > 0.")
    if rmin_valid <= 0.0 or rmax_valid <= 0.0:
        raise RuntimeError("Invalid rmin_valid/rmax_valid: must be > 0.")
    if rmax_valid <= rmin_valid:
        raise RuntimeError("Invalid r-range: require rmax_valid > rmin_valid.")
    if int(Nphi) < 2:
        raise RuntimeError("Nphi must be >= 2.")

    # 根据有效半径范围，确定 spiral 对应的连续 phi 窗口
    phi_min = s_ref + tan_a * np.log(rmin_valid / r_ref)
    phi_max = s_ref + tan_a * np.log(rmax_valid / r_ref)
    if phi_max < phi_min:
        phi_min, phi_max = phi_max, phi_min

    # 给一点边界余量，避免漏掉端点附近交点
    dphi_pad = max((phi_max - phi_min) * 0.02, 1e-6)
    phi_min -= dphi_pad
    phi_max += dphi_pad

    phi_grid = np.linspace(phi_min, phi_max, int(Nphi), dtype=np.float64)

    phi_pick = np.empty_like(Ek_nodes)
    r_pick = np.empty_like(Ek_nodes)
    ok = np.ones_like(Ek_nodes, dtype=bool)

    for i in range(len(Ek_nodes)):
        # 闭轨插值到连续 phi_grid 上
        r_co_sp = interp_periodic_phi(phi_grid, phi_seo, r_mat[i])

        # 只在有效半径范围内搜索
        mask_valid = (r_co_sp >= rmin_valid) & (r_co_sp <= rmax_valid)
        if np.count_nonzero(mask_valid) < 2:
            ok[i] = False
            phi_pick[i] = np.nan
            r_pick[i] = np.nan
            continue

        phi_use = phi_grid[mask_valid]
        r_use = r_co_sp[mask_valid]

        # 螺旋线改为 theta(r) 形式
        phi_curve = phi_spiral_of_r(r_use, r_ref=r_ref, s_ref=s_ref, tan_a=tan_a)

        pts = find_intersections_phi_theta(
            phi_use, r_use, phi_curve,
            rmin=rmin_valid, rmax=rmax_valid
        )

        if len(pts) == 0:
            ok[i] = False
            phi_pick[i] = np.nan
            r_pick[i] = np.nan
        elif len(pts) == 1:
            phi_pick[i] = pts[0][0]
            r_pick[i] = pts[0][1]
        else:
            raise RuntimeError(
                f"Ek={Ek_nodes[i]:.6g} MeV has {len(pts)} intersections in valid range; "
                "expected exactly 1."
            )

    Ek_ok = Ek_nodes[ok]
    phi_ok = phi_pick[ok]
    r_ok = r_pick[ok]

    if Ek_ok.size < 2:
        raise RuntimeError(
            "Too few valid spiral-gap intersections. "
            "Please check spiral parameters and r-range."
        )

    if verbose:
        print("[phi_gap(E)] built from SEO_r.txt")
        print("  SEO_r:", SEO_r_path)
        print("  Spiral(theta(r)): alpha=%.6g deg, r_ref=%.6g, s_ref=%.6g deg"
              % (alpha_deg, r_ref, s_ref_deg))
        print("  r-range: [%.6g, %.6g] m" % (rmin_valid, rmax_valid))
        print("  valid points: %d / %d" % (Ek_ok.size, Ek_nodes.size))
        print("  phi_gap(deg) span: [%.6g, %.6g]"
              % (np.rad2deg(phi_ok.min()), np.rad2deg(phi_ok.max())))

    phi_gap_of_Ek = interp1d(
        Ek_ok,
        phi_ok,
        kind=interp_kind,
        fill_value="extrapolate",
        assume_sorted=True,
    )

    return phi_gap_of_Ek, Ek_ok, phi_ok, r_ok


# ============================================================
# spiral map 生成用：最近点与法向方向
# ============================================================
@nb.njit(parallel=True, fastmath=True)
def spiral_distance_direction_rphi_with_flag(
    r_grid, phi_grid,
    r_ref, s_ref, alpha_rad,
    r_min, r_max,
    n_iter=10,
    normal_side=1
):
    a = np.tan(alpha_rad)
    a2 = a * a
    one_plus_a2 = 1.0 + a2

    dist = np.empty(r_grid.shape)
    ux = np.empty(r_grid.shape)
    uy = np.empty(r_grid.shape)
    flag = np.zeros(r_grid.shape, dtype=np.int32)

    rg = r_grid.ravel()
    pg = phi_grid.ravel()

    out_d = dist.ravel()
    out_ux = ux.ravel()
    out_uy = uy.ravel()
    out_f = flag.ravel()

    n = rg.size
    inv_tnorm = 1.0 / np.sqrt(one_plus_a2)

    side = 1.0
    if normal_side < 0:
        side = -1.0

    eps_end = 1e-12 * (1.0 + r_max)

    for i in nb.prange(n):
        r_p = rg[i]
        phi_p = pg[i]

        x_p = r_p * np.cos(phi_p)
        y_p = r_p * np.sin(phi_p)

        r = r_p
        if r < r_min:
            r = r_min
        if r > r_max:
            r = r_max
        if r <= 0.0:
            r = r_min

        # Newton 迭代寻找 spiral 上最近点
        for _ in range(n_iter):
            s = s_ref + a * np.log(r / r_ref)
            cs = np.cos(s)
            sn = np.sin(s)

            x = r * cs
            y = r * sn

            x1 = cs - a * sn
            y1 = sn + a * cs

            invr = 1.0 / r
            x2 = -(a * invr) * (sn + a * cs)
            y2 = (a * invr) * (cs - a * sn)

            dx = x - x_p
            dy = y - y_p

            g = dx * x1 + dy * y1
            gp = one_plus_a2 + dx * x2 + dy * y2

            if np.abs(gp) < 1e-14:
                break

            r_new = r - g / gp

            if r_new < r_min:
                r_new = r_min
            if r_new > r_max:
                r_new = r_max

            if np.abs(r_new - r) < 1e-12 * (1.0 + r):
                r = r_new
                break

            r = r_new

        # 标记最近点是否落在端点
        if np.abs(r - r_min) < eps_end:
            out_f[i] = 1
        elif np.abs(r - r_max) < eps_end:
            out_f[i] = 2
        else:
            out_f[i] = 0

        s = s_ref + a * np.log(r / r_ref)
        cs = np.cos(s)
        sn = np.sin(s)

        x = r * cs
        y = r * sn

        vx = x_p - x
        vy = y_p - y
        d = np.sqrt(vx * vx + vy * vy)
        out_d[i] = d

        # spiral 切向
        x1 = cs - a * sn
        y1 = sn + a * cs

        tx = x1 * inv_tnorm
        ty = y1 * inv_tnorm

        # 法向，normal_side = +1 左法向，-1 右法向
        nx = (-ty) * side
        ny = (tx) * side

        out_ux[i] = nx
        out_uy[i] = ny

    return dist, ux, uy, flag


def project_to_polar_components(ux, uy, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    n_r = ux * c + uy * s
    n_phi = -ux * s + uy * c
    return n_r, n_phi


def build_phi_axis_from_spiral(
    r_min, r_max,
    r_ref, s_ref, alpha_rad,
    nphi_full,
    pad_deg,
    min_nphi
):
    """
    根据 spiral 在给定半径范围内的覆盖角度，自动建立连续角窗口。
    """
    a = np.tan(alpha_rad)

    phi1 = s_ref + a * np.log(r_min / r_ref)
    phi2 = s_ref + a * np.log(r_max / r_ref)

    phi_lo = min(phi1, phi2) - np.deg2rad(pad_deg)
    phi_hi = max(phi1, phi2) + np.deg2rad(pad_deg)

    width = phi_hi - phi_lo
    dphi_full = 2.0 * np.pi / nphi_full
    nphi = int(np.ceil(width / dphi_full))
    if nphi < min_nphi:
        nphi = min_nphi

    phi_axis_cont = np.linspace(phi_lo, phi_hi, nphi, endpoint=False)
    phi_axis = np.mod(phi_axis_cont, 2.0 * np.pi)

    return phi_axis, phi_axis_cont


def deriv_r(F, dr):
    dF = np.empty_like(F)
    dF[1:-1] = (F[2:] - F[:-2]) / (2.0 * dr)
    dF[0] = (F[1] - F[0]) / dr
    dF[-1] = (F[-1] - F[-2]) / dr
    return dF


def deriv_phi(F, dphi):
    # 这里是连续角窗口，不做周期差分
    dF = np.empty_like(F)
    dF[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2.0 * dphi)
    dF[:, 0] = (F[:, 1] - F[:, 0]) / dphi
    dF[:, -1] = (F[:, -1] - F[:, -2]) / dphi
    return dF


# ============================================================
# 频率曲线生成
# ============================================================
def build_frequency_curve_from_config(config, Emap, machine):
    save_path = config['filename']

    energy_inj = float(machine['energy_inj'])
    energy_ext = float(machine['energy_ext'])

    gap_model = str(Emap['gap_model']).lower()
    if gap_model not in ("fixed", "spiral"):
        raise ValueError("Emap['gap_model'] must be 'fixed' or 'spiral'.")

    gap_azimuth_raw = Emap['gap_azimuth']
    gap_width = float(Emap['gap_width'])
    E_rmin = float(Emap['rmin'])
    E_rmax = float(Emap['rmax'])
    harmonic = int(Emap['harmonic'])
    Ngap = int(Emap['Ngap'])
    t_start = float(Emap['t_start'])
    spiral_rmin_calc = float(Emap['spiral_rmin_calc'])
    spiral_rmax_calc = float(Emap['spiral_rmax_calc'])

    if Ngap <= 0:
        raise ValueError("Emap['Ngap'] must be a positive integer.")

    if t_start > 0.0:
        raise ValueError("Emap['t_start'] should be <= 0.")

    phi_start_deg = float(Emap['acc_phase_start'])
    phi_end_deg = float(Emap['acc_phase_end'])
    phase_ramp_start = int(Emap['phase_ramp_start'])   # 按 turn 配置
    phase_ramp_end = int(Emap['phase_ramp_end'])       # 按 turn 配置
    num_turns = int(Emap['num_turns'])

    acc_voltage_start = float(Emap['acc_voltage_start'])
    acc_voltage_end = float(Emap['acc_voltage_end'])
    voltage_ramp_start = int(Emap['voltage_ramp_start'])   # 按 turn 配置
    voltage_ramp_end = int(Emap['voltage_ramp_end'])       # 按 turn 配置

    # --------------------------------------------------------
    # 解析 gap_azimuth，并自动生成 rf_shift
    # --------------------------------------------------------
    def _to_float_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [float(v) for v in x]
        if isinstance(x, str):
            s = x.strip()
            if "," in s:
                return [float(v.strip()) for v in s.split(",") if v.strip() != ""]
            return [float(s)]
        return [float(x)]

    gap_azimuth_list = _to_float_list(gap_azimuth_raw)

    if gap_model == "fixed":
        if len(gap_azimuth_list) != Ngap:
            raise ValueError(
                "For fixed gap model, len(gap_azimuth) must equal Emap['Ngap']."
            )
    elif gap_model == "spiral":
        gap_azimuth_list = [360.0 * i / Ngap for i in range(Ngap)]

    # 各 gap 的固定相位偏移由程序自动生成：
    #   rf_shift_i = 360 * harmonic * i / Ngap   (deg)
    rf_shift_list = [(-360.0 * harmonic * i / Ngap) for i in range(Ngap)]
    rf_shift_rad_list = [np.deg2rad(v) for v in rf_shift_list]

    gap_azimuth_str = str(gap_azimuth_list)
    rf_shift_str = str(rf_shift_list)

    print("[Emap] gap_model        =", gap_model)
    print("[Emap] Ngap             =", Ngap)
    print("[Emap] harmonic         =", harmonic)
    print("[Emap] gap_shift (deg)=", gap_azimuth_list)
    print("[Emap] rf_shift (deg)   =", rf_shift_list)

    # --------------------------------------------------------
    # spiral 模型相关数据。若 gap_model='fixed'，这些数组保持为空
    # --------------------------------------------------------
    Ek_phi_nodes = np.array([], dtype=np.float64)
    r_gap_nodes = np.array([], dtype=np.float64)
    phi_gap_nodes_unwrapped = np.array([], dtype=np.float64)
    phi_gap_nodes_wrapped = np.array([], dtype=np.float64)

    spiral_alpha_deg = np.nan
    spiral_r_ref = np.nan
    spiral_s_ref_deg = np.nan
    phi_gap_of_Ek = None

    if gap_model == "spiral":
        Bmap_fold = Emap['Bmap_fold']
        SEO_r_path = os.path.join(Bmap_fold, "resultsSEO_noError", "SEO_r.txt")

        spiral_alpha_deg = float(Emap['spiral_alpha_deg'])
        spiral_r_ref = float(Emap['spiral_r_ref'])
        spiral_s_ref_deg = float(Emap['spiral_s_ref_deg'])
        spiral_Nphi = int(Emap['spiral_Nphi'])

        phi_gap_of_Ek, Ek_phi_nodes, phi_gap_nodes_unwrapped, r_gap_nodes = \
            build_phi_gap_interpolator_from_SEO(
                SEO_r_path=SEO_r_path,
                alpha_deg=spiral_alpha_deg,
                r_ref=spiral_r_ref,
                s_ref_deg=spiral_s_ref_deg,
                rmin_valid=spiral_rmin_calc,
                rmax_valid=spiral_rmax_calc,
                Nphi=spiral_Nphi,
                interp_kind="linear",
                verbose=True,
            )

        phi_gap_nodes_wrapped = wrap_0_2pi(phi_gap_nodes_unwrapped)

    # --------------------------------------------------------
    # 回旋频率插值器：从 SEO 数据构造 f_rev(E)
    # --------------------------------------------------------
    Bmap_fold = Emap['Bmap_fold']
    SEO_ini_path = os.path.join(Bmap_fold, "resultsSEO_noError", "SEO_ini.txt")
    if not os.path.exists(SEO_ini_path):
        raise FileNotFoundError(
            f"{Bmap_fold} 缺少 resultsSEO/SEO_ini.txt 文件：需要先计算闭轨信息以获取回旋频率"
        )
    SEO_ini_data = np.loadtxt(SEO_ini_path, skiprows=2)

    Ek_nodes_f = SEO_ini_data[:, 1].astype(np.float64)
    f_nodes = SEO_ini_data[:, 4].astype(np.float64)

    order_f = np.argsort(Ek_nodes_f)
    Ek_nodes_f = Ek_nodes_f[order_f]
    f_nodes = f_nodes[order_f]

    f_rev_of_Ek = interp1d(
        Ek_nodes_f,
        f_nodes,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=True,
    )

    # --------------------------------------------------------
    # 将按圈编号给出的 ramp 参数转为按 gap-event 编号
    # --------------------------------------------------------
    phase_gap_start = phase_ramp_start * Ngap
    phase_gap_end = phase_ramp_end * Ngap

    voltage_gap_start = voltage_ramp_start * Ngap
    voltage_gap_end = voltage_ramp_end * Ngap

    phi_start_rad = np.deg2rad(phi_start_deg)
    phi_end_rad = np.deg2rad(phi_end_deg)

    def get_sync_phase_rad(gap_event_idx):
        if gap_event_idx < phase_gap_start:
            return phi_start_rad
        elif gap_event_idx <= phase_gap_end:
            denom = max(1, (phase_gap_end - phase_gap_start))
            s = (gap_event_idx - phase_gap_start) / denom
            return phi_start_rad + (phi_end_rad - phi_start_rad) * s
        else:
            return phi_end_rad

    def get_voltage(gap_event_idx):
        if gap_event_idx < voltage_gap_start:
            return acc_voltage_start
        elif gap_event_idx <= voltage_gap_end:
            denom = max(1, (voltage_gap_end - voltage_gap_start))
            s = (gap_event_idx - voltage_gap_start) / denom
            return acc_voltage_start + (acc_voltage_end - acc_voltage_start) * s
        else:
            return acc_voltage_end

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------
    E_values = [energy_inj]
    t_values = [0.0]
    V0_values = [get_voltage(0)]
    phi_values = []       # 同步相位（不含 rf_shift）

    # 初始输出频率为 RF 频率
    f_rev_inj = float(f_rev_of_Ek(energy_inj))
    dt0 = (1.0 / f_rev_inj) / Ngap
    phi0 = get_sync_phase_rad(0)
    phi1 = get_sync_phase_rad(1)
    delta_phi0 = phi1 - phi0
    f_rf0 = harmonic / (Ngap * dt0) + delta_phi0 / (2.0 * np.pi * dt0)
    f_values = [f_rf0]

    # 调试/检查量
    turn_idx_values = [0]
    gap_idx_values = [0]
    dt_values = []
    dtheta_values = []
    delta_phi_values = []
    rev_f_values = [f_rev_inj]

    current_turn = 0
    current_gap = 0
    max_events = num_turns * Ngap
    nominal_gap_angle = 2.0 * np.pi / Ngap

    # --------------------------------------------------------
    # 主循环：按 gap event 推进
    # --------------------------------------------------------
    for event_idx in range(max_events):
        E_old = E_values[-1]

        # 当前 event 的同步相位 / 电压（按 gap event 调度）
        phi_sync = get_sync_phase_rad(event_idx)
        V0 = get_voltage(event_idx)

        phi_values.append(phi_sync)

        # 当前 gap 能量更新
        dE = V0 * np.sin(phi_sync) / 1e6
        E_new = E_old + dE

        # gap 后能量对应的回旋频率，只用于估算飞行时间
        f_rev_new = float(f_rev_of_Ek(E_new))

        # spiral 修正：用 gap 交点方位变化修正 gap -> gap 飞行时间
        if gap_model == "spiral":
            theta_old = float(phi_gap_of_Ek(E_old))
            theta_new = float(phi_gap_of_Ek(E_new))
            dtheta = wrap_pm_pi(theta_new - theta_old)
        else:
            dtheta = 0.0

        # dt = T_rev(E_new)/Ngap * ((2π/Ngap + dtheta)/(2π/Ngap))
        dt = (1.0 / f_rev_new) / Ngap * ((nominal_gap_angle + dtheta) / nominal_gap_angle)
        t_new = t_values[-1] + dt

        # 下一事件编号
        next_event_idx = event_idx + 1

        # 下一事件的同步相位（不含 rf_shift）
        phi_sync_next = get_sync_phase_rad(next_event_idx)
        delta_phi = phi_sync_next - phi_sync

        # 反推 RF 频率
        # f_rf = h/(Ngap*dt) + delta_phi/(2π*dt)
        f_rf = harmonic / (Ngap * dt) + delta_phi / (2.0 * np.pi * dt)

        # 保存新状态
        E_values.append(E_new)
        f_values.append(f_rf)
        t_values.append(t_new)
        V0_values.append(V0)

        rev_f_values.append(f_rev_new)
        dt_values.append(dt)
        dtheta_values.append(dtheta)
        delta_phi_values.append(delta_phi)

        # 下一事件的 gap / turn
        next_gap = (current_gap + 1) % Ngap
        if next_gap == 0:
            next_turn = current_turn + 1
        else:
            next_turn = current_turn

        gap_idx_values.append(next_gap)
        turn_idx_values.append(next_turn)

        if E_new > energy_ext:
            break

        current_gap = next_gap
        current_turn = next_turn

        if current_turn >= num_turns:
            break

    t_arr = np.array(t_values, dtype=np.float64)
    f_arr = np.array(f_values, dtype=np.float64)
    E_arr = np.array(E_values, dtype=np.float64)
    V0_arr = np.array(V0_values, dtype=np.float64)

    # --------------------------------------------------------
    # 重采样到等时间步长，供后续场插值程序使用
    # --------------------------------------------------------
    if f_arr.size == 0 or np.max(f_arr) <= 0:
        raise RuntimeError("频率数组无效，无法确定等时间步长。")

    dt_const = 0.5 / np.max(f_arr)
    t_end = max(t_arr[-1], t_start)

    # 不平移原始时间轴，只在前面补负时间点
    t_src = np.concatenate(([t_start], t_arr))
    f_src = np.concatenate(([f_arr[0]], f_arr))
    V_src = np.concatenate(([V0_arr[0]], V0_arr))
    E_src = np.concatenate(([E_arr[0]], E_arr))

    t_uniform = np.arange(t_start, t_end + 0.5 * dt_const, dt_const, dtype=np.float64)
    f_uniform = np.interp(t_uniform, t_src, f_src)
    V_uniform = np.interp(t_uniform, t_src, V_src)
    E_uniform = np.interp(t_uniform, t_src, E_src)

    data_to_save = np.column_stack((t_uniform, f_uniform, V_uniform, E_uniform))

    # --------------------------------------------------------
    # 写文件头
    # --------------------------------------------------------
    Ek_gap_list = Ek_phi_nodes.tolist() if Ek_phi_nodes.size > 0 else []
    r_gap_list = r_gap_nodes.tolist() if r_gap_nodes.size > 0 else []
    phi_gap_deg_list = np.rad2deg(phi_gap_nodes_wrapped).tolist() if phi_gap_nodes_wrapped.size > 0 else []
    phi_gap_unwrapped_deg_list = np.rad2deg(phi_gap_nodes_unwrapped).tolist() if phi_gap_nodes_unwrapped.size > 0 else []

    header_lines = [
        f"gap_model = {gap_model}",
        f"Ngap = {Ngap}",
        f"acc_voltage_paint = {acc_voltage_start}",
        f"acc_voltage_acc = {acc_voltage_end}",
        f"voltage_ramp_start = {voltage_ramp_start}",
        f"voltage_ramp_end = {voltage_ramp_end}",
        f"voltage_gap_start = {voltage_gap_start}",
        f"voltage_gap_end = {voltage_gap_end}",
        f"acc_phi_paint = {phi_start_deg}",
        f"acc_phi_acc = {phi_end_deg}",
        f"turn_ramp_start = {phase_ramp_start}",
        f"turn_ramp_end = {phase_ramp_end}",
        f"phase_gap_start = {phase_gap_start}",
        f"phase_gap_end = {phase_gap_end}",
        f"gap_azimuth = {gap_azimuth_str}",
        f"rf_shift = {rf_shift_str}",
        f"gap_width = {gap_width}",
        f"E_rmin = {E_rmin}",
        f"E_rmax = {E_rmax}",
        f"harmonic = {harmonic}",
        f"spiral_alpha_deg = {spiral_alpha_deg}",
        f"spiral_r_ref = {spiral_r_ref}",
        f"spiral_s_ref_deg = {spiral_s_ref_deg}",
        f"spiral_gap_node_count = {len(Ek_gap_list)}",
        f"Ek_gap = {Ek_gap_list}",
        f"r_gap = {r_gap_list}",
        f"phi_gap_deg = {phi_gap_deg_list}",
        f"phi_gap_unwrapped_deg = {phi_gap_unwrapped_deg_list}",
        f"dt_uniform = {dt_const:.6e}",
        f"Er_map_file = {Emap['Er_map_file']}",
        f"Ez_map_file = {Emap['Ez_map_file']}",
        f"Ephi_map_file = {Emap['Ephi_map_file']}",
        "#Time (s)        Frequency (Hz)    V0 (V)    Energy (MeV)",
    ]

    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    with open(save_path, "w+", encoding="utf-8") as file:
        file.write("\n".join(header_lines) + "\n")
        np.savetxt(
            file,
            data_to_save,
            fmt=config.get('save_precision', "%.8e"),
            delimiter="\t"
        )

    # --------------------------------------------------------
    # 绘图
    # --------------------------------------------------------
    if config['plot_frequency_curve']:
        plt.figure(figsize=(8, 5))
        plt.plot(t_arr * 1e6, f_arr / 1e6, label='RF Frequency $f_{rf}(t)$', color='k', lw=3)
        plt.xlabel('Time (μs)', fontsize=22)
        plt.ylabel('Frequency (MHz)', fontsize=22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(fontsize=18)
        plt.grid()
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(t_arr * 1e6, E_arr, label='Kinetic Energy $E_k(t)$', color='k', lw=3)
        plt.xlabel('Time (μs)', fontsize=22)
        plt.ylabel('Energy (MeV)', fontsize=22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(fontsize=18)
        plt.grid()
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(t_arr * 1e6, V0_arr / 1e3, label='RF voltage $V_0(t)$', color='k', lw=3)
        plt.xlabel('Time (μs)', fontsize=22)
        plt.ylabel('V0 (kV)', fontsize=22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(fontsize=18)
        plt.grid()
        plt.tight_layout()

        if len(phi_values) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(t_arr[1:] * 1e6, np.rad2deg(phi_values), label='Sync phase', color='k', lw=3)
            plt.xlabel('Time (μs)', fontsize=22)
            plt.ylabel('Phase (deg)', fontsize=22)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.legend(fontsize=18)
            plt.grid()
            plt.tight_layout()

        if gap_model == "spiral" and Ek_phi_nodes.size > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(Ek_phi_nodes, np.rad2deg(phi_gap_nodes_wrapped), 'o-', color='k', lw=2, ms=5,
                     label='Spiral gap angle')
            plt.xlabel('Energy (MeV)', fontsize=22)
            plt.ylabel('Gap angle (deg)', fontsize=22)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.legend(fontsize=18)
            plt.grid()
            plt.tight_layout()

    return data_to_save

# ============================================================
# spiral E-map 生成
# ============================================================
def build_spiral_map_from_config(config, Emap):
    # spiral 几何参数
    alpha_deg = float(Emap['spiral_alpha_deg'])
    alpha_rad = np.deg2rad(alpha_deg)

    r_min_save = float(Emap['spiral_rmin_save'])
    r_max_save = float(Emap['spiral_rmax_save'])

    r_min = float(Emap['spiral_rmin_calc'])
    r_max = float(Emap['spiral_rmax_calc'])

    r_ref = float(Emap['spiral_r_ref'])
    s_ref_deg = float(Emap['spiral_s_ref_deg'])
    s_ref = np.deg2rad(s_ref_deg)

    # gap 场分布参数
    U = float(Emap['spiral_U'])
    FWHM = float(Emap['spiral_FWHM'])
    delta = FWHM / 2.35482

    # 网格参数
    nr = int(Emap['spiral_nr'])
    nphi_full = int(Emap['spiral_nphi_full'])
    pad_deg = float(Emap['spiral_pad_deg'])
    min_nphi = int(Emap['spiral_min_nphi'])

    # 数值参数
    newton_iter = int(Emap['spiral_newton_iter'])
    normal_side = int(Emap['spiral_normal_side'])

    # 输出路径
    save_dir = Emap['emap_save_dir']
    er_name = Emap['Er_map_file']
    ez_name = Emap['Ez_map_file']
    ephi_name = Emap['Ephi_map_file']

    r_axis = np.linspace(r_min, r_max, nr)

    # 自动建立 spiral 对应的连续角窗口
    phi_axis, phi_axis_cont = build_phi_axis_from_spiral(
        r_min, r_max, r_ref, s_ref, alpha_rad,
        nphi_full=nphi_full,
        pad_deg=pad_deg,
        min_nphi=min_nphi
    )

    R, PHI = np.meshgrid(r_axis, phi_axis_cont, indexing="ij")

    # 最近点、法向、端点标记
    dist, ux, uy, flag = spiral_distance_direction_rphi_with_flag(
        R, PHI, r_ref, s_ref, alpha_rad, r_min, r_max,
        n_iter=newton_iter,
        normal_side=normal_side
    )

    # 法向高斯分布
    Eh = (U / (delta * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (dist / delta) ** 2)

    # 投影到极坐标分量
    n_r, n_phi = project_to_polar_components(ux, uy, PHI)
    Er = Eh * n_r
    Ephi = Eh * n_phi

    # 最近点在 spiral 端点上的区域，场置零，后续通过裁剪去除
    mask_end = (flag != 0)
    Er[mask_end] = 0.0
    Ephi[mask_end] = 0.0
    Eh[mask_end] = 0.0

    Emag = np.sqrt(Er * Er + Ephi * Ephi)

    # 由 div(E)=0 推出 Ez/z
    dr = r_axis[1] - r_axis[0]
    dphi = phi_axis_cont[1] - phi_axis_cont[0]

    dEr_dr = deriv_r(Er, dr)
    dEphi_dphi = deriv_phi(Ephi, dphi)

    Ez_over_z = -(Er / R + dEr_dr + (1.0 / R) * dEphi_dphi)

    # 只保存指定半径范围
    mask_r_save = (r_axis >= r_min_save) & (r_axis <= r_max_save)
    r_axis_plot = r_axis[mask_r_save]

    Ez_plot = Ez_over_z[mask_r_save, :]
    Er_plot = Er[mask_r_save, :]
    Ephi_plot = Ephi[mask_r_save, :]
    Emag_plot = Emag[mask_r_save, :]

    os.makedirs(save_dir, exist_ok=True)

    data_Ez = np.zeros((Ez_plot.shape[0] + 1, Ez_plot.shape[1] + 1))
    data_Er = np.zeros((Er_plot.shape[0] + 1, Er_plot.shape[1] + 1))
    data_Ephi = np.zeros((Ephi_plot.shape[0] + 1, Ephi_plot.shape[1] + 1))

    # 第一行存连续角 phi_axis_cont，第一列存半径
    data_Ez[0, 1:] = phi_axis_cont
    data_Er[0, 1:] = phi_axis_cont
    data_Ephi[0, 1:] = phi_axis_cont

    data_Ez[1:, 0] = r_axis_plot
    data_Er[1:, 0] = r_axis_plot
    data_Ephi[1:, 0] = r_axis_plot

    data_Ez[1:, 1:] = Ez_plot
    data_Er[1:, 1:] = Er_plot
    data_Ephi[1:, 1:] = Ephi_plot

    ez_path = os.path.join(save_dir, ez_name)
    er_path = os.path.join(save_dir, er_name)
    ephi_path = os.path.join(save_dir, ephi_name)

    np.savetxt(ez_path, data_Ez)
    np.savetxt(er_path, data_Er)
    np.savetxt(ephi_path, data_Ephi)

    print("[Saved spiral E-map]")
    print("  ", er_path)
    print("  ", ez_path)
    print("  ", ephi_path)

    if config['plot_spiral_map']:
        Re, Pe = np.meshgrid(r_axis_plot, phi_axis_cont, indexing="ij")
        Xe = Re * np.cos(Pe)
        Ye = Re * np.sin(Pe)

        Ez_vis = Ez_plot[:-1, :-1]
        Er_vis = Er_plot[:-1, :-1]
        Ephi_vis = Ephi_plot[:-1, :-1]
        Emag_vis = Emag_plot[:-1, :-1]

        fig, axs = plt.subplots(2, 2, figsize=(10, 6))

        lim = r_max * 1.05
        for ax in axs.ravel():
            ax.set_aspect("equal")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

        vmax = np.max(np.abs(Ez_vis))
        if vmax == 0.0:
            vmax = 1.0
        norm0 = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        m0 = axs[0, 0].pcolormesh(Xe, Ye, Ez_vis, cmap="RdBu_r", norm=norm0, shading="flat")
        plt.colorbar(m0, ax=axs[0, 0])
        axs[0, 0].set_title("Ez/z")

        vmax = np.max(np.abs(Er_vis))
        if vmax == 0.0:
            vmax = 1.0
        norm1 = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        m1 = axs[0, 1].pcolormesh(Xe, Ye, Er_vis, cmap="RdBu_r", norm=norm1, shading="flat")
        plt.colorbar(m1, ax=axs[0, 1])
        axs[0, 1].set_title("Er")

        vmax = np.max(np.abs(Ephi_vis))
        if vmax == 0.0:
            vmax = 1.0
        norm2 = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        m2 = axs[1, 0].pcolormesh(Xe, Ye, Ephi_vis, cmap="RdBu_r", norm=norm2, shading="flat")
        plt.colorbar(m2, ax=axs[1, 0])
        axs[1, 0].set_title("Ephi")

        vmax = np.max(Emag_vis)
        if vmax == 0.0:
            vmax = 1.0
        norm3 = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        m3 = axs[1, 1].pcolormesh(Xe, Ye, Emag_vis, cmap="RdBu_r", norm=norm3, shading="flat")
        plt.colorbar(m3, ax=axs[1, 1])
        axs[1, 1].set_title("|E|")

        plt.tight_layout()


def vis_multi_gap_field(freq_curve_path, z_vis=1.0e-3, v0_volt=1.0, nphi_vis=720):
    """
    读取生成好的频率曲线和 map，
    在整圈 [0, 2pi) 上画实际多-gap 的 Er / Ez / Ephi / data region。

    参数：
    - freq_curve_path : 频率曲线文件路径（header 里会带 map 文件名）
    - z_vis           : 可视化时使用的 z
    - v0_volt         : 电压缩放
    - nphi_vis        : 整圈角向采样数
    """
    EmapObj = FFAG_EField_spiral(freq_curve_path, EnableFlag=True)

    r_vis = EmapObj.r_axis.copy()
    phi_vis = np.linspace(0.0, 2.0 * np.pi, nphi_vis, endpoint=False)

    RR, PP = np.meshgrid(r_vis, phi_vis, indexing="ij")

    r_flat = RR.ravel().astype(np.float64)
    phi_flat = PP.ravel().astype(np.float64)
    z_flat = np.full_like(r_flat, z_vis, dtype=np.float64)

    Er_flat = np.zeros_like(r_flat)
    Ez_flat = np.zeros_like(r_flat)
    Ephi_flat = np.zeros_like(r_flat)
    Hit_flat = np.full(r_flat.shape, -1, dtype=np.int32)

    EmapObj.Interpolation2D_EMap(
        r_flat, phi_flat, z_flat,
        Er_flat, Ez_flat, Ephi_flat,
        Hit_flat,
        v0_volt=v0_volt)

    Er_2d = Er_flat.reshape(RR.shape)
    Ez_2d = Ez_flat.reshape(RR.shape)
    Ephi_2d = Ephi_flat.reshape(RR.shape)
    Hit_2d = Hit_flat.reshape(RR.shape).astype(np.float64)

    def build_edges_from_centers(x):
        x = np.asarray(x, dtype=np.float64)
        dx = np.diff(x)
        edges = np.empty(len(x) + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (x[:-1] + x[1:])
        edges[0] = x[0] - 0.5 * dx[0]
        edges[-1] = x[-1] + 0.5 * dx[-1]
        return edges

    def plot_map_cartesian(r_axis, phi_axis, F, title, ax, cmap="RdBu_r", symmetric=True):
        r_edges = build_edges_from_centers(r_axis)
        phi_edges = build_edges_from_centers(phi_axis)

        Re, Pe = np.meshgrid(r_edges, phi_edges, indexing="ij")
        Xe = Re * np.cos(Pe)
        Ye = Re * np.sin(Pe)

        if symmetric:
            vmax = np.max(np.abs(F))
            if vmax == 0.0:
                vmax = 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        else:
            norm = None

        m = ax.pcolormesh(Xe, Ye, F, cmap=cmap, norm=norm, shading="flat")
        plt.colorbar(m, ax=ax)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return m

    ngap = len(EmapObj.gap_azimuths)

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    plot_map_cartesian(
        r_vis, phi_vis, Er_2d,
        f"Actual multi-gap Er ({ngap} gaps)",
        ax=axs[0, 0]
    )

    plot_map_cartesian(
        r_vis, phi_vis, Ez_2d,
        f"Actual multi-gap Ez at z={z_vis:g} ({ngap} gaps)",
        ax=axs[0, 1]
    )

    plot_map_cartesian(
        r_vis, phi_vis, Ephi_2d,
        f"Actual multi-gap Ephi ({ngap} gaps)",
        ax=axs[1, 0]
    )

    plot_map_cartesian(
        r_vis, phi_vis, Hit_2d,
        f"map region ({ngap} gaps)",
        ax=axs[1, 1],
        cmap="viridis",
        symmetric=False
    )

    plt.tight_layout()
    return fig


# ============================================================
# 主程序
# ============================================================
def main():
    args = parse_args()

    with open(args.json, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    config = config_data['config']
    Emap = config_data['Emap']
    machine = config_data['machine']

    do_freq = bool(config['build_frequency_curve'])
    do_map = bool(config['build_spiral_map'])
    # gap_model 必须显式配置
    gap_model = str(Emap['gap_model']).lower()
    if gap_model not in ("fixed", "spiral"):
        raise ValueError("Emap['gap_model'] must be 'fixed' or 'spiral'.")

    if do_freq:
        build_frequency_curve_from_config(config, Emap, machine)

    if do_map and gap_model == "spiral":
        build_spiral_map_from_config(config, Emap)

        vis_multi_gap_field(
            config['filename'],
            z_vis=float(config.get("plot_all_gaps_z", 1.0e-3)),
            v0_volt=float(config.get("plot_all_gaps_v0", 1.0)),
            nphi_vis=int(config.get("plot_all_gaps_nphi", 720))
        )

    if config['plot_frequency_curve'] or config['plot_spiral_map']:
        plt.show()


if __name__ == "__main__":
    main()