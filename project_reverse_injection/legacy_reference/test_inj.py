# -*- coding: utf-8 -*-
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from numba import set_num_threads
set_num_threads(1)

from matplotlib.colors import TwoSlopeNorm
from itertools import groupby
from operator import itemgetter

from FFAG_Distribution import GenerateBunches, LoadSEOParams
from FFAG_track import FFAG_RungeKutta, FunctionForAccelerationBunch_dt
from FFAG_ParasAndConversion import FFAG_GlobalParameters, FFAG_ConversionTools
from FFAG_Field import FFAG_EField_new, FFAG_Bfield_analytical
from FFAG_Bunch import FFAG_Bunch, FFAG_ManageBunchAttribute
from FFAG_dump import StepDump, StepDumpBunch, PositionDump, Dumps, PositionDumpBunch

import pyfftw


# =========================================================
# 0. 通用
# =========================================================
def load_config_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# =========================================================
# 1. 周期性 gap 探测
# =========================================================
def find_periodic_gaps(theta_rad, Bz_array, threshold=0.05, min_gap_width=2):
    """
    在周期性 Bz(theta) 数据中寻找所有 gap。
    首尾视为相连；若 gap 在边界被截断，会自动合并。
    """
    theta_rad = np.asarray(theta_rad)
    Bz_array = np.asarray(Bz_array)

    if theta_rad.ndim != 1 or Bz_array.ndim != 1:
        raise ValueError("theta_rad 和 Bz_array 必须是一维数组")
    if len(theta_rad) != len(Bz_array):
        raise ValueError("theta_rad 和 Bz_array 长度不一致")
    if len(theta_rad) < 2:
        return []

    theta_deg = np.rad2deg(theta_rad)

    gap_mask = np.abs(Bz_array) < threshold
    gap_indices = np.where(gap_mask)[0]
    if gap_indices.size == 0:
        return []

    groups = []
    for _, g in groupby(enumerate(gap_indices), lambda ix: ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        groups.append(group)

    # 首尾合并
    if len(groups) >= 2 and gap_mask[0] and gap_mask[-1]:
        merged = groups[-1] + groups[0]
        groups = [merged] + groups[1:-1]

    groups = [g for g in groups if len(g) >= min_gap_width]
    if len(groups) == 0:
        return []

    dtheta = np.median(np.diff(theta_rad))
    period_rad = (theta_rad[-1] - theta_rad[0]) + dtheta
    period_deg = np.rad2deg(period_rad)

    gap_list = []
    for ig, group in enumerate(groups):
        wraps = np.any(np.diff(group) < 0)

        if not wraps:
            start_idx = group[0]
            end_idx = group[-1]
            center_idx = group[len(group) // 2]

            start_deg = theta_deg[start_idx]
            end_deg = theta_deg[end_idx]
            center_deg = theta_deg[center_idx]
            width_deg = end_deg - start_deg

        else:
            start_idx = group[0]
            end_idx = group[-1]
            center_idx = group[len(group) // 2]

            start_deg = theta_deg[start_idx]
            end_deg = theta_deg[end_idx]
            width_deg = (period_deg - start_deg) + end_deg

            deg_unwrap = []
            prev_idx = group[0]
            wrapped_part = False
            for idx in group:
                val = theta_deg[idx]
                if idx < prev_idx:
                    wrapped_part = True
                if wrapped_part:
                    val += period_deg
                deg_unwrap.append(val)
                prev_idx = idx

            center_unwrap = deg_unwrap[len(deg_unwrap) // 2]
            center_deg = center_unwrap % period_deg

        gap_list.append({
            "id": ig + 1,
            "indices": group,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "center_idx": center_idx,
            "start_deg": start_deg,
            "end_deg": end_deg,
            "center_deg": center_deg,
            "start_rad": np.deg2rad(start_deg),
            "end_rad": np.deg2rad(end_deg),
            "center_rad": np.deg2rad(center_deg),
            "width_deg": width_deg,
            "npts": len(group),
            "wraps": wraps,
        })

    gap_list.sort(key=lambda g: g["center_deg"])
    for i, gap in enumerate(gap_list):
        gap["id"] = i + 1

    return gap_list


# =========================================================
# 2. 画 2-cell Bz 曲线与 gap
# =========================================================
def plot_two_cell_bz_with_gaps(theta_rad, bz_line, gap_list, ek=None):
    theta_deg = np.rad2deg(theta_rad)
    theta_min = theta_deg[0]
    theta_max = theta_deg[-1]

    plt.figure(figsize=(11, 5.5))
    label = 'Bz on SEO (2-cell)'
    if ek is not None:
        label += f' (Ek={ek:.6f} MeV)'
    plt.plot(theta_deg, bz_line, lw=2.5, label=label)

    colors = ['orange', 'cyan', 'violet', 'lightgreen', 'pink', 'gold', 'skyblue']
    ymax = np.max(bz_line)
    ymin = np.min(bz_line)

    for i, gap in enumerate(gap_list):
        color = colors[i % len(colors)]

        if not gap["wraps"]:
            plt.axvspan(gap["start_deg"], gap["end_deg"],
                        color=color, alpha=0.28,
                        label=f'Gap {gap["id"]}')
        else:
            plt.axvspan(gap["start_deg"], theta_max,
                        color=color, alpha=0.28,
                        label=f'Gap {gap["id"]}')
            plt.axvspan(theta_min, gap["end_deg"],
                        color=color, alpha=0.28)

        plt.axvline(gap["center_deg"], color='red', linestyle='--', lw=1.8,
                    alpha=0.9, label='Gap centers' if i == 0 else None)

        y_text = ymax - 0.06 * (ymax - ymin)
        plt.text(gap["center_deg"], y_text, f'G{gap["id"]}',
                 color='red', fontsize=13, ha='center', va='bottom')

    plt.xlabel(r'$\theta$ [deg]', fontsize=16)
    plt.ylabel('Bz [T]', fontsize=16)
    plt.title('SEO Bz over 2 cells with periodic gap detection', fontsize=17)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()


# =========================================================
# 3. 读取 SEO_r.txt（一圈）
# =========================================================
def load_seo_r_data(seo_r_txt_path):
    raw = np.loadtxt(seo_r_txt_path)
    SEO_azimuth = raw[0, 1:]
    SEO_EK = raw[1:, 0]
    SEO_r = raw[1:, 1:]

    if SEO_r.shape != (len(SEO_EK), len(SEO_azimuth)):
        raise RuntimeError(
            f"SEO_r 维度不匹配：SEO_r.shape={SEO_r.shape}, "
            f"expected=({len(SEO_EK)}, {len(SEO_azimuth)})"
        )
    if not np.all(np.diff(SEO_azimuth) > 0):
        raise RuntimeError("SEO_azimuth 不是严格递增，无法直接插值。")

    return SEO_azimuth, SEO_EK, SEO_r

def load_seo_pr_data(seo_pr_txt_path):
    raw = np.loadtxt(seo_pr_txt_path)
    SEO_azimuth = raw[0, 1:]
    SEO_EK = raw[1:, 0]
    SEO_pr = raw[1:, 1:]

    if SEO_pr.shape != (len(SEO_EK), len(SEO_azimuth)):
        raise RuntimeError(
            f"SEO_pr 维度不匹配：SEO_pr.shape={SEO_pr.shape}, "
            f"expected=({len(SEO_EK)}, {len(SEO_azimuth)})"
        )
    if not np.all(np.diff(SEO_azimuth) > 0):
        raise RuntimeError("SEO_pr 的 azimuth 不是严格递增，无法直接插值。")

    return SEO_azimuth, SEO_EK, SEO_pr
# =========================================================
# 4. 读取 SEO_Bz.txt（连续 2 个 cell）
# =========================================================
def load_seo_bz_two_cell_data(seo_bz_txt_path):
    raw = np.loadtxt(seo_bz_txt_path, skiprows=1)

    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 3:
        raise RuntimeError("SEO_Bz.txt 读入后维度异常，请检查文件格式。")

    theta_two_cell = raw[0, 1:]
    bz_ek = raw[1:, 0]
    SEO_Bz_two_cell = raw[1:, 1:]

    if SEO_Bz_two_cell.shape != (len(bz_ek), len(theta_two_cell)):
        raise RuntimeError(
            f"SEO_Bz_two_cell 维度不匹配：{SEO_Bz_two_cell.shape}, "
            f"expected=({len(bz_ek)}, {len(theta_two_cell)})"
        )

    if not np.all(np.diff(theta_two_cell) > 0):
        raise RuntimeError("SEO_Bz.txt 第一行给出的 2-cell azimuth 不是严格递增。")

    return theta_two_cell, bz_ek, SEO_Bz_two_cell


# =========================================================
# 5. 能量匹配：把 SEO_Bz 的每一行对应到 SEO_r 的某一行
# =========================================================
def match_bz_rows_to_seo_rows(bz_ek, seo_ek, tol=1e-6):
    bz_ek = np.asarray(bz_ek)
    seo_ek = np.asarray(seo_ek)

    matched_idx = []
    for e in bz_ek:
        idx = np.argmin(np.abs(seo_ek - e))
        if np.abs(seo_ek[idx] - e) > tol:
            print(f"[warning] energy match not exact: Bz Ek={e:.8f}, nearest SEO Ek={seo_ek[idx]:.8f}")
        matched_idx.append(idx)

    return np.array(matched_idx, dtype=int)


# =========================================================
# 6. 基于 2-cell SEO_Bz 做 gap 探测，并展开到整圈
# =========================================================
def detect_gaps_on_all_seo_from_two_cell_bz(
    theta_two_cell_rad,
    bz_ek,
    SEO_Bz_two_cell,
    seo_row_indices,
    SEO_azimuth,
    SEO_EK,
    SEO_r,
    total_cells,
    template_cells,
    threshold=0.08,
    min_gap_width=2
):
    if total_cells % template_cells != 0:
        raise RuntimeError("total_cells 必须能被 template_cells 整除。")

    repeat_times = total_cells // template_cells
    template_period = (2.0 * np.pi) * (template_cells / total_cells)

    results = []

    for i_bz in range(SEO_Bz_two_cell.shape[0]):
        i_seo = seo_row_indices[i_bz]
        ek = SEO_EK[i_seo]
        bz_line = SEO_Bz_two_cell[i_bz, :]

        gap_list_template = find_periodic_gaps(
            theta_two_cell_rad,
            bz_line,
            threshold=threshold,
            min_gap_width=min_gap_width
        )

        gap_points_full_ring = []
        for gap in gap_list_template:
            phi0 = gap["center_rad"]

            for k in range(repeat_times):
                phi_abs = (phi0 + k * template_period) % (2.0 * np.pi)

                r_val = np.interp(phi_abs, SEO_azimuth, SEO_r[i_seo, :])
                x_val = r_val * np.cos(phi_abs)
                y_val = r_val * np.sin(phi_abs)

                gap_points_full_ring.append({
                    "gap_id_in_template": gap["id"],
                    "repeat_id": k,
                    "phi_template_rad": phi0,
                    "phi_abs_rad": phi_abs,
                    "phi_abs_deg": np.rad2deg(phi_abs),
                    "r_center": r_val,
                    "x_center": x_val,
                    "y_center": y_val,
                    "gap_width_deg": gap["width_deg"],
                    "wraps_in_template": gap["wraps"],
                })

        gap_points_full_ring.sort(key=lambda d: d["phi_abs_rad"])

        results.append({
            "bz_row_index": i_bz,
            "seo_row_index": i_seo,
            "Ek": ek,
            "bz_two_cell": bz_line,
            "gap_list_template": gap_list_template,
            "gap_points_full_ring": gap_points_full_ring,
        })

    return results


# =========================================================
# 7. 构造整圈背景 top view
# =========================================================
def build_full_ring_bmap(data, r_axis, fi_axis_single_cell, total_cells, thin=10):
    fi_axis_thin = fi_axis_single_cell[::thin]
    data_thin = data[:, ::thin]

    if fi_axis_thin.size < 2:
        raise RuntimeError("theta 稀疏化后点数 < 2，请减小 thin。")

    fi0 = fi_axis_thin[:-1]
    d0 = data_thin[:, :-1]

    data_full = np.tile(d0, (1, total_cells))
    nphi_full = data_full.shape[1]
    fi_axis_full = np.linspace(0.0, 2.0 * np.pi, nphi_full, endpoint=False)

    mesh_f, mesh_r = np.meshgrid(fi_axis_full, r_axis)
    mesh_x = mesh_r * np.cos(mesh_f)
    mesh_y = mesh_r * np.sin(mesh_f)

    return data_full, fi_axis_full, mesh_x, mesh_y


# =========================================================
# 8. 画整圈 top view：背景磁场 + SEO 闭轨 + SEO gap 点
# =========================================================
def plot_topview_with_seo_detected_gap_points(
    mesh_x, mesh_y, data_full, norm,
    SEO_azimuth, SEO_r,
    seo_gap_results,
    plot_seo_orbits=True,
    every_n_orbit=8,
    plot_gap_points=True,
    every_n_gap_orbit=1,
    selected_point=None
):
    plt.figure(10, figsize=(8.0, 5.0))

    cf = plt.contourf(
        mesh_x, mesh_y, data_full,
        levels=120,
        cmap='RdBu_r',
        norm=norm
    )

    if plot_seo_orbits:
        first = True
        for i in range(0, SEO_r.shape[0], every_n_orbit):
            r_orbit = SEO_r[i, :]
            x = r_orbit * np.cos(SEO_azimuth)
            y = r_orbit * np.sin(SEO_azimuth)
            plt.plot(
                x, y,
                color='k', linewidth=0.8, alpha=0.22,
                label='SEO orbits' if first else None
            )
            first = False

    if plot_gap_points:
        first = True
        for i, item in enumerate(seo_gap_results):
            if i % every_n_gap_orbit != 0:
                continue

            for pt in item["gap_points_full_ring"]:
                plt.plot(
                    pt["x_center"], pt["y_center"],
                    'ko', markersize=3.4, alpha=0.92,
                    label='Detected SEO gap centers' if first else None
                )
                first = False

    if selected_point is not None:
        plt.plot(selected_point["x_center"], selected_point["y_center"],
                 marker='o', markersize=10, color='yellow',
                 markeredgecolor='black', markeredgewidth=1.2,
                 label='Selected stripping foil point')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(cf, label='Bz (T)', shrink=0.84, aspect=35, pad=0.02)
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('y (m)', fontsize=16)
    plt.title('FFAG top view with SEO-detected gap centers', fontsize=18)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()


# =========================================================
# 9. 选择目标 gap 点
# =========================================================
def choose_target_gap_point(seo_gap_results, orbit_index=0, point_index=0):
    if not (0 <= orbit_index < len(seo_gap_results)):
        raise IndexError(f"orbit_index={orbit_index} 越界，合法范围 0 ~ {len(seo_gap_results)-1}")

    item = seo_gap_results[orbit_index]
    pts = item["gap_points_full_ring"]

    if len(pts) == 0:
        raise RuntimeError(f"所选轨道 orbit_index={orbit_index} 上没有找到 gap 点。")

    if not (0 <= point_index < len(pts)):
        raise IndexError(f"point_index={point_index} 越界，合法范围 0 ~ {len(pts)-1}")

    return item, pts[point_index]


# =========================================================
# 10. 打印选中 gap 点信息
# =========================================================
def print_selected_gap_info(item, pt):
    print("=" * 80)
    print("Selected stripping foil / gap point:")
    print(f"Ek               = {item['Ek']:.6f} MeV")
    print(f"phi_abs_rad       = {pt['phi_abs_rad']:.10f}")
    print(f"phi_abs_deg       = {pt['phi_abs_deg']:.6f}")
    print(f"r_center          = {pt['r_center']:.10f} m")
    print(f"x_center          = {pt['x_center']:.10f} m")
    print(f"y_center          = {pt['y_center']:.10f} m")
    print(f"gap_id_in_template= {pt['gap_id_in_template']}")
    print(f"repeat_id         = {pt['repeat_id']}")
    print("=" * 80)


# =========================================================
# 11. 构造 dump 管理器
# =========================================================
def build_dumps_manager(config):
    dumps_manager = Dumps()
    for dump_config in config['DumpPara']['modules']:
        if dump_config['type'] == 'StepDump':
            step_dump = StepDump(
                dump_config['start_time'],
                dump_config['end_time'],
                dump_config['interval_time'],
                dump_config['tracked_particle_id_global'],
                save_folder=dump_config['save_folder']
            )
            dumps_manager.add_dump(step_dump)

        elif dump_config['type'] == 'PositionDump':
            for dump_azimuth in dump_config['dump_azimuth']:
                position_dump = PositionDump(
                    dump_azimuth,
                    dump_config['start_time'],
                    dump_config['end_time'],
                    dump_config['num_particles_to_dump_global'],
                    save_folder=dump_config['save_folder']
                )
                dumps_manager.add_dump(position_dump)

        elif dump_config['type'] == 'StepDumpBunch':
            bunch_dump = StepDumpBunch(
                dump_config['start_time'],
                dump_config['end_time'],
                dump_config['interval_time'],
                save_folder=dump_config['save_folder']
            )
            dumps_manager.add_dump(bunch_dump)

        elif dump_config['type'] == 'PositionDumpBunch':
            for dump_azimuth in dump_config['dump_azimuth']:
                bunch_dump = PositionDumpBunch(
                    dump_azimuth,
                    dump_config['start_time'],
                    dump_config['end_time'],
                    save_folder=dump_config['save_folder']
                )
                dumps_manager.add_dump(bunch_dump)

    return dumps_manager


# =========================================================
# 12. 初始化全局场图与 SEO 参数
# =========================================================
def build_global_parameters(config, target_ek):
    BmapPATHName = config['BmapAndSEO']['maps']
    max_order = config['BmapAndSEO']['max_order']
    SEOPATHName = os.path.join(BmapPATHName, 'resultsSEO')
    EmapPATHName = config['Emap']['maps']

    SEO_inj, BmapPath, SEO_info = LoadSEOParams(os.path.join(SEOPATHName, "SEO_ini.txt"), target_ek)
    SEO_r_info = np.loadtxt(os.path.join(SEOPATHName, "SEO_r.txt"))
    SEO_fi_axis = SEO_r_info[0, 1:]
    SEO_Ek_axis = SEO_r_info[1:, 0]
    SEO_r_matrix = SEO_r_info[1:, 1:]

    GlobalParameters = FFAG_GlobalParameters()
    BMap = FFAG_Bfield_analytical(os.path.join(BmapPath, "config_Bmap.json"), max_order)
    EMap = FFAG_EField_new(EmapPATHName, config['Emap']['enable'])

    GlobalParameters.AddBMap(BMap)
    GlobalParameters.AddEMap(EMap)
    GlobalParameters.AddSEOInfo(SEO_info)

    if config['Aperture']['enable']:
        GlobalParameters.AddAperture(
            config['Aperture']['Rmin'],
            config['Aperture']['Rmax'],
            config['Aperture']['Zmin'],
            config['Aperture']['Zmax']
        )

    return {
        "GlobalParameters": GlobalParameters,
        "BMap": BMap,
        "EMap": EMap,
        "SEO_inj": SEO_inj,
        "SEO_info": SEO_info,
        "SEO_fi_axis": SEO_fi_axis,
        "SEO_Ek_axis": SEO_Ek_axis,
        "SEO_r_matrix": SEO_r_matrix,
        "SEOPATHName": SEOPATHName,
        "BmapPATHName": BmapPATHName,
        "EmapPATHName": EmapPATHName,
    }


# =========================================================
# 13. 由模板粒子构造膜位置单粒子，并转为 Boris 形式
# =========================================================
def build_reverse_particle_from_gap_by_template(
    config,
    target_ek,
    target_phi,
    target_r,
    reverse_vr=True,
    reverse_vphi=True,
    reverse_vz=True,
    set_zero_transverse_offsets=True,
    verbose=True
):
    """
    用 GenerateBunches 生成一个模板粒子，然后把它改到指定 gap 点。
    这样可以尽量避免你我对 array_dist_p 各列定义不一致的问题。

    注意：
    1. 这里仍然假设 array_dist_p 至少兼容下面这些列索引。
    2. 如果你的 array_dist_p 列定义和这里不一致，需要你自己按实际修改。
    """

    BunchPara = config['BunchPara'].copy()
    check = config['check']
    PaintPara = config['Paint']

    BmapPATHName = config['BmapAndSEO']['maps']
    SEOPATHName = os.path.join(BmapPATHName, 'resultsSEO')
    EmapPATHName = config['Emap']['maps']

    BunchPara['PlotFlag'] = check['CheckBunch']
    BunchPara['PlotRMS'] = check['PlotRMS']
    BunchPara['BmapPATHName'] = BmapPATHName
    BunchPara['SEO'] = SEOPATHName
    BunchPara["InjectEk"] = target_ek
    BunchPara["InjPosition"] = target_phi
    BunchPara["PaintEnable"] = PaintPara['enable']
    BunchPara["PaintMaxNum"] = PaintPara['MaxBunchNum']
    BunchPara["PaintTimeInterval"] = PaintPara['TimeInterval']
    BunchPara["PaintCurve"] = PaintPara['Curve']
    BunchPara["EmapPATHName"] = EmapPATHName

    # 只生成 1 个模板粒子
    # 若你的 GenerateBunches 默认会生成很多粒子，可考虑在配置里把粒子数改成 1
    array_dist_p = GenerateBunches(BunchPara)
    if array_dist_p.ndim != 2 or array_dist_p.shape[0] < 1:
        raise RuntimeError("GenerateBunches(BunchPara) 未返回合法二维粒子数组。")

    array_dist_p = array_dist_p[:1, :].copy()

    # -----------------------------------------------------
    # 下面这些列定义需要你结合自己的程序核对
    # -----------------------------------------------------
    # 你前面代码中多次直接用 array_dist_p[:,0:4] 做 x,px,z,pz 的测试，
    # 因此这里暂按如下假设：
    #
    # col 0 : radial offset or x-like coordinate
    # col 1 : radial momentum / px-like
    # col 2 : z
    # col 3 : pz
    #
    # 同时假设其中还有某列代表 Ek、某列代表 azimuth / inj_t 等。
    #
    # 由于这些列你没有完整给出，这里不强行猜全部，
    # 只做“尽可能少改”的模板构造：
    #   1) 让横纵偏移与横纵动量清零
    #   2) 通过 BunchPara["InjectEk"], BunchPara["InjPosition"] 把能量和起始方位放进去
    #
    # 也就是说，这个模板法主要依赖 GenerateBunches 已经知道如何把 InjectEk / InjPosition
    # 映射为内部正确的粒子初值。
    # -----------------------------------------------------

    if set_zero_transverse_offsets:
        if array_dist_p.shape[1] >= 4:
            array_dist_p[0, 0] = 0.0
            array_dist_p[0, 1] = 0.0
            array_dist_p[0, 2] = 0.0
            array_dist_p[0, 3] = 0.0

    conv = FFAG_ConversionTools()
    array_dist_v_boris = conv.ConvertPrzek2Vrzek_boris(array_dist_p)

    # 这里非常关键：把 Boris 速度分量反向
    # 这是你代码里已经暗示过的列位置
    if reverse_vr:
        array_dist_v_boris[:, 1] *= -1
    if reverse_vphi:
        array_dist_v_boris[:, 3] *= -1
    if reverse_vz:
        array_dist_v_boris[:, 5] *= -1

    # 分配 Global_ID
    attr_mgr = FFAG_ManageBunchAttribute()
    if 'Global_ID' in attr_mgr.Attribute:
        array_dist_v_boris[:, attr_mgr.Attribute['Global_ID']] = 0

    if verbose:
        print("=" * 80)
        print("Reverse particle generated from template bunch:")
        print("Used strategy:")
        print("  1) Generate one template particle by GenerateBunches")
        print("  2) Use InjectEk / InjPosition = target gap point")
        print("  3) Clear transverse offsets if requested")
        print("  4) Convert to Boris variables")
        print("  5) Reverse v-components at columns [1, 3, 5]")
        print("array_dist_p.shape       =", array_dist_p.shape)
        print("array_dist_v_boris.shape =", array_dist_v_boris.shape)
        print("=" * 80)

    return array_dist_p, array_dist_v_boris


# =========================================================
# 14. 直接运行 Boris 反向跟踪
# =========================================================
def run_reverse_tracking_boris(
    config,
    target_ek,
    array_dist_v_boris,
    dump_override_folder=None,
    force_disable_sc=True
):
    init_pack = build_global_parameters(config, target_ek)

    GlobalParameters = init_pack["GlobalParameters"]
    SEO_info = init_pack["SEO_info"]
    SEO_fi_axis = init_pack["SEO_fi_axis"]
    SEO_Ek_axis = init_pack["SEO_Ek_axis"]
    SEO_r_matrix = init_pack["SEO_r_matrix"]

    dumps_manager = build_dumps_manager(config)

    if dump_override_folder is not None:
        # 简单做法：统一修改所有 dump 的保存目录
        # 你的 Dumps 内部对象若属性名不同，可自行调整
        for dump_obj in dumps_manager.dumps:
            if hasattr(dump_obj, "save_folder"):
                dump_obj.save_folder = dump_override_folder
        os.makedirs(dump_override_folder, exist_ok=True)

    LocalParameters = dict()
    LocalParameters['stop_condition'] = config['track']['stop_condition']
    LocalParameters['time_step'] = config['track']['time_step']
    LocalParameters['step_dumps'] = dumps_manager

    if force_disable_sc:
        LocalParameters['enable_SC'] = False
    else:
        LocalParameters['enable_SC'] = config['SC']['enable_SC']

    LocalParameters['SC_type'] = config['SC']['SC_type']
    LocalParameters['SC_step'] = config['SC']['step']
    SC_grid_size = tuple(config['SC']['grid_size_rfz'])

    BunchPara = config['BunchPara']
    rk = FFAG_RungeKutta()

    BunchObj = FFAG_Bunch(array_dist_v_boris, marcosize=BunchPara['ParticleDensity'])
    BunchObj.SEO_fi_axis = SEO_fi_axis
    BunchObj.SEO_Ek_axis = SEO_Ek_axis
    BunchObj.SEO_r_matrix = SEO_r_matrix
    BunchObj.SEO_perimeter = SEO_info[:, 8]

    t_start = 0.0

    print("=" * 80)
    print("Start reverse tracking with Boris solver")
    print(f"target_ek         = {target_ek:.6f} MeV")
    print(f"time_step         = {LocalParameters['time_step']}")
    print(f"enable_SC         = {LocalParameters['enable_SC']}")
    print(f"SC_grid_size      = {SC_grid_size}")
    print(f"n_particles_local = {BunchObj.LocalBunch.shape[0]}")
    print("=" * 80)

    rk.rk4_solve_dt_bunch3_withSC_boris(
        FunctionForAccelerationBunch_dt,
        t_start, BunchObj,
        GlobalParameters, LocalParameters
    )

    return BunchObj



# =========================================================
# 15. 主程序
# =========================================================
def main():
    # -----------------------------------------------------
    # 用户可调参数
    # -----------------------------------------------------
    bmap_1p1_folder = './Bmap_inj_strip_1p15'
    output_folder = './output/reverse_track_test_strip_1p15'

    config_track_json = './config_track.json'

    npz_path = os.path.join(bmap_1p1_folder, 'Bz_order_0.npz')
    npz_key = 'Bz_Taylor_coeff'
    bmap_json = os.path.join(bmap_1p1_folder, 'config_Bmap.json')

    seo_folder = os.path.join(bmap_1p1_folder, 'resultsSEO')
    seo_r_path = os.path.join(seo_folder, 'SEO_r.txt')
    seo_bz_path = os.path.join(seo_folder, 'Bz.txt')
    seo_pr_path = os.path.join(seo_folder, 'SEO_pr.txt')

    threshold = 0.08
    min_gap_width = 2
    thin = 10

    total_cells = 16
    template_cells = 2

    preview_orbit_index = 0
    target_orbit_index = 0
    target_gap_point_index = 0

    plot_flag = True
    dump_folder = output_folder
    enable_sc = False

    # -----------------------------------------------------
    # 读取主配置
    # -----------------------------------------------------
    config = load_config_from_json(config_track_json)

    # -----------------------------------------------------
    # 读取 Bmap（单 cell）
    # -----------------------------------------------------
    data = np.load(npz_path)[npz_key]

    with open(bmap_json, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    Bmap_cfg = config_data['Bmap']
    rmin_max_step_m = Bmap_cfg["rmin_max_step_m"]
    theta_step_rad = Bmap_cfg["theta_step_rad"]
    NSector_from_json = int(Bmap_cfg["NSector"])

    if NSector_from_json != total_cells:
        print(f"[warning] config_Bmap.json 中 NSector={NSector_from_json}, "
              f"但你手动设置 total_cells={total_cells}。当前以后者为准。")

    rmin, rmax, rstep = rmin_max_step_m
    n_r_steps = int((rmax - rmin) / rstep) + 1
    n_fi_steps = int((2.0 * np.pi / total_cells) / theta_step_rad) + 1

    fi_axis_single_cell = np.linspace(0.0, 2.0 * np.pi / total_cells, n_fi_steps)
    r_axis = np.linspace(rmin, rmax, n_r_steps)

    if data.shape != (len(r_axis), len(fi_axis_single_cell)):
        raise RuntimeError(
            f"data.shape = {data.shape}, expected = ({len(r_axis)}, {len(fi_axis_single_cell)})"
        )

    # -----------------------------------------------------
    # 读取 SEO_r / SEO_pr / SEO_Bz
    # -----------------------------------------------------
    SEO_azimuth, SEO_EK, SEO_r = load_seo_r_data(seo_r_path)
    print(f"SEO_r loaded: n_energy = {len(SEO_EK)}, n_azimuth = {len(SEO_azimuth)}")

    SEO_pr_azimuth, SEO_pr_EK, SEO_pr = load_seo_pr_data(seo_pr_path)
    print(f"SEO_pr loaded: n_energy = {len(SEO_pr_EK)}, n_azimuth = {len(SEO_pr_azimuth)}")

    if not np.allclose(SEO_azimuth, SEO_pr_azimuth):
        raise RuntimeError("SEO_r.txt 与 SEO_pr.txt 的 azimuth 轴不一致。")
    if not np.allclose(SEO_EK, SEO_pr_EK):
        raise RuntimeError("SEO_r.txt 与 SEO_pr.txt 的能量轴不一致。")

    theta_two_cell_rad, bz_ek, SEO_Bz_two_cell = load_seo_bz_two_cell_data(seo_bz_path)
    print(f"SEO_Bz(2-cell) loaded: n_orbit = {SEO_Bz_two_cell.shape[0]}, n_pts = {SEO_Bz_two_cell.shape[1]}")

    template_period_expected = 2.0 * np.pi * (template_cells / total_cells)
    template_period_file = theta_two_cell_rad[-1] - theta_two_cell_rad[0] + np.median(np.diff(theta_two_cell_rad))
    print(f"2-cell period expected = {template_period_expected:.10f} rad")
    print(f"2-cell period from file = {template_period_file:.10f} rad")

    # -----------------------------------------------------
    # 能量匹配：Bz 每一行 -> SEO_r / SEO_pr 某一行
    # -----------------------------------------------------
    seo_row_indices = match_bz_rows_to_seo_rows(
        bz_ek=bz_ek,
        seo_ek=SEO_EK,
        tol=1e-6
    )

    # -----------------------------------------------------
    # gap 探测
    # -----------------------------------------------------
    seo_gap_results = detect_gaps_on_all_seo_from_two_cell_bz(
        theta_two_cell_rad=theta_two_cell_rad,
        bz_ek=bz_ek,
        SEO_Bz_two_cell=SEO_Bz_two_cell,
        seo_row_indices=seo_row_indices,
        SEO_azimuth=SEO_azimuth,
        SEO_EK=SEO_EK,
        SEO_r=SEO_r,
        total_cells=total_cells,
        template_cells=template_cells,
        threshold=threshold,
        min_gap_width=min_gap_width
    )

    print("=" * 80)
    print("Gap detection results based on SEO_Bz(2-cell):")
    for i, item in enumerate(seo_gap_results[:10]):
        print(f"[orbit {i}] Ek = {item['Ek']:.6f}, n_gap_in_2cell = {len(item['gap_list_template'])}")
        for gap in item["gap_list_template"]:
            print(
                f"    Gap {gap['id']}: "
                f"start={gap['start_deg']:.3f} deg, "
                f"end={gap['end_deg']:.3f} deg, "
                f"center={gap['center_deg']:.3f} deg, "
                f"width={gap['width_deg']:.3f} deg, "
                f"wraps={gap['wraps']}"
            )
    print("=" * 80)

    # -----------------------------------------------------
    # 检查 preview 轨道
    # -----------------------------------------------------
    if not (0 <= preview_orbit_index < len(seo_gap_results)):
        raise RuntimeError(f"preview_orbit_index={preview_orbit_index} 越界。")

    preview_item = seo_gap_results[preview_orbit_index]

    # -----------------------------------------------------
    # 选中目标 foil / gap 点
    # -----------------------------------------------------
    if not (0 <= target_orbit_index < len(seo_gap_results)):
        raise RuntimeError(f"target_orbit_index={target_orbit_index} 越界。")

    target_item = seo_gap_results[target_orbit_index]

    if len(target_item["gap_points_full_ring"]) == 0:
        raise RuntimeError(f"target_orbit_index={target_orbit_index} 对应轨道没有找到 gap 点。")

    if not (0 <= target_gap_point_index < len(target_item["gap_points_full_ring"])):
        raise RuntimeError(
            f"target_gap_point_index={target_gap_point_index} 越界。"
        )

    target_pt = target_item["gap_points_full_ring"][target_gap_point_index]

    target_ek = target_item["Ek"]
    target_phi = target_pt["phi_abs_rad"]
    target_r = target_pt["r_center"]
    seo_row_idx = target_item["seo_row_index"]

    print("=" * 80)
    print("Selected stripping foil / gap point:")
    print(f"Ek                 = {target_ek:.6f} MeV")
    print(f"seo_row_idx        = {seo_row_idx}")
    print(f"phi_abs_rad        = {target_phi:.10f}")
    print(f"phi_abs_deg        = {np.rad2deg(target_phi):.6f}")
    print(f"r_center           = {target_r:.10f} m")
    print(f"x_center           = {target_pt['x_center']:.10f} m")
    print(f"y_center           = {target_pt['y_center']:.10f} m")
    print(f"gap_id_in_template = {target_pt['gap_id_in_template']}")
    print(f"repeat_id          = {target_pt['repeat_id']}")
    print("=" * 80)

    # -----------------------------------------------------
    # 构造整圈 top view 背景
    # -----------------------------------------------------
    data_full, fi_axis_full, mesh_x, mesh_y = build_full_ring_bmap(
        data=data,
        r_axis=r_axis,
        fi_axis_single_cell=fi_axis_single_cell,
        total_cells=total_cells,
        thin=thin
    )

    norm = TwoSlopeNorm(
        vmin=data_full.min(),
        vcenter=0.0,
        vmax=data_full.max()
    )

    # -----------------------------------------------------
    # 画图
    # -----------------------------------------------------
    if plot_flag:
        plot_two_cell_bz_with_gaps(
            theta_rad=theta_two_cell_rad,
            bz_line=preview_item["bz_two_cell"],
            gap_list=preview_item["gap_list_template"],
            ek=preview_item["Ek"]
        )

        plot_topview_with_seo_detected_gap_points(
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            data_full=data_full,
            norm=norm,
            SEO_azimuth=SEO_azimuth,
            SEO_r=SEO_r,
            seo_gap_results=seo_gap_results,
            plot_seo_orbits=True,
            every_n_orbit=8,
            plot_gap_points=True,
            every_n_gap_orbit=1,
            selected_point=target_pt
        )

    # -----------------------------------------------------
    # 在目标 gap 点构造反向单粒子
    # array_dist_p 前几列定义：
    #   0 -> r
    #   1 -> pr
    #   2 -> z
    #   3 -> pz
    #   4 -> fi
    #   5 -> Ek
    # -----------------------------------------------------
    BmapPATHName = bmap_1p1_folder
    max_order = config['BmapAndSEO']['max_order']
    SEOPATHName = seo_folder
    EmapPATHName = config['Emap']['maps']

    Ek_MeV = target_ek
    Azimuth_start = target_phi

    check = config['check']
    PaintPara = config['Paint']
    BunchPara = config['BunchPara'].copy()

    PaintEnable = PaintPara['enable']
    PaintMaxNum = PaintPara['MaxBunchNum']
    PaintTimeInterval = PaintPara['TimeInterval']
    PaintCurve = PaintPara['Curve']

    BunchPara['PlotFlag'] = check['CheckBunch']
    BunchPara['PlotRMS'] = check['PlotRMS']
    BunchPara['BmapPATHName'] = BmapPATHName
    BunchPara['SEO'] = SEOPATHName
    BunchPara["InjectEk"] = Ek_MeV
    BunchPara["InjPosition"] = Azimuth_start
    BunchPara["PaintEnable"] = PaintEnable
    BunchPara["PaintMaxNum"] = PaintMaxNum
    BunchPara["PaintTimeInterval"] = PaintTimeInterval
    BunchPara["PaintCurve"] = PaintCurve
    BunchPara["EmapPATHName"] = EmapPATHName

    # 先生成一个模板粒子，只为继承数组长度和其余附加列
    array_dist_p = GenerateBunches(BunchPara)
    if array_dist_p.ndim != 2 or array_dist_p.shape[0] < 1:
        raise RuntimeError("GenerateBunches(BunchPara) 未返回合法二维粒子数组。")
    array_dist_p = array_dist_p[:1, :].copy()

    # -----------------------------------------------------
    # 由 SEO_pr.txt 在 foil 点方位处插值得到 pr0
    # z, pz 先取 0
    # -----------------------------------------------------
    r0 = target_r
    fi0 = target_phi
    Ek0 = target_ek

    pr0 = np.interp(fi0, SEO_azimuth, SEO_pr[seo_row_idx, :])
    z0 = 0.0
    pz0 = 0.0

    array_dist_p[0, 0] = r0
    array_dist_p[0, 1] = pr0
    array_dist_p[0, 2] = z0
    array_dist_p[0, 3] = pz0
    array_dist_p[0, 4] = fi0
    array_dist_p[0, 5] = Ek0

    conv = FFAG_ConversionTools()
    array_dist_v_boris = conv.ConvertPrzek2Vrzek_boris(array_dist_p)

    attr_mgr = FFAG_ManageBunchAttribute()
    if 'Global_ID' in attr_mgr.Attribute:
        array_dist_v_boris[:, attr_mgr.Attribute['Global_ID']] = 0

    print("=" * 80)
    print("Reverse particle generated from stripping foil point:")
    print(f"  r0             = {r0:.10f} m")
    print(f"  fi0            = {fi0:.10f} rad  ({np.rad2deg(fi0):.6f} deg)")
    print(f"  Ek0            = {Ek0:.10f} MeV")
    print(f"  pr0(interp)    = {pr0:.10e}")
    print(f"  z0             = {z0:.10e} m")
    print(f"  pz0            = {pz0:.10e}")
    print(f"  array_dist_p.shape       = {array_dist_p.shape}")
    print(f"  array_dist_v_boris.shape = {array_dist_v_boris.shape}")
    print("  reverse columns: [1, 3, 5]")
    print("=" * 80)

    # -----------------------------------------------------
    # 构造 dump 管理器
    # -----------------------------------------------------
    os.makedirs(dump_folder, exist_ok=True)

    dumps_manager = Dumps()
    for dump_config in config['DumpPara']['modules']:
        if dump_config['type'] == 'StepDump':
            step_dump = StepDump(
                dump_config['start_time'],
                dump_config['end_time'],
                dump_config['interval_time'],
                dump_config['tracked_particle_id_global'],
                save_folder=dump_folder
            )
            dumps_manager.add_dump(step_dump)

        elif dump_config['type'] == 'PositionDump':
            for dump_azimuth in dump_config['dump_azimuth']:
                position_dump = PositionDump(
                    dump_azimuth,
                    dump_config['start_time'],
                    dump_config['end_time'],
                    dump_config['num_particles_to_dump_global'],
                    save_folder=dump_folder
                )
                dumps_manager.add_dump(position_dump)

        elif dump_config['type'] == 'StepDumpBunch':
            bunch_dump = StepDumpBunch(
                dump_config['start_time'],
                dump_config['end_time'],
                dump_config['interval_time'],
                save_folder=dump_folder
            )
            dumps_manager.add_dump(bunch_dump)

        elif dump_config['type'] == 'PositionDumpBunch':
            for dump_azimuth in dump_config['dump_azimuth']:
                bunch_dump = PositionDumpBunch(
                    dump_azimuth,
                    dump_config['start_time'],
                    dump_config['end_time'],
                    save_folder=dump_folder
                )
                dumps_manager.add_dump(bunch_dump)

    # -----------------------------------------------------
    # 初始化全局参数
    # -----------------------------------------------------
    # -----------------------------------------------------
    # 反向跟踪：速度反向，但 dt 保持为正
    # Boris 变量中速度列沿用你现有程序：
    #   1, 3, 5
    # -----------------------------------------------------
    array_dist_v_boris[:, 1] *= -1
    array_dist_v_boris[:, 3] *= -1
    array_dist_v_boris[:, 5] *= -1

    SEO_inj, _, SEO_info = LoadSEOParams(os.path.join(SEOPATHName, "SEO_ini.txt"), Ek_MeV)

    SEO_r_info = np.loadtxt(os.path.join(SEOPATHName, "SEO_r.txt"))
    SEO_fi_axis = SEO_r_info[0, 1:]
    SEO_Ek_axis = SEO_r_info[1:, 0]
    SEO_r_matrix = SEO_r_info[1:, 1:]

    GlobalParameters = FFAG_GlobalParameters()
    BMap = FFAG_Bfield_analytical(os.path.join(BmapPATHName, "config_Bmap.json"), max_order)
    EMap = FFAG_EField_new(EmapPATHName, config['Emap']['enable'])

    GlobalParameters.AddBMap(BMap)
    GlobalParameters.AddEMap(EMap)
    GlobalParameters.AddSEOInfo(SEO_info)

    config['Aperture']['Rmin']=BMap.r_min; config['Aperture']['Rmax']=BMap.r_max

    if config['Aperture']['enable']:
        GlobalParameters.AddAperture(
            config['Aperture']['Rmin'],
            config['Aperture']['Rmax'],
            config['Aperture']['Zmin'],
            config['Aperture']['Zmax']
        )

    # -----------------------------------------------------
    # 跟踪局部参数
    # -----------------------------------------------------
    LocalParameters = dict()
    LocalParameters['stop_condition'] = config['track']['stop_condition']
    LocalParameters['time_step'] = config['track']['time_step']
    LocalParameters['step_dumps'] = dumps_manager
    LocalParameters['enable_SC'] = enable_sc
    LocalParameters['SC_type'] = config['SC']['SC_type']
    LocalParameters['SC_step'] = config['SC']['step']

    SC_grid_size = tuple(config['SC']['grid_size_rfz'])

    # -----------------------------------------------------
    # 建 bunch 并进行 Boris 反向跟踪
    # -----------------------------------------------------
    rk = FFAG_RungeKutta()

    BunchObj = FFAG_Bunch(array_dist_v_boris, marcosize=BunchPara['ParticleDensity'])
    BunchObj.SEO_fi_axis = SEO_fi_axis
    BunchObj.SEO_Ek_axis = SEO_Ek_axis
    BunchObj.SEO_r_matrix = SEO_r_matrix
    BunchObj.SEO_perimeter = SEO_info[:, 8]

    t_start = 0.0

    print("=" * 80)
    print("Start reverse tracking with Boris solver")
    print(f"target_ek         = {Ek_MeV:.6f} MeV")
    print(f"time_step         = {LocalParameters['time_step']}")
    print(f"enable_SC         = {LocalParameters['enable_SC']}")
    print(f"SC_grid_size      = {SC_grid_size}")
    print(f"n_particles_local = {BunchObj.LocalBunch.shape[0]}")
    print("=" * 80)

    rk.rk4_solve_dt_bunch3_withSC_boris(
        FunctionForAccelerationBunch_dt,
        t_start, BunchObj,
        GlobalParameters, LocalParameters
    )

    # 读取刚生成的轨迹文件
    # 每列依次为r,r',z,z',fi,Ek, ...画出轨迹xy叠加在图上
    trajectory_path = os.path.join(output_folder, 'particle_0_rank_0.npz')
    data_out = np.load(trajectory_path)['particles']
    r=data_out[:,0]
    fi=data_out[:,4]
    plt.figure(10)
    plt.plot(r*np.cos(fi), r*np.sin(fi))


    if plot_flag:
        plt.show()




if __name__ == '__main__':
    main()
