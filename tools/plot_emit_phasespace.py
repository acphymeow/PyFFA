#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm

from pptx import Presentation
from pptx.util import Inches, Pt

from FFAG_MathTools import Bilinear_interp_2D_vect_uniform
from FFAG_Distribution import generate_ellipse_points, generate_ellipse_points_from_emittance

# —— 中文字体支持，可按需删除 ——
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


# ===================== 配置区 =====================
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize']  = 15
plt.rcParams['axes.titlesize']  = 15


# 自动扫描的基础目录：其下包含 TBT_Dump_XXXX/merged_files
# BASE_DIR = Path("./output/simulation1")
# BASE_DIR = Path("./output/simulation_plot0")
# BASE_DIR = Path("../output/FD16_paint_1_1_newE_sp0")
# BASE_DIR = Path("../output/FD16_paint_1_1")
# BASE_DIR = Path("../output/FD16_paint_1_1_newE")
# BASE_DIR = Path("../output/paint_optimize/paint_FD16_0p47_0p47_cor")
# BASE_DIR = Path("../output/paint_optimize/paint_FD16_0p82_0p82_cor")
# BASE_DIR = Path("../output/paint_optimize/paint_FD16_0p73_0p73_cor")

# BASE_DIR = Path("../output/paint_optimize/paint_FD16_260_70_0p45")
# BASE_DIR = Path("../output/max_450/max450_1p40")
# BASE_DIR = Path("../output/simulation_300_2_450/fix300_full_1p40")
# BASE_DIR = Path("../output/paint_region/cu_n_r35_z8_ar062_az062")
# BASE_DIR = Path("../output/paint_region/cu_n_r25_z4_ar062_az162")
# ppt_path = "FFAG_emittance_phase_report_cor_35_8.pptx"

# BASE_DIR = Path("../output/paint_optimize_alpha/cu_n_r20_z2_ar060_az130")
# ppt_path = "FFAG_emittance_phase_report_cor_20_2_060_130.pptx"
# CASE_DIR = Path("./HPC_ffa_tracking/optimize_paint_alpha/au_r20_z2_ar025_az130")
# BASE_DIR = CASE_DIR / "output/au_r20_z2_ar025_az130"

CASE_DIR = Path("./HPC_dir")
BASE_DIR = CASE_DIR / "Longitudinal_0p5"
ppt_path = CASE_DIR / "FFAG_emittance_phase_report_au_r20_z2_ar025_az130.pptx"

# ---- 读入 SEO 闭轨 & Twiss 数据 ----
SEO_DIR = Path("./HPC_ffa_tracking/Bmaps/Bmap_FD16/resultsSEO_noError")

# CSV 列号定义（与你当前输出保持一致）
COL_X, COL_PX = 0, 1
COL_Y, COL_PY = 2, 3
COL_FI, COL_EK = 4, 5
COL_RF_PHASE = 9     # RF 相位列
COL_T = -1           # 时间列（最后一列）

# 质子静质量能量（如为其它粒子请修改）
M0C2_MEV = 938.2720813  # MeV

# 存放 PNG 图像的缓存目录
FIG_DIR = CASE_DIR / "ppt_figure_emit"
FigText_xpx = "(d) x-px"
FigText_ypy = "(e) z-pz"
FigText_xy = "(f) x-z"
Fig_emit_panel = "(a) with SC"
# FigText_xpx = "(a) x-px"
# FigText_ypy = "(b) z-pz"
# FigText_xy = "(c) x-z"
# Fig_emit_panel = "(b) without SC"
# ppt_path = "FFAG_emittance_phase_report_47.pptx"

# ===================== 工具函数 =====================

def rms_emittance(x, px):
    """
    计算几何 RMS 发射度（未归一化），定义:
        eps_rms = sqrt( <(x-<x>)^2> * <(px-<px>)^2> - <(x-<x>)(px-<px>)>^2 )
    返回单位: m·rad
    """
    x = np.asarray(x)
    px = np.asarray(px)
    xm = x - x.mean()
    pm = px - px.mean()
    s_xx = np.mean(xm * xm)
    s_pp = np.mean(pm * pm)
    s_xp = np.mean(xm * pm)
    eps = np.sqrt(max(s_xx * s_pp - s_xp * s_xp, 0.0))
    return eps


def auto_collect_phase_dirs(base_dir: Path, turn_min=None, turn_max=None):
    """
    自动扫描 base_dir 下的 TBT_Dump_XXXX 目录，
    返回按 turn 排序的 { "Turn_XXXX": "path/to/merged_files" } 字典
    """
    phase_dirs = {}
    pattern = re.compile(r"TBT_Dump_(\d+)$")

    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue
        m = pattern.match(subdir.name)
        if not m:
            continue
        turn = int(m.group(1))
        if turn_min is not None and turn < turn_min:
            continue
        if turn_max is not None and turn > turn_max:
            continue
        merged = subdir / "merged_files"
        if merged.exists():
            phase_dirs[f"Turn_{turn}"] = str(merged)

    # 按 turn 数字大小排序
    phase_dirs = dict(
        sorted(phase_dirs.items(), key=lambda item: int(item[0].split("_")[1]))
    )
    return phase_dirs


def add_five_pngs_to_ppt(prs, png_xpx, png_ypy, png_phaseEk,
                         png_xy, png_phaseEk_wide, title=None):

    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # ====== 标题 ======
    if title is not None:
        tx_box = slide.shapes.add_textbox(
            Inches(0.3),
            Inches(0.2),
            Inches(9.4),
            Inches(0.6)
        )
        tf = tx_box.text_frame
        tf.text = title
        tf.paragraphs[0].font.size = Pt(24)

    # ====== 左侧四张小图（5:4 等比） ======
    w_small = Inches(3.35)
    h_small = w_small * 4.0 / 5.0   # 等比缩放

    left_col1 = Inches(0.3)
    left_col2 = Inches(0.3 + 3.35 + 0.25)  # = 3.90"

    top_row1 = Inches(1.1)
    gap_row  = Inches(0.20)
    top_row2 = top_row1 + h_small + gap_row   # ✅ 用 EMU 相加，不再套 Inches()

    lefts_small = [left_col1, left_col2, left_col1, left_col2]
    tops_small  = [top_row1,  top_row1,  top_row2,  top_row2]

    for path, left, top in zip(
        [png_xpx, png_ypy, png_phaseEk, png_xy],
        lefts_small, tops_small
    ):
        slide.shapes.add_picture(str(path), left, top,
                                 width=w_small, height=h_small)

    # ====== 右侧宽图（80% 宽度、等比） ======
    w_wide = Inches(2.40)
    h_wide = w_wide * 6.0 / 4.0

    left_wide = Inches(7.30)
    top_wide  = Inches(1.1)

    slide.shapes.add_picture(
        str(png_phaseEk_wide),
        left_wide, top_wide,
        width=w_wide, height=h_wide
    )



def add_emittance_page(
    prs,
    png_eps99,
    png_eps_rms,
    title="Emittance evolution vs turn"
):
    """
    在一页 PPT 中把 90% 和 RMS 发射度图左右排布。
    自动适配 4:3 模板。
    """

    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    # ---------- 标题 ----------
    tx = slide.shapes.add_textbox(
        Inches(0.3),  # 左边距
        Inches(0.2),  # 顶部
        Inches(9.4),  # 宽度
        Inches(0.6)   # 高度
    )
    tf = tx.text_frame
    tf.text = title
    tf.paragraphs[0].font.size = Pt(26)

    # ---------- 图像布局 ----------
    # 图宽度（左右各一张）
    w_plot = Inches(4.35)          # 约占页面一半宽
    h_plot = w_plot * 4.0 / 7.0     # 保持 figsize=(7,4) 比例

    top = Inches(1.0)               # 上方对齐

    # 左图位置
    left1 = Inches(0.3)
    slide.shapes.add_picture(
        str(png_eps99),
        left1, top,
        width=w_plot, height=h_plot
    )

    # 右图位置（留 0.3" 间隔）
    left2 = left1 + w_plot + Inches(0.3)
    slide.shapes.add_picture(
        str(png_eps_rms),
        left2, top,
        width=w_plot, height=h_plot
    )


def hist2d_with_projections(
    x, y,
    xlabel="", ylabel="", title="",
    xlim=None, ylim=None,
    bins=(256, 256),
    norm=None,
    figsize=(5, 4),
    proj_frac=0.18,        # 投影带高度/宽度占比（相对主图高度/宽度）
    proj_alpha=0.9,
    proj_lw=1.0,
    proj_fill=False,       # 是否填充投影带
    proj_fill_alpha=0.15,
    proj_color_x='k',      # 底部投影颜色
    proj_color_y='k'       # 左侧投影颜色
):
    """
    画 2D hist2d 主图，并在底部/左侧加入投影分布（映射到主图内部的带状区域）。
    - x 为横轴数据，y 为纵轴数据（已是你希望显示的单位，比如 mm/mrad/deg/MeV）
    - 投影带是“画在主图坐标系内”的（不需要额外 axes），效果类似你给的参考代码。
    """
    x = np.asarray(x)
    y = np.asarray(y)

    fig, ax = plt.subplots(figsize=figsize)

    # ---- 主图 2D ----
    ax.hist2d(x, y, bins=bins, norm=norm)

    # ---- 轴范围 ----
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        xlim = ax.get_xlim()

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # ===============================
    # 底部投影：x 的直方图 → 映射到 y 的底部带状区
    # ===============================
    # 用与 xlim 一致的范围，保证投影与主图严格对齐
    histx, edgesx = np.histogram(x, bins=bins[0] if isinstance(bins, (tuple, list)) else bins, range=xlim)
    centersx = 0.5 * (edgesx[:-1] + edgesx[1:])
    maxx = max(histx.max(), 1)

    y_min, y_max = ylim
    total_h = y_max - y_min
    band_h = proj_frac * total_h
    base_y = y_min

    y_histx = base_y + band_h * (histx / maxx)

    ax.plot(centersx, y_histx, '-', lw=proj_lw, color=proj_color_x, alpha=proj_alpha)
    if proj_fill:
        ax.fill_between(centersx, base_y, y_histx, color=proj_color_x, alpha=proj_fill_alpha)

    # ===============================
    # 左侧投影：y 的直方图 → 映射到 x 的左侧带状区
    # ===============================
    histy, edgesy = np.histogram(y, bins=bins[1] if isinstance(bins, (tuple, list)) else bins, range=ylim)
    centersy = 0.5 * (edgesy[:-1] + edgesy[1:])
    maxy = max(histy.max(), 1)

    x_min, x_max = xlim
    total_w = x_max - x_min
    band_w = proj_frac * total_w
    base_x = x_min

    x_histy = base_x + band_w * (histy / maxy)

    # 注意：这是“投影曲线横着画”，x 变化、y 为 centersy
    ax.plot(x_histy, centersy, '-', lw=proj_lw, color=proj_color_y, alpha=proj_alpha)
    if proj_fill:
        ax.fill_betweenx(centersy, base_x, x_histy, color=proj_color_y, alpha=proj_fill_alpha)

    return fig, ax


def robust_width_centered(x, pct=99.0):
    """
    对已中心化的数据 x（均值≈0），返回 pct% 全宽：
        width = q_hi - q_lo,  其中 q_lo=(100-pct)/2, q_hi=100-q_lo
    """
    x = np.asarray(x)
    lo = (100.0 - pct) * 0.5
    hi = 100.0 - lo
    qlo, qhi = np.percentile(x, [lo, hi])
    return qhi - qlo

def rms_width_centered(x):
    """已中心化数据的 RMS 宽度（1σ），返回 std"""
    x = np.asarray(x)
    return np.std(x)


# ===================== 主程序 =====================
def main(
    ppt_path,
    base_dir=BASE_DIR,
    seo_dir=SEO_DIR,
    fig_dir=FIG_DIR,
    turn_min=None,
    turn_max=None,
):
    global BASE_DIR, FIG_DIR
    BASE_DIR = Path(base_dir).resolve()
    FIG_DIR = Path(fig_dir).resolve()
    seo_dir = Path(seo_dir).resolve()
    ppt_path = Path(ppt_path).resolve()

    required_seo_files = (
        "SEO_r.txt", "SEO_pr.txt",
        "AlphaFuncR.txt", "AlphaFuncZ.txt",
        "BetaFuncR.txt", "BetaFuncZ.txt",
    )
    missing_seo_files = [
        str(seo_dir / filename)
        for filename in required_seo_files
        if not (seo_dir / filename).is_file()
    ]
    if missing_seo_files:
        raise FileNotFoundError(
            "Missing SEO/Twiss files:\n  " + "\n  ".join(missing_seo_files)
        )
    if not BASE_DIR.is_dir():
        raise FileNotFoundError(f"TBT parent directory not found: {BASE_DIR}")

    SEO_r_info = np.loadtxt(seo_dir / "SEO_r.txt", skiprows=1)
    SEO_pr_info = np.loadtxt(seo_dir / "SEO_pr.txt", skiprows=1)
    SEO_alphaR_info = np.loadtxt(seo_dir / "AlphaFuncR.txt", skiprows=1)
    SEO_alphaZ_info = np.loadtxt(seo_dir / "AlphaFuncZ.txt", skiprows=1)
    SEO_betaR_info = np.loadtxt(seo_dir / "BetaFuncR.txt", skiprows=1)
    SEO_betaZ_info = np.loadtxt(seo_dir / "BetaFuncZ.txt", skiprows=1)
    # 缓存目录
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ppt_path.parent.mkdir(parents=True, exist_ok=True)

    # 自动收集所有 TBT_Dump_XXXX/merged_files
    if turn_min is not None and turn_max is not None and turn_min > turn_max:
        raise ValueError("turn_min cannot be greater than turn_max")

    phase_dirs = auto_collect_phase_dirs(BASE_DIR, turn_min, turn_max)
    if not phase_dirs:
        print(f"在 {BASE_DIR} 下没有找到任何 TBT_Dump_XXXX/merged_files 目录，退出。")
        return

    print("自动检测到的 TBT 目录：")
    for k, v in phase_dirs.items():
        print("  ", k, "→", v)

    # 新建 PPT
    prs = Presentation()

    # # ---- 读入 SEO 闭轨 & Twiss 数据 ----
    # SEO_r_info = np.loadtxt(os.path.join('./Bmap_FD16/resultsSEO_noError', "SEO_r.txt"), skiprows=1)
    # SEO_pr_info = np.loadtxt(os.path.join('./Bmap_FD16/resultsSEO_noError', "SEO_pr.txt"), skiprows=1)
    # SEO_alphaR_info = np.loadtxt(os.path.join('./Bmap_FD16/resultsSEO_noError', "AlphaFuncR.txt"), skiprows=1)
    # SEO_alphaZ_info = np.loadtxt(os.path.join('./Bmap_FD16/resultsSEO_noError', "AlphaFuncZ.txt"), skiprows=1)
    # SEO_betaR_info = np.loadtxt(os.path.join('./Bmap_FD16/resultsSEO_noError', "BetaFuncR.txt"), skiprows=1)
    # SEO_betaZ_info = np.loadtxt(os.path.join('./Bmap_FD16/resultsSEO_noError', "BetaFuncZ.txt"), skiprows=1)


    SEO_fi_axis, SEO_Ek_axis, SEO_r_matrix, SEO_pr_matrix = (
        SEO_r_info[0, 1:], SEO_r_info[1:, 0],
        SEO_r_info[1:, 1:], SEO_pr_info[1:, 1:]
    )

    twiss_fi_axis, twiss_Ek_axis, twiss_alphaR, twiss_alphaZ, twiss_betaR, twiss_betaZ = (
        SEO_alphaR_info[0, 1:], SEO_alphaR_info[1:, 0],
        SEO_alphaR_info[1:, 1:], SEO_alphaZ_info[1:, 1:],
        SEO_betaR_info[1:, 1:], SEO_betaZ_info[1:, 1:]
    )

    twiss_cell_fi = twiss_fi_axis[-1] - twiss_fi_axis[0]

    # ---- 统计量容器 ----
    pos_names = []
    epsx_90_list, epsy_90_list = [], []
    epsx_rms_list, epsy_rms_list = [], []
    epsx_90_norm_list, epsy_90_norm_list = [], []
    epsx_rms_norm_list, epsy_rms_norm_list = [], []
    mean_r, mean_Ek_r = [], []
    rf_width_90_list, rf_width_rms_list = [], []
    Ek_width_90_list, Ek_width_rms_list = [], []

    # ===================== 遍历各个 turn 的 TBT 数据 =====================
    for name, d in phase_dirs.items():
        dpath = Path(d)
        files = list(dpath.glob("*.csv"))
        if not files:
            print(f"Warning: no csv files under {dpath}")
            continue

        data = np.vstack([np.loadtxt(f, delimiter=",") for f in files])

        # 拆列
        x_p = data[:, COL_X]
        px_p = data[:, COL_PX]
        y_p = data[:, COL_Y]
        py_p = data[:, COL_PY]
        fi_p = data[:, COL_FI]
        Ek_p = data[:, COL_EK]
        phase_p = data[:, COL_RF_PHASE]
        t = data[:, COL_T] * 1000.0  # ms，仅调试用

        mean_r.append(np.mean(x_p))

        # RF 相位折叠到 (-π, π]
        mask_phase = phase_p > np.pi
        phase_p[mask_phase] -= 2.0 * np.pi

        # ---- SEO 闭轨插值 ----
        coords_fi_mod = np.mod(fi_p, 2.0 * np.pi)

        Malloc_r_SEO = np.zeros_like(x_p)
        Malloc_pr_SEO = np.zeros_like(x_p)
        Malloc_interp_flag1 = np.zeros_like(x_p)
        Malloc_interp_flag2 = np.zeros_like(x_p)

        Bilinear_interp_2D_vect_uniform(
            SEO_Ek_axis[0], SEO_Ek_axis[1] - SEO_Ek_axis[0], len(SEO_Ek_axis),
            SEO_fi_axis[0], SEO_fi_axis[1] - SEO_fi_axis[0], len(SEO_fi_axis),
            SEO_r_matrix,
            Ek_p,
            coords_fi_mod,
            Malloc_r_SEO, Malloc_interp_flag1
        )

        Bilinear_interp_2D_vect_uniform(
            SEO_Ek_axis[0], SEO_Ek_axis[1] - SEO_Ek_axis[0], len(SEO_Ek_axis),
            SEO_fi_axis[0], SEO_fi_axis[1] - SEO_fi_axis[0], len(SEO_fi_axis),
            SEO_pr_matrix,
            Ek_p,
            coords_fi_mod,
            Malloc_pr_SEO, Malloc_interp_flag2
        )

        # ---- Twiss 函数插值 ----
        Malloc_AlphaFuncR = np.zeros_like(x_p)
        Malloc_AlphaFuncZ = np.zeros_like(x_p)
        Malloc_BetaFuncR = np.zeros_like(x_p)
        Malloc_BetaFuncZ = np.zeros_like(x_p)
        Malloc_interp_flag3 = np.zeros_like(x_p)
        Malloc_interp_flag4 = np.zeros_like(x_p)
        Malloc_interp_flag5 = np.zeros_like(x_p)
        Malloc_interp_flag6 = np.zeros_like(x_p)

        coords_fi_mod_cell = np.mod(np.rad2deg(coords_fi_mod), twiss_cell_fi)

        Bilinear_interp_2D_vect_uniform(
            twiss_Ek_axis[0], twiss_Ek_axis[1] - twiss_Ek_axis[0], len(twiss_Ek_axis),
            twiss_fi_axis[0], twiss_fi_axis[1] - twiss_fi_axis[0], len(twiss_fi_axis),
            twiss_alphaR,
            Ek_p,
            coords_fi_mod_cell,
            Malloc_AlphaFuncR, Malloc_interp_flag3
        )

        Bilinear_interp_2D_vect_uniform(
            twiss_Ek_axis[0], twiss_Ek_axis[1] - twiss_Ek_axis[0], len(twiss_Ek_axis),
            twiss_fi_axis[0], twiss_fi_axis[1] - twiss_fi_axis[0], len(twiss_fi_axis),
            twiss_alphaZ,
            Ek_p,
            coords_fi_mod_cell,
            Malloc_AlphaFuncZ, Malloc_interp_flag4
        )

        Bilinear_interp_2D_vect_uniform(
            twiss_Ek_axis[0], twiss_Ek_axis[1] - twiss_Ek_axis[0], len(twiss_Ek_axis),
            twiss_fi_axis[0], twiss_fi_axis[1] - twiss_fi_axis[0], len(twiss_fi_axis),
            twiss_betaR,
            Ek_p,
            coords_fi_mod_cell,
            Malloc_BetaFuncR, Malloc_interp_flag5
        )

        Bilinear_interp_2D_vect_uniform(
            twiss_Ek_axis[0], twiss_Ek_axis[1] - twiss_Ek_axis[0], len(twiss_Ek_axis),
            twiss_fi_axis[0], twiss_fi_axis[1] - twiss_fi_axis[0], len(twiss_fi_axis),
            twiss_betaZ,
            Ek_p,
            coords_fi_mod_cell,
            Malloc_BetaFuncZ, Malloc_interp_flag6
        )

        print(f"{name}: max beta_r = {np.max(Malloc_BetaFuncR):.3e}, "
              f"min beta_r = {np.min(Malloc_BetaFuncR):.3e}, "
              f"mean beta_r = {np.mean(Malloc_BetaFuncR):.3e}")
        print(f"{name}: max Ek = {np.max(Ek_p):.3f}, "
              f"min Ek = {np.min(Ek_p):.3f}, "
              f"mean Ek = {np.mean(Ek_p):.3f}")

        mean_Ek_r.append(np.mean(Malloc_r_SEO))

        # ---- 换成相对闭轨坐标，并去掉束心偏移 ----
        x = x_p - Malloc_r_SEO
        px = px_p - Malloc_pr_SEO
        y = y_p
        py = py_p

        x = x - np.mean(x)
        px = px - np.mean(px)

        # ---- 单粒子 action + 90% 分位 ----
        beta_x = Malloc_BetaFuncR
        beta_y = Malloc_BetaFuncZ
        alpha_x = Malloc_AlphaFuncR
        alpha_y = Malloc_AlphaFuncZ
        gamma_x = (1.0 + alpha_x ** 2) / beta_x
        gamma_y = (1.0 + alpha_y ** 2) / beta_y

        eps_x = beta_x * px ** 2 + 2.0 * alpha_x * px * x + gamma_x * x ** 2
        eps_y = beta_y * py ** 2 + 2.0 * alpha_y * py * y + gamma_y * y ** 2

        # 90% 分位的几何发射度（m·rad）
        epsx_90 = np.percentile(eps_x, 99.0)
        epsy_90 = np.percentile(eps_y, 99.0)

        print(f"{name}: N={data.shape[0]}  εx_90%={epsx_90:.6e}  εy_90%={epsy_90:.6e}")

        pos_names.append(name)
        # m·rad → mm·mrad（几何 90%，不除 π）
        epsx_90_list.append(epsx_90 * 1e6)
        epsy_90_list.append(epsy_90 * 1e6)

        # ---- RMS 几何发射度（m·rad） ----
        epsx_rms = rms_emittance(x, px)
        epsy_rms = rms_emittance(y, py)

        # 转换为 mm·mrad，并除以 π
        epsx_rms_mm_mrad = epsx_rms * 1e6 / np.pi
        epsy_rms_mm_mrad = epsy_rms * 1e6 / np.pi

        print(f"{name}: eps_x_rms={epsx_rms_mm_mrad:.6e} mm·mrad   "
              f"eps_y_rms={epsy_rms_mm_mrad:.6e} mm·mrad")

        epsx_rms_list.append(epsx_rms_mm_mrad)
        epsy_rms_list.append(epsy_rms_mm_mrad)

        # ---- 归一化发射度 ----
        Ek_mean = np.mean(Ek_p)
        gamma_mean = (Ek_mean + M0C2_MEV) / M0C2_MEV
        beta_mean = np.sqrt(1.0 - 1.0 / gamma_mean**2)
        beta_gamma_mean = beta_mean * gamma_mean

        # RMS 归一化（m·rad）
        epsx_rms_norm = epsx_rms * beta_gamma_mean
        epsy_rms_norm = epsy_rms * beta_gamma_mean
        epsx_rms_norm_mm_mrad = epsx_rms_norm * 1e6 / np.pi
        epsy_rms_norm_mm_mrad = epsy_rms_norm * 1e6 / np.pi

        print(f"{name}: eps_x_n_rms={epsx_rms_norm_mm_mrad:.6e} mm·mrad   "
              f"eps_y_n_rms={epsy_rms_norm_mm_mrad:.6e} mm·mrad")

        epsx_rms_norm_list.append(epsx_rms_norm_mm_mrad)
        epsy_rms_norm_list.append(epsy_rms_norm_mm_mrad)

        # 90% 归一化
        epsx_90_norm = epsx_90 * beta_gamma_mean
        epsy_90_norm = epsy_90 * beta_gamma_mean
        epsx_90_norm_list.append(epsx_90_norm * 1e6)
        epsy_90_norm_list.append(epsy_90_norm * 1e6)

        # ===================== 画 5 张图 =====================

        # 1) x-px
        # fig_xpx, ax = plt.subplots(figsize=(5, 4))
        # ax.hist2d(x * 1000.0, px * 1000.0,
        #           bins=[256, 256], norm=LogNorm())
        fig_xpx, ax = hist2d_with_projections(
            x * 1000.0, px * 1000.0,
            # xlabel="x (mm)",
            # ylabel="px (mrad)",
            # title=f"{name} - x-px",
            xlim=(-80, 80),
            ylim=(-25, 25),
            # xlim=(-50, 50),
            # ylim=(-10, 10),
            # xlim=(-10, 10),
            # ylim=(-4, 4),
            bins=(256, 256),
            norm=LogNorm(),
            figsize=(5, 4),
            proj_frac=0.16,
            proj_color_x='g',  # 底部投影颜色
            proj_color_y='g'
        )
        # ellipse_x, ellipse_xp = generate_ellipse_points(
        #     epsx_rms, alpha_x[0], beta_x[0], 4.0, num_points=1000
        # )
        ellipse_x, ellipse_xp = generate_ellipse_points_from_emittance(epsx_90, alpha_x[0], beta_x[0])
        ax.plot(ellipse_x * 1000.0, ellipse_xp * 1000.0, linewidth=1.5, color="r", label=f'99% = {epsx_90 * 1e6 : .2f} pi.mm.mrad')
        ellipse_x_2, ellipse_xp_2 = generate_ellipse_points_from_emittance(epsx_rms*4, alpha_x[0], beta_x[0])
        ax.plot(ellipse_x_2 * 1000.0, ellipse_xp_2 * 1000.0, linewidth=1.5, color="r", linestyle='-.',
                label=f'4RMS = {epsx_rms*4 * 1e6 : .2f} pi.mm.mrad')

        ax.text(
            0.04, 0.94,
            rf"{FigText_xpx}",
            transform=ax.transAxes, fontsize=30,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white",edgecolor="none",alpha=0.75,pad=2.0),
            zorder=20,)

        ax.set_xlabel("x (mm)", fontsize=18)
        ax.set_ylabel("px (mrad)", fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        # ax.set_xlim([-80, 80])
        # ax.set_ylim([-25, 25])
        # ax.set_title(f"{name} - x-px")
        fig_xpx.tight_layout()
        ax.legend(loc="upper right")
        png_xpx = FIG_DIR / f"{name}_xpx.png"
        fig_xpx.savefig(png_xpx, dpi=300, bbox_inches='tight')
        plt.close(fig_xpx)

        # 2) y-py
        # fig_ypy, ax = plt.subplots(figsize=(5, 4))
        # ax.hist2d(y * 1000.0, py * 1000.0,
        #           bins=[256, 256], norm=LogNorm())
        fig_ypy, ax = hist2d_with_projections(
            y * 1000.0, py * 1000.0,
            # xlabel="y (mm)",
            # ylabel="py (mrad)",
            # title=f"{name} - y-py",
            xlim=(-80, 80),
            ylim=(-25, 25),
            # xlim=(-50, 50),
            # ylim=(-10, 10),
            # xlim=(-10, 10),
            # ylim=(-4, 4),
            bins=(256, 256),
            norm=LogNorm(),
            figsize=(5, 4),
            proj_frac=0.16,
            proj_color_x='g',  # 底部投影颜色
            proj_color_y='g'
        )
        # ellipse_y, ellipse_yp = generate_ellipse_points(
        #     epsy_rms, alpha_y[0], beta_y[0], 4.0, num_points=1000
        # )
        ellipse_y, ellipse_yp = generate_ellipse_points_from_emittance(epsy_90, alpha_y[0], beta_y[0])
        ax.plot(ellipse_y * 1000.0, ellipse_yp * 1000.0, linewidth=1.5, color="r", label=f'99% = {epsy_90 * 1e6 : .2f} pi.mm.mrad')
        ellipse_y_2, ellipse_yp_2 = generate_ellipse_points_from_emittance(epsy_rms * 4, alpha_y[0], beta_y[0])
        ax.plot(ellipse_y_2 * 1000.0, ellipse_yp_2 * 1000.0, linewidth=1.5, color="r", linestyle='-.',
                label=f'4RMS = {epsy_rms * 4 * 1e6 : .2f} pi.mm.mrad')

        ax.text(
            0.04, 0.94,
            rf"{FigText_ypy}",
            transform=ax.transAxes, fontsize=30,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
            zorder=20, )

        ax.set_xlabel("z (mm)", fontsize=20)
        ax.set_ylabel("pz (mrad)", fontsize=20)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        # ax.set_xlim([-80, 80])
        # ax.set_ylim([-25, 25])
        # ax.set_title(f"{name} - y-py")
        fig_ypy.tight_layout()
        ax.legend(loc="upper right")
        png_ypy = FIG_DIR / f"{name}_ypy.png"
        fig_ypy.savefig(png_ypy, dpi=300, bbox_inches='tight')
        plt.close(fig_ypy)

        # 3) phase-Ek (局部)
        # fig_phaseEk, ax = plt.subplots(figsize=(5, 4))
        # ax.hist2d(
        #     np.rad2deg(phase_p), Ek_p,
        #     bins=[256, 256], norm=LogNorm()
        # )
        plot_phase_center = np.mean(np.rad2deg(phase_p))-90
        plot_Ek_center = np.mean(Ek_p)
        rf_width = np.rad2deg(phase_p) - plot_phase_center
        Ek_width = Ek_p - plot_Ek_center
        # 计算束团的rf束长和能散，99.9%百分位
        # --- RF 束长 & 能散：RMS(σ) 与 99%全宽 ---
        rf_sigma = rms_width_centered(rf_width)  # deg
        Ek_sigma = rms_width_centered(Ek_width)  # MeV

        rf_w99 = robust_width_centered(rf_width, pct=99.0)  # deg, 全宽
        Ek_w99 = robust_width_centered(Ek_width, pct=99.0)  # MeV, 全宽

        rf_width_rms_list.append(rf_sigma)
        Ek_width_rms_list.append(Ek_sigma)

        rf_width_90_list.append(rf_w99)  # 你原变量名叫 90_list，这里就沿用
        Ek_width_90_list.append(Ek_w99)

        fig_phaseEk, ax = hist2d_with_projections(
            np.rad2deg(phase_p)-90, Ek_p,
            # xlabel="x (mm)",
            # ylabel="px (mrad)",
            # title=f"{name} - x-px",
            xlim=(plot_phase_center-180, plot_phase_center+180),
            ylim=(plot_Ek_center-10, plot_Ek_center+10),
            bins=(256, 256),
            norm=LogNorm(),
            figsize=(5, 4),
            proj_frac=0.16,
            proj_color_x='g',  # 底部投影颜色
            proj_color_y='g'
        )
        ax.set_xlabel("RF phase (deg)", fontsize=18)
        ax.set_ylabel("Ek (MeV)", fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        # ax.set_title(f"{name} - phase-Ek")
        fig_phaseEk.tight_layout()
        png_phaseEk = FIG_DIR / f"{name}_phaseEk.png"
        fig_phaseEk.savefig(png_phaseEk, dpi=300, bbox_inches='tight')
        plt.close(fig_phaseEk)

        # 4) x-y
        # fig_xy, ax = plt.subplots(figsize=(5, 4))
        # ax.hist2d(
        #     x * 1000.0, y * 1000.0,
        #     bins=[256, 256], norm=LogNorm()
        # )
        fig_xy, ax = hist2d_with_projections(
            x * 1000.0, y * 1000.0,
            # xlabel="x (mm)",
            # ylabel="px (mrad)",
            # title=f"{name} - x-px",
            xlim=(-80, 80),
            ylim=(-80, 80),
            # xlim=(-50, 50),
            # ylim=(-50, 50),
            # xlim=(-10, 10),
            # ylim=(-10, 10),
            bins=(256, 256),
            norm=LogNorm(),
            figsize=(5, 4),
            proj_frac=0.16,
            proj_color_x='g',  # 底部投影颜色
            proj_color_y='g'
        )
        ax.text(
            0.04, 0.94,
            rf"{FigText_xy}",
            transform=ax.transAxes, fontsize=30,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
            zorder=20, )

        ax.set_xlabel("x (mm)", fontsize=20)
        ax.set_ylabel("z (mm)", fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        # ax.set_xlim([-80, 80])
        # ax.set_ylim([-80, 80])
        # ax.set_title(f"{name} - x-y")
        fig_xy.tight_layout()
        png_xy = FIG_DIR / f"{name}_xy.png"
        fig_xy.savefig(png_xy, dpi=300, bbox_inches='tight')
        plt.close(fig_xy)

        # 5) phase-Ek 宽范围、细长图
        fig_phaseEk_wide, ax = plt.subplots(figsize=(4, 6))
        ax.hist2d(
            np.rad2deg(phase_p)-70, Ek_p,
            bins=[256, 256], norm=LogNorm()
        )
        ax.set_xlabel("RF phase (deg)", fontsize=18)
        ax.set_ylabel("Ek (MeV)", fontsize=18)
        ax.set_xlim([-40.0, 80.0])
        ax.set_ylim([290.0, 620.0])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        # ax.set_title(f"{name} - phase-Ek wide")
        fig_phaseEk_wide.tight_layout()
        png_phaseEk_wide = FIG_DIR / f"{name}_phaseEk_wide.png"
        fig_phaseEk_wide.savefig(png_phaseEk_wide, dpi=300, bbox_inches='tight')
        plt.close(fig_phaseEk_wide)

        # 加到一页 PPT
        add_five_pngs_to_ppt(
            prs,
            png_xpx, png_ypy, png_phaseEk, png_xy, png_phaseEk_wide,
            title=f"{name} - phase space & longitudinal"
        )

    # ===================== 发射度 vs turn 曲线 =====================

    if not pos_names:
        ppt_path = "FFAG_emittance_phase_report.pptx"
        prs.save(ppt_path)
        print(f"无有效数据，仅生成空 PPT: {ppt_path}")
        return

    # Turn 数组
    pos_idx = np.array([int(name.split('_')[1]) for name in pos_names])
    plot_turn_max = max(500, int(np.ceil(np.max(pos_idx) / 500.0) * 500))

    # ---- 90% 几何 & 归一化 ----
    epsx_90_arr = np.array(epsx_90_list)
    epsy_90_arr = np.array(epsy_90_list)
    epsx_90_norm_arr = np.array(epsx_90_norm_list)
    epsy_90_norm_arr = np.array(epsy_90_norm_list)

    fig1, ax1 = plt.subplots(figsize=(7, 4))

    if epsx_90_arr.size > 0:
        ax1.plot(pos_idx, epsx_90_arr, marker='o', label=r'$\epsilon_{x,99\%}$')
    if epsy_90_arr.size > 0:
        ax1.plot(pos_idx, epsy_90_arr, marker='s', label=r'$\epsilon_{z,99\%}$')

    # if epsx_90_norm_arr.size > 0:
    #     ax1.plot(pos_idx, epsx_90_norm_arr, marker='^', label=r'$\epsilon_{x,90\%}^{(n)}$')
    # if epsy_90_norm_arr.size > 0:
    #     ax1.plot(pos_idx, epsy_90_norm_arr, marker='v', label=r'$\epsilon_{y,90\%}^{(n)}$')

    ax1.set_xlabel("turn", fontsize=24)
    ax1.set_ylabel(r"99% emit. [$\pi$ mm mrad]", fontsize=20)
    ax1.legend(fontsize=21)
    ax1.set_xlim([0, plot_turn_max])
    ax1.set_ylim([0, 600])
    ax1.tick_params(axis='both', labelsize=24)
    ax1.grid(alpha=0.3)

    # # 阶段标记：避开 legend 和曲线
    # y_stage = 300
    # ax1.axvline(1000, color="k", linestyle="--", linewidth=1.5, alpha=0.70)
    # ax1.text(450, 300, "Painting injection", fontsize=15, ha="center", va="center")
    # ax1.annotate("Acceleration", xy=(2300, y_stage), xytext=(1350, y_stage),
    #              fontsize=15, ha="left", va="center",
    #              arrowprops=dict(arrowstyle="->", linewidth=1.4, color="k", alpha=0.75))
    # # legend 放右上角，避免和 injection 标注重叠"upper right""lower right"
    # ax1.legend(fontsize=19, frameon=False, loc="lower right")
    # ax1.text(0.04, 0.88, str(Fig_emit_panel), transform=ax1.transAxes,
    #          fontsize=20, fontweight="bold", va="top", ha="left",
    #          bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
    #          zorder=20)

    # # 阶段标记：放在底部空白区，避免 legend 和曲线
    # y_stage = 30
    # ax1.axvline(1000, color="k", linestyle="--", linewidth=1.5, alpha=0.70)
    # ax1.text(450, 300, "Painting injection", fontsize=15,
    #          ha="center", va="center")
    # ax1.annotate("Acceleration", xy=(2300, y_stage), xytext=(1350, y_stage),
    #              fontsize=15, ha="left", va="center",
    #              arrowprops=dict(arrowstyle="->", linewidth=1.4, color="k", alpha=0.75))
    # ax1.legend(fontsize=19, frameon=False, loc="upper right")
    # ax1.text(0.04, 0.92, str(Fig_emit_panel), transform=ax1.transAxes,
    #          fontsize=20, fontweight="bold", va="top", ha="left",
    #          bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
    #          zorder=20)

    fig1.tight_layout()
    png_90 = FIG_DIR / "emittance_90_vs_turn.png"
    fig1.savefig(png_90, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ---- RMS 几何 & 归一化 ----
    epsx_rms_arr = np.array(epsx_rms_list)
    epsy_rms_arr = np.array(epsy_rms_list)
    epsx_rms_norm_arr = np.array(epsx_rms_norm_list)
    epsy_rms_norm_arr = np.array(epsy_rms_norm_list)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    if epsx_rms_arr.size > 0:
        ax2.plot(pos_idx, epsx_rms_arr, marker='o',
                 label=r'$\epsilon_{x,\mathrm{rms}}$ (geom)')
    if epsy_rms_arr.size > 0:
        ax2.plot(pos_idx, epsy_rms_arr, marker='s',
                 label=r'$\epsilon_{y,\mathrm{rms}}$ (geom)')
    # if epsx_rms_norm_arr.size > 0:
    #     ax2.plot(pos_idx, epsx_rms_norm_arr, marker='^',
    #              label=r'$\epsilon_{x,\mathrm{n,rms}}$')
    # if epsy_rms_norm_arr.size > 0:
    #     ax2.plot(pos_idx, epsy_rms_norm_arr, marker='v',
    #              label=r'$\epsilon_{y,\mathrm{n,rms}}$')

    ax2.set_xlabel('turn')
    ax2.set_ylabel('RMS emittance (mm·mrad, $\\pi$ removed)')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=21)
    ax2.set_xlim([0, plot_turn_max])
    ax2.set_ylim([0, 30])
    ax2.tick_params(axis='both', labelsize=18)
    fig2.tight_layout()

    png_rms = FIG_DIR / "emittance_rms_vs_turn.png"
    fig2.savefig(png_rms, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    add_emittance_page(
        prs,
        png_eps99=png_90,
        png_eps_rms=png_rms,
        title="Emittance evolution vs turn"
    )

    # ===================== RF 束长 & 能散 vs turn =====================

    rf_w99_arr = np.array(rf_width_90_list)  # deg (99% 全宽)
    rf_sig_arr = np.array(rf_width_rms_list)  # deg (RMS)

    Ek_w99_arr = np.array(Ek_width_90_list)  # MeV (99% 全宽)
    Ek_sig_arr = np.array(Ek_width_rms_list)  # MeV (RMS)

    # --- RF bunch length vs turn ---
    figL, axL = plt.subplots(figsize=(7, 4))
    # axL.plot(pos_idx, rf_sig_arr, marker='o', label='RF phase σ (deg)')
    axL.plot(pos_idx, rf_w99_arr, marker='s', label='RF phase 99% full width (deg)')
    axL.set_xlabel('turn')
    axL.set_ylabel('RF bunch length (deg)')
    axL.grid(alpha=0.3)
    axL.legend()
    figL.tight_layout()

    png_rf = FIG_DIR / "rf_bunch_length_vs_turn.png"
    figL.savefig(png_rf, dpi=300, bbox_inches='tight')
    plt.close(figL)

    # --- Energy spread vs turn ---
    figE, axE = plt.subplots(figsize=(7, 4))
    # axE.plot(pos_idx, Ek_sig_arr, marker='o', label='ΔEk σ (MeV)')
    axE.plot(pos_idx, Ek_w99_arr, marker='s', label='ΔEk 99% full width (MeV)')
    axE.set_xlabel('turn')
    axE.set_ylabel('Energy spread (MeV)')
    axE.grid(alpha=0.3)
    axE.legend()
    figE.tight_layout()

    png_dE = FIG_DIR / "energy_spread_vs_turn.png"
    figE.savefig(png_dE, dpi=300, bbox_inches='tight')
    plt.close(figE)

    print(f"[OK] Saved: {png_rf}")
    print(f"[OK] Saved: {png_dE}")

    add_emittance_page(
        prs,
        png_eps99=png_rf,
        png_eps_rms=png_dE,
        title="Longitudinal: RF bunch length & energy spread vs turn"
    )

    # fig3, ax3 = plt.subplots(figsize=(7, 4))
    # ax3.plot(pos_idx, (np.array(mean_r) - np.array(mean_Ek_r))*1000, marker='v',
    #          label='mean radius')
    # plt.ylabel('deltaX (mm)')
    # png_rms1 = FIG_DIR / "meanr.png"
    # fig3.savefig(png_rms1, dpi=300, bbox_inches='tight')
    # plt.close(fig3)
    #
    # data_meanr = np.column_stack([
    #     pos_idx,
    #     mean_r,
    #     mean_Ek_r,
    #     np.array(mean_r) - np.array(mean_Ek_r)
    # ])
    # txt_meanr = FIG_DIR / "meanr.dat"
    # np.savetxt(
    #     txt_meanr,
    #     data_meanr,
    #     fmt="%.8e",
    #     header=(
    #         "pos_idx            "
    #         "mean_r             "
    #         "mean_Ek_r          "
    #         "mean_r_minus_mean_Ek_r"
    #     ),
    #     comments=""
    # )
    # print(f"[OK] Figure saved to: {png_meanr}")
    # print(f"[OK] Data   saved to: {txt_meanr}")

    # ===================== 保存 PPT =====================
    # ppt_path = "FFAG_emittance_phase_report_cor_25_4.pptx"
    prs.save(ppt_path)
    print(f"PPT 已生成: {ppt_path}")
    print(f"所有 PNG 缓存在目录: {FIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate phase-space figures and an emittance PowerPoint report."
    )
    parser.add_argument(
        "--base-dir", type=Path, default=BASE_DIR,
        help="Directory containing TBT_Dump_*/merged_files."
    )
    parser.add_argument(
        "--seo-dir", type=Path, default=SEO_DIR,
        help="Directory containing SEO_r/SEO_pr and Twiss text files."
    )
    parser.add_argument(
        "--fig-dir", type=Path, default=FIG_DIR,
        help="Directory for generated PNG figures."
    )
    parser.add_argument(
        "--ppt", type=Path, default=ppt_path,
        help="Output PowerPoint path."
    )
    parser.add_argument(
        "--turn-min", type=int, default=None,
        help="Only process TBT dumps at or above this turn."
    )
    parser.add_argument(
        "--turn-max", type=int, default=None,
        help="Only process TBT dumps at or below this turn."
    )
    args = parser.parse_args()
    main(
        args.ppt,
        args.base_dir,
        args.seo_dir,
        args.fig_dir,
        args.turn_min,
        args.turn_max,
    )
