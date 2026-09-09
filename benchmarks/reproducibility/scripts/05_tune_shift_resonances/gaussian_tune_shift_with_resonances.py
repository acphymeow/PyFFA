import argparse
import re
import time
import os
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from numba import njit, prange
from scipy.interpolate import CubicSpline

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bundle_paths import REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from FFAG_MathTools import Bilinear_interp_2D_vect_uniform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one Gaussian PIC/theory resonance-line pair."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            REPO_ROOT.parent
            / "output"
            / "NewBenchmark"
            / "FFT_Tune_Gauss80_uniform72"
        ),
    )
    parser.add_argument("--emittance", type=float, default=80.0)
    parser.add_argument("--pic-label", default="e")
    parser.add_argument("--theory-label", default="f")
    parser.add_argument("--core-qx", type=float, default=3.077)
    parser.add_argument("--core-qz", type=float, default=2.640)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


ARGS = parse_args()


# ============================================================
# 参数
# ============================================================
DATA_DIR = ARGS.data_dir
SEO_BASE = REPO_ROOT / "Bmap_FD16" / "resultsSEO_noError_X"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEO_r_filepath = f"{SEO_BASE}/SEO_r.txt"
SEO_pr_filepath = f"{SEO_BASE}/SEO_pr.txt"
betaR_filepath = f"{SEO_BASE}/BetaFuncR.txt"
betaZ_filepath = f"{SEO_BASE}/BetaFuncZ.txt"

COL_R, COL_PR, COL_Z, COL_PZ, COL_FI, COL_EK = 0, 1, 2, 3, 4, 5
SUBTRACT_CLOSED_ORBIT = True

MAX_FILES = 20000
N_THEORY_PARTICLES = 2000
N_SKIP, MIN_TURNS, RNG_SEED = 10, 128, 11111

PAD_FACTOR = 8

# 这里先给初值，后面会用强平滑 beta 积分结果覆盖 bare tune
QR_BARE_ABS, QZ_BARE_ABS = 3.230202618580, 2.813029944565

# 这个是max tune shift参考线
# QR_CORE_ABS, QZ_CORE_ABS = 2.925, 2.482
# QR_CORE_ABS, QZ_CORE_ABS = 3.015, 2.574
QR_CORE_ABS, QZ_CORE_ABS = ARGS.core_qx, ARGS.core_qz

QX_DISPLAY_SHIFT = 0.00

COMPLEX_SIGN_R, COMPLEX_SIGN_Z = -1.0, -1.0

N_CELLS = 16
CELL_START_DEG, CELL_END_DEG = 0.0, 360.0 / N_CELLS

N_THETA_CELL = 300
N_TURNS_THEORY = 256
N_STEPS_PER_CELL = 50
N_QUAD = 12

N_PROTONS = 2.08e13
BUNCH_LENGTH_M = 13
KINETIC_ENERGY_MEV = 300.0
EMITTANCE_IS_NORMALIZED = False
EMITTANCE_40PI_MM_MRAD = ARGS.emittance
# 全局变量
FigText1 = ARGS.pic_label
FigText2 = ARGS.theory_label
EPS0 = 8.8541878128e-12
QE = 1.602176634e-19
MP_MEV = 938.2720813
MP_KG = 1.67262192369e-27
C_LIGHT = 299792458.0

QR_XLIM, QZ_XLIM = (2.80, 3.30), (2.40, 2.90)

# 三阶及以下共振线：与 KV、Application 图保持一致。
RESONANCE_MAX_ORDER = 3
RESONANCE_LINE_COLOR = "#9f9f9f"
RESONANCE_TEXT_COLOR = "#858585"
RESONANCE_LINEWIDTH = 0.58
RESONANCE_LINE_ALPHA = 0.95
RESONANCE_LABEL_FONTSIZE = 7.8
RESONANCE_TEXT_ALPHA = 0.82
RESONANCE_LABEL_BOX_ALPHA = 0.36

# 强平滑参数
CUTOFF_FRAC = 0.08
FILTER_ORDER = 10


# ============================================================
# 基础工具函数
# ============================================================
def get_pid(path):
    m = re.search(r"particle_(\d+)_", path.name)
    return int(m.group(1)) if m else -1


def print_stats(name, y):
    y = np.asarray(y)
    rms = np.sqrt(np.mean(y**2))
    print(
        f"{name:50s} min/mean/max/rms = "
        f"{y.min(): .8e}, {y.mean(): .8e}, {y.max(): .8e}, {rms: .8e}"
    )


def canonical_resonance(m, n, p):
    """把同一条几何共振线化为唯一整数表示。"""
    divisor = math.gcd(math.gcd(abs(m), abs(n)), abs(p))
    if divisor > 1:
        m, n, p = m // divisor, n // divisor, p // divisor
    if m < 0 or (m == 0 and n < 0):
        m, n, p = -m, -n, -p
    return m, n, p


def resonance_label(m, n, p):
    terms = []
    for coefficient, symbol in ((m, r"Q_x"), (n, r"Q_z")):
        if coefficient == 0:
            continue
        if not terms:
            sign = "-" if coefficient < 0 else ""
        else:
            sign = "-" if coefficient < 0 else "+"
        magnitude = "" if abs(coefficient) == 1 else str(abs(coefficient))
        terms.append(f"{sign}{magnitude}{symbol}")
    return "$" + "".join(terms) + f"={p}$"


def resonance_segment(m, n, p, qx_lim, qz_lim):
    xmin, xmax = qx_lim
    ymin, ymax = qz_lim
    candidates = []

    if n != 0:
        for x in (xmin, xmax):
            y = (p - m * x) / n
            if ymin - 1e-10 <= y <= ymax + 1e-10:
                candidates.append((x, y))
    if m != 0:
        for y in (ymin, ymax):
            x = (p - n * y) / m
            if xmin - 1e-10 <= x <= xmax + 1e-10:
                candidates.append((x, y))

    distinct = []
    for point in candidates:
        if not any(np.hypot(point[0] - q[0], point[1] - q[1]) < 1e-9 for q in distinct):
            distinct.append(point)
    if len(distinct) < 2:
        return None

    best_pair = None
    best_distance = -1.0
    for i, point_a in enumerate(distinct):
        for point_b in distinct[i + 1:]:
            distance = (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2
            if distance > best_distance:
                best_distance = distance
                best_pair = (point_a, point_b)
    return best_pair


def resonances_in_window(qx_lim, qz_lim, max_order=3):
    """返回给定 tune 窗口内所有唯一的 mQx+nQz=p 共振线。"""
    xmin, xmax = qx_lim
    ymin, ymax = qz_lim
    corners = ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))
    unique = {}

    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            if abs(m) + abs(n) == 0 or abs(m) + abs(n) > max_order:
                continue
            values = [m * x + n * y for x, y in corners]
            p_min = math.ceil(min(values) - 1e-12)
            p_max = math.floor(max(values) + 1e-12)
            for p in range(p_min, p_max + 1):
                cm, cn, cp = canonical_resonance(m, n, p)
                order = abs(cm) + abs(cn)
                if order <= max_order:
                    unique[(cm, cn, cp)] = order

    return [
        (m, n, p, order)
        for (m, n, p), order in sorted(
            unique.items(), key=lambda item: (item[1], *item[0])
        )
    ]


def point_in_box(x, y, box):
    xmin, xmax, ymin, ymax = box
    return xmin <= x <= xmax and ymin <= y <= ymax


def choose_resonance_label_position(
        segment, data_center, avoid_boxes, occupied, qx_lim, qz_lim):
    """优先把文字放到线段边缘、远离 footprint 主体和已有标签。"""
    (x0, y0), (x1, y1) = segment
    xspan = qx_lim[1] - qx_lim[0]
    yspan = qz_lim[1] - qz_lim[0]
    candidate_fractions = (0.15, 0.85, 0.24, 0.76, 0.34, 0.66)
    best = None
    best_score = -np.inf

    for fraction in candidate_fractions:
        x = x0 + fraction * (x1 - x0)
        y = y0 + fraction * (y1 - y0)
        dx = (x - data_center[0]) / xspan
        dy = (y - data_center[1]) / yspan
        score = np.hypot(dx, dy)

        if any(point_in_box(x, y, box) for box in avoid_boxes):
            score -= 8.0
        for old_x, old_y in occupied:
            distance = np.hypot((x - old_x) / xspan, (y - old_y) / yspan)
            if distance < 0.13:
                score -= 5.0 * (0.13 - distance) / 0.13

        # 文字保留少量边界余量，降低保存时裁切风险。
        edge_distance = min(
            (x - qx_lim[0]) / xspan,
            (qx_lim[1] - x) / xspan,
            (y - qz_lim[0]) / yspan,
            (qz_lim[1] - y) / yspan,
        )
        if edge_distance < 0.065:
            score -= 2.5 * (0.065 - edge_distance) / 0.065

        if score > best_score:
            best_score = score
            best = (x, y)
    return best


def draw_resonances_up_to_order3(ax, qx_data, qz_data, qx_lim, qz_lim):
    """画共振线，并把普通字重标签自动放到 footprint 主体之外。"""
    qx_data = np.asarray(qx_data, dtype=float)
    qz_data = np.asarray(qz_data, dtype=float)
    good = (
        np.isfinite(qx_data) & np.isfinite(qz_data)
        & (qx_data >= qx_lim[0]) & (qx_data <= qx_lim[1])
        & (qz_data >= qz_lim[0]) & (qz_data <= qz_lim[1])
    )
    qx_good = qx_data[good]
    qz_good = qz_data[good]
    if qx_good.size:
        data_center = (float(np.median(qx_good)), float(np.median(qz_good)))
        qx_low, qx_high = np.percentile(qx_good, [3.0, 97.0])
        qz_low, qz_high = np.percentile(qz_good, [3.0, 97.0])
    else:
        data_center = (
            0.5 * (qx_lim[0] + qx_lim[1]),
            0.5 * (qz_lim[0] + qz_lim[1]),
        )
        qx_low = qx_high = data_center[0]
        qz_low = qz_high = data_center[1]

    xspan = qx_lim[1] - qx_lim[0]
    yspan = qz_lim[1] - qz_lim[0]
    ax.set_xlim(qx_lim)
    ax.set_ylim(qz_lim)
    ax.figure.canvas.draw()

    avoid_boxes = [
        # tune footprint 的 3%-97% 主体区域并加少量边界。
        (
            qx_low - 0.025 * xspan, qx_high + 0.025 * xspan,
            qz_low - 0.025 * yspan, qz_high + 0.025 * yspan,
        ),
        # 左侧 (e)/(f) 子图编号区域。
        (
            qx_lim[0], qx_lim[0] + 0.15 * xspan,
            qz_lim[0] + 0.60 * yspan, qz_lim[0] + 0.76 * yspan,
        ),
    ]

    # 使用实际渲染后的 legend 外框，而不是估计其占用范围。
    legend = ax.get_legend()
    if legend is not None:
        renderer = ax.figure.canvas.get_renderer()
        legend_bbox_display = legend.get_window_extent(renderer=renderer)
        legend_bbox_data = ax.transData.inverted().transform(
            np.asarray([
                [legend_bbox_display.x0, legend_bbox_display.y0],
                [legend_bbox_display.x1, legend_bbox_display.y1],
            ])
        )
        avoid_boxes.append((
            legend_bbox_data[0, 0] - 0.015 * xspan,
            legend_bbox_data[1, 0] + 0.015 * xspan,
            legend_bbox_data[0, 1] - 0.015 * yspan,
            legend_bbox_data[1, 1] + 0.015 * yspan,
        ))
    occupied = []

    for m, n, p, _order in resonances_in_window(
            qx_lim, qz_lim, RESONANCE_MAX_ORDER):
        segment = resonance_segment(m, n, p, qx_lim, qz_lim)
        if segment is None:
            continue
        (x0, y0), (x1, y1) = segment
        ax.plot(
            [x0, x1], [y0, y1],
            color=RESONANCE_LINE_COLOR,
            linewidth=RESONANCE_LINEWIDTH,
            alpha=RESONANCE_LINE_ALPHA,
            linestyle="-",
            zorder=3,
        )

        position = choose_resonance_label_position(
            segment, data_center, avoid_boxes, occupied, qx_lim, qz_lim
        )
        occupied.append(position)
        display_points = ax.transData.transform(np.asarray([[x0, y0], [x1, y1]]))
        delta = display_points[1] - display_points[0]
        angle = np.degrees(np.arctan2(delta[1], delta[0]))
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0

        ax.text(
            position[0], position[1], resonance_label(m, n, p),
            color=RESONANCE_TEXT_COLOR,
            alpha=RESONANCE_TEXT_ALPHA,
            fontsize=RESONANCE_LABEL_FONTSIZE,
            fontweight="normal",
            rotation=angle,
            rotation_mode="anchor",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.08",
                facecolor="white",
                edgecolor="none",
                alpha=RESONANCE_LABEL_BOX_ALPHA,
            ),
            clip_on=True,
            zorder=10,
        )


def parabolic_interpolate(amp, k):
    y0, y1, y2 = amp[k - 1], amp[k], amp[k + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0:
        return float(k)
    return k + 0.5 * (y0 - y2) / denom


def cfft_frac_interp(x, p, sign=-1.0, pad_factor=8):
    x = np.asarray(x, float)
    p = np.asarray(p, float)

    if len(x) < MIN_TURNS or np.std(x) == 0 or np.std(p) == 0:
        return np.nan

    x = x - np.mean(x)
    p = p - np.mean(p)

    scale = np.std(x) / np.std(p)
    u = x + 1j * sign * scale * p

    n = len(u)
    nfft = int(pad_factor * n)

    spec = np.abs(np.fft.fft(u, n=nfft))
    freq = np.fft.fftfreq(nfft, d=1.0)

    spec[0] = 0.0

    k = int(np.argmax(spec))
    k_use = min(max(k, 1), nfft - 2)

    k_peak = parabolic_interpolate(spec, k_use)
    df = freq[1] - freq[0]

    q_frac = np.mod(k_peak * df, 1.0)

    return q_frac


def abs_tune(qf, qb, qc):
    n0 = int(np.floor(qb))
    qmin = min(qb, qc) - 0.10
    qmax = max(qb, qc) + 0.10

    cand = np.array([
        n0 - 2 + qf,
        n0 - 1 + qf,
        n0 + qf,
        n0 + 1 + qf,
        n0 + 2 + qf,
    ])

    inside = cand[(cand >= qmin) & (cand <= qmax)]

    if len(inside) > 0:
        return inside[np.argmin(np.abs(inside - 0.5 * (qb + qc)))]

    return cand[np.argmin(np.abs(cand - qb))]


def fft_range(x, xp, beta0, alpha0, q_low, q_high):
    X = x / np.sqrt(beta0)
    P = (alpha0 * x + beta0 * xp) / np.sqrt(beta0)
    u = X - 1j * P

    spec = np.abs(np.fft.fft((u - np.mean(u)) * np.hanning(len(u))))
    freq = np.fft.fftfreq(len(u))

    spec[np.argmin(np.abs(freq))] = 0.0

    best_q = np.nan
    best_a = -1.0

    for k, f in enumerate(freq):
        qf = np.mod(f, 1.0)

        for n in range(int(np.floor(q_low)) - 2, int(np.ceil(q_high)) + 3):
            q = n + qf

            if q_low <= q <= q_high and spec[k] > best_a:
                best_q = q
                best_a = spec[k]

    return best_q


# ============================================================
# 强平滑 lattice 工具函数
# ============================================================
def periodic_cubic_eval_full_rad(phi_axis, y_axis, phi_eval):
    phi = np.asarray(phi_axis, dtype=float).copy()
    y = np.asarray(y_axis, dtype=float).copy()

    phi = np.mod(phi, 2.0 * np.pi)

    idx = np.argsort(phi)
    phi = phi[idx]
    y = y[idx]

    phi_unique, uidx = np.unique(phi, return_index=True)
    phi = phi_unique
    y = y[uidx]

    if abs(phi[0]) > 1.0e-14:
        phi = np.r_[0.0, phi]
        y = np.r_[y[-1], y]

    phi_ext = np.r_[phi, 2.0 * np.pi]
    y_ext = np.r_[y, y[0]]

    cs = CubicSpline(phi_ext, y_ext, bc_type="periodic")

    phi_mod = np.mod(phi_eval, 2.0 * np.pi)

    y_eval = cs(phi_mod)
    dy_dphi = cs.derivative()(phi_mod)

    return y_eval, dy_dphi


def periodic_cubic_eval_cell_deg(
    theta_axis_deg,
    y_axis,
    theta_eval_deg,
    cell_start_deg,
    cell_end_deg,
):
    x = np.asarray(theta_axis_deg, dtype=float).copy()
    y = np.asarray(y_axis, dtype=float).copy()

    period_deg = cell_end_deg - cell_start_deg

    x = np.mod(x - cell_start_deg, period_deg)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    x_unique, uidx = np.unique(x, return_index=True)
    x = x_unique
    y = y[uidx]

    if abs(x[0]) > 1.0e-12:
        x = np.r_[0.0, x]
        y = np.r_[y[-1], y]

    x_ext = np.r_[x, period_deg]
    y_ext = np.r_[y, y[0]]

    cs = CubicSpline(x_ext, y_ext, bc_type="periodic")

    theta_mod = np.mod(theta_eval_deg - cell_start_deg, period_deg)

    return cs(theta_mod)


def build_periodic_s_grid(ds_dtheta, dtheta):
    h = np.asarray(ds_dtheta, dtype=float)
    h_next = np.roll(h, -1)
    ds_seg = 0.5 * (h + h_next) * dtheta

    s_nodes = np.zeros(len(h), dtype=float)
    s_nodes[1:] = np.cumsum(ds_seg[:-1])

    cell_length = np.sum(ds_seg)

    return s_nodes, cell_length


def periodic_interp_to_uniform_s(s_nodes, y_nodes, cell_length, n_out):
    s_nodes = np.asarray(s_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)

    idx = np.argsort(s_nodes)
    s_sorted = s_nodes[idx]
    y_sorted = y_nodes[idx]

    s_ext = np.r_[s_sorted, cell_length]
    y_ext = np.r_[y_sorted, y_sorted[0]]

    cs = CubicSpline(s_ext, y_ext, bc_type="periodic")

    s_uniform = np.linspace(0.0, cell_length, n_out, endpoint=False)
    y_uniform = cs(s_uniform)

    return s_uniform, y_uniform


def periodic_interp_uniform_s_to_theta(s_uniform, y_uniform, s_nodes, cell_length):
    s_ext = np.r_[s_uniform, cell_length]
    y_ext = np.r_[y_uniform, y_uniform[0]]

    cs = CubicSpline(s_ext, y_ext, bc_type="periodic")

    return cs(s_nodes)


def fourier_filter_derivatives_uniform_s(
    y,
    cell_length,
    cutoff_frac=0.08,
    order=10,
):
    y = np.asarray(y, dtype=float)
    n = len(y)

    ds = cell_length / n

    Y = np.fft.fft(y)

    k = 2.0 * np.pi * np.fft.fftfreq(n, d=ds)
    k_abs = np.abs(k)

    k_nyquist = np.pi / ds
    k_cut = cutoff_frac * k_nyquist

    if k_cut <= 0.0:
        raise ValueError("cutoff_frac is too small.")

    filt = np.exp(- (k_abs / k_cut) ** order)

    Yf = Y * filt

    y0 = np.real(np.fft.ifft(Yf))
    y1 = np.real(np.fft.ifft(1j * k * Yf))
    y2 = np.real(np.fft.ifft((1j * k) ** 2 * Yf))

    return y0, y1, y2


# ============================================================
# 读取 SEO / Twiss
# ============================================================
seo_r = np.loadtxt(SEO_r_filepath, skiprows=1)
seo_pr = np.loadtxt(SEO_pr_filepath, skiprows=1)

fi_axis, Ek_axis = seo_r[0, 1:], seo_r[1:, 0]
r_mat, pr_mat = seo_r[1:, 1:], seo_pr[1:, 1:]

betaR_data = np.loadtxt(betaR_filepath, skiprows=1)
betaZ_data = np.loadtxt(betaZ_filepath, skiprows=1)

theta_axis_deg = betaR_data[0, 1:]
twiss_Ek_axis = betaR_data[1:, 0]

betaR_mat = betaR_data[1:, 1:]
betaZ_mat = betaZ_data[1:, 1:]


def sub_co(r, pr, fi, Ek):
    fi_mod = np.mod(fi, 2 * np.pi)

    r0 = np.zeros_like(r)
    pr0 = np.zeros_like(pr)
    f1 = np.zeros_like(r)
    f2 = np.zeros_like(pr)

    Bilinear_interp_2D_vect_uniform(
        Ek_axis[0], Ek_axis[1] - Ek_axis[0], len(Ek_axis),
        fi_axis[0], fi_axis[1] - fi_axis[0], len(fi_axis),
        r_mat, Ek, fi_mod, r0, f1,
    )

    Bilinear_interp_2D_vect_uniform(
        Ek_axis[0], Ek_axis[1] - Ek_axis[0], len(Ek_axis),
        fi_axis[0], fi_axis[1] - fi_axis[0], len(fi_axis),
        pr_mat, Ek, fi_mod, pr0, f2,
    )

    return r - r0, pr - pr0


# ============================================================
# 强平滑自洽理论 lattice
# ============================================================
theta_deg = np.linspace(
    CELL_START_DEG,
    CELL_END_DEG,
    N_THETA_CELL,
    endpoint=False,
)

theta = np.deg2rad(theta_deg)
dtheta_grid = theta[1] - theta[0]

# closed orbit geometry
r_E = np.array([
    np.interp(KINETIC_ENERGY_MEV, Ek_axis, r_mat[:, j])
    for j in range(len(fi_axis))
])

r_cell, dr_dtheta = periodic_cubic_eval_full_rad(
    fi_axis,
    r_E,
    theta,
)

ds_dtheta = np.sqrt(r_cell**2 + dr_dtheta**2)

s_nodes, cell_length = build_periodic_s_grid(
    ds_dtheta,
    dtheta_grid,
)

ring_length = cell_length * N_CELLS

# beta(theta)
betaR_axis = np.array([
    np.interp(KINETIC_ENERGY_MEV, twiss_Ek_axis, betaR_mat[:, j])
    for j in range(betaR_mat.shape[1])
])

betaZ_axis = np.array([
    np.interp(KINETIC_ENERGY_MEV, twiss_Ek_axis, betaZ_mat[:, j])
    for j in range(betaZ_mat.shape[1])
])

betaR_raw_theta = periodic_cubic_eval_cell_deg(
    theta_axis_deg,
    betaR_axis,
    theta_deg,
    CELL_START_DEG,
    CELL_END_DEG,
)

betaZ_raw_theta = periodic_cubic_eval_cell_deg(
    theta_axis_deg,
    betaZ_axis,
    theta_deg,
    CELL_START_DEG,
    CELL_END_DEG,
)

# uniform s grid
N_S = N_THETA_CELL

s_uniform, betaR_raw_s = periodic_interp_to_uniform_s(
    s_nodes,
    betaR_raw_theta,
    cell_length,
    N_S,
)

_, betaZ_raw_s = periodic_interp_to_uniform_s(
    s_nodes,
    betaZ_raw_theta,
    cell_length,
    N_S,
)

# smooth beta and derivatives
betaR_s, betaR_p_s, betaR_pp_s = fourier_filter_derivatives_uniform_s(
    betaR_raw_s,
    cell_length,
    cutoff_frac=CUTOFF_FRAC,
    order=FILTER_ORDER,
)

betaZ_s, betaZ_p_s, betaZ_pp_s = fourier_filter_derivatives_uniform_s(
    betaZ_raw_s,
    cell_length,
    cutoff_frac=CUTOFF_FRAC,
    order=FILTER_ORDER,
)

# self-consistent Twiss
alphaR_s = -0.5 * betaR_p_s
alphaZ_s = -0.5 * betaZ_p_s

alphaR_p_s = -0.5 * betaR_pp_s
alphaZ_p_s = -0.5 * betaZ_pp_s

gammaR_s = (1.0 + alphaR_s**2) / betaR_s
gammaZ_s = (1.0 + alphaZ_s**2) / betaZ_s

KR_s = (alphaR_p_s + gammaR_s) / betaR_s
KZ_s = (alphaZ_p_s + gammaZ_s) / betaZ_s

# interpolate back to theta grid for Numba tracking
betaR = periodic_interp_uniform_s_to_theta(
    s_uniform,
    betaR_s,
    s_nodes,
    cell_length,
)

betaZ = periodic_interp_uniform_s_to_theta(
    s_uniform,
    betaZ_s,
    s_nodes,
    cell_length,
)

alphaR = periodic_interp_uniform_s_to_theta(
    s_uniform,
    alphaR_s,
    s_nodes,
    cell_length,
)

alphaZ = periodic_interp_uniform_s_to_theta(
    s_uniform,
    alphaZ_s,
    s_nodes,
    cell_length,
)

KR = periodic_interp_uniform_s_to_theta(
    s_uniform,
    KR_s,
    s_nodes,
    cell_length,
)

KZ = periodic_interp_uniform_s_to_theta(
    s_uniform,
    KZ_s,
    s_nodes,
    cell_length,
)

# Twiss consistency checks
err_beta_alpha_R = betaR_p_s + 2.0 * alphaR_s
err_beta_alpha_Z = betaZ_p_s + 2.0 * alphaZ_s

err_invariant_R = betaR_s * gammaR_s - alphaR_s**2 - 1.0
err_invariant_Z = betaZ_s * gammaZ_s - alphaZ_s**2 - 1.0

err_alpha_K_R = alphaR_p_s + gammaR_s - KR_s * betaR_s
err_alpha_K_Z = alphaZ_p_s + gammaZ_s - KZ_s * betaZ_s

gammaR_p_chain_s = (
    2.0 * alphaR_s * alphaR_p_s * betaR_s
    - (1.0 + alphaR_s**2) * betaR_p_s
) / (betaR_s**2)

gammaZ_p_chain_s = (
    2.0 * alphaZ_s * alphaZ_p_s * betaZ_s
    - (1.0 + alphaZ_s**2) * betaZ_p_s
) / (betaZ_s**2)

err_gamma_K_R = gammaR_p_chain_s - 2.0 * KR_s * alphaR_s
err_gamma_K_Z = gammaZ_p_chain_s - 2.0 * KZ_s * alphaZ_s

# beta integral tune
ds_uniform = cell_length / N_S

muR_cell_beta = np.sum(1.0 / betaR_s) * ds_uniform
muZ_cell_beta = np.sum(1.0 / betaZ_s) * ds_uniform

QR_beta_abs = N_CELLS * muR_cell_beta / (2.0 * np.pi)
QZ_beta_abs = N_CELLS * muZ_cell_beta / (2.0 * np.pi)

# 用新的自洽 beta 积分结果覆盖 bare tune
QR_BARE_ABS = QR_beta_abs
QZ_BARE_ABS = QZ_beta_abs

QR_BARE_PLOT = QR_BARE_ABS + QX_DISPLAY_SHIFT
QR_CORE_PLOT = QR_CORE_ABS + QX_DISPLAY_SHIFT

betaR0 = betaR[0]
betaZ0 = betaZ[0]

alphaR0 = alphaR[0]
alphaZ0 = alphaZ[0]

# beam size and space charge parameters
gamma_rel = 1.0 + KINETIC_ENERGY_MEV / MP_MEV
beta_rel = np.sqrt(1.0 - 1.0 / gamma_rel**2)

eps_rms = EMITTANCE_40PI_MM_MRAD * 1e-6

if EMITTANCE_IS_NORMALIZED:
    eps_rms /= beta_rel * gamma_rel

sigmaR = np.sqrt(betaR * eps_rms)
sigmaZ = np.sqrt(betaZ * eps_rms)

line_charge = N_PROTONS * QE / BUNCH_LENGTH_M

force_factor = QE / (
    gamma_rel * MP_KG * beta_rel**2 * C_LIGHT**2
)

field_coef = line_charge / (
    4.0 * np.pi * EPS0 * gamma_rel**2
)

quad_x, quad_w = np.polynomial.legendre.leggauss(N_QUAD)
quad_s = 0.5 * (quad_x + 1.0)
quad_w = 0.5 * quad_w

theta_steps = np.deg2rad(
    np.linspace(
        CELL_START_DEG,
        CELL_END_DEG,
        N_STEPS_PER_CELL + 1,
        endpoint=True,
    )
)

theta0 = theta[0]
dtheta_grid = theta[1] - theta[0]


def _rms(y):
    y = np.asarray(y)
    return np.sqrt(np.mean(y * y))


print("\n" + "=" * 72)
print("Strong-smoothed self-consistent theory lattice")
print("=" * 72)
print(f"CUTOFF_FRAC = {CUTOFF_FRAC}")
print(f"FILTER_ORDER = {FILTER_ORDER}")
print(f"cell_length = {cell_length:.12e} m")
print(f"ring_length = {ring_length:.12e} m")
print(f"muR_cell_beta = {muR_cell_beta:.12e} rad/cell")
print(f"muZ_cell_beta = {muZ_cell_beta:.12e} rad/cell")
print(f"QR_beta_abs = {QR_beta_abs:.12f}")
print(f"QZ_beta_abs = {QZ_beta_abs:.12f}")

print("\nTwiss consistency on uniform s grid:")
print(f"R: rms(betaR_prime + 2 alphaR) = {_rms(err_beta_alpha_R):.8e}")
print(f"R: rms(betaR gammaR - alphaR^2 - 1) = {_rms(err_invariant_R):.8e}")
print(f"R: rms(alphaR_prime + gammaR - KR betaR) = {_rms(err_alpha_K_R):.8e}")
print(f"R: rms(gammaR_prime - 2 KR alphaR) = {_rms(err_gamma_K_R):.8e}")

print(f"Z: rms(betaZ_prime + 2 alphaZ) = {_rms(err_beta_alpha_Z):.8e}")
print(f"Z: rms(betaZ gammaZ - alphaZ^2 - 1) = {_rms(err_invariant_Z):.8e}")
print(f"Z: rms(alphaZ_prime + gammaZ - KZ betaZ) = {_rms(err_alpha_K_Z):.8e}")
print(f"Z: rms(gammaZ_prime - 2 KZ alphaZ) = {_rms(err_gamma_K_Z):.8e}")

print("\n" + "=" * 72)
print("Theory model parameters")
print("=" * 72)
print(f"line_charge = {line_charge:.8e} C/m")
print(f"beta_rel = {beta_rel:.8e}, gamma_rel = {gamma_rel:.8e}")
print(f"eps_rms = {eps_rms:.8e} m rad")
print(f"sigmaR mean = {sigmaR.mean() * 1e3:.6f} mm")
print(f"sigmaZ mean = {sigmaZ.mean() * 1e3:.6f} mm")
print(f"KR min/mean/max = {KR.min():.8e}, {KR.mean():.8e}, {KR.max():.8e}")
print(f"KZ min/mean/max = {KZ.min():.8e}, {KZ.mean():.8e}, {KZ.max():.8e}")
print(f"N_STEPS_PER_CELL = {N_STEPS_PER_CELL}")
print(f"N_TURNS_THEORY = {N_TURNS_THEORY}")
print(f"QX_DISPLAY_SHIFT = {QX_DISPLAY_SHIFT:.6f}")


np.savetxt(
    "strong_smoothed_theory_lattice_theta.csv",
    np.column_stack([
        theta_deg,
        s_nodes,
        r_cell,
        ds_dtheta,
        betaR,
        alphaR,
        KR,
        sigmaR,
        betaZ,
        alphaZ,
        KZ,
        sigmaZ,
    ]),
    delimiter=",",
    header=(
        "theta_deg,s,r_cell,ds_dtheta,"
        "betaR,alphaR,KR,sigmaR,"
        "betaZ,alphaZ,KZ,sigmaZ"
    ),
    comments="",
)

np.savetxt(
    "strong_smoothed_theory_lattice_s.csv",
    np.column_stack([
        s_uniform,
        betaR_raw_s,
        betaR_s,
        alphaR_s,
        gammaR_s,
        KR_s,
        betaZ_raw_s,
        betaZ_s,
        alphaZ_s,
        gammaZ_s,
        KZ_s,
    ]),
    delimiter=",",
    header=(
        "s,"
        "betaR_raw,betaR_smooth,alphaR,gammaR,KR,"
        "betaZ_raw,betaZ_smooth,alphaZ,gammaZ,KZ"
    ),
    comments="",
)

print("Saved: strong_smoothed_theory_lattice_theta.csv")
print("Saved: strong_smoothed_theory_lattice_s.csv")


# ============================================================
# Numba 理论 nonlinear tracking
# ============================================================
@njit(cache=True)
def interp_uniform(th, th0, dth, arr):
    """
    周期线性插值。
    arr 对应一个 cell 上 endpoint=False 的均匀 theta 网格。
    """
    n = arr.shape[0]
    period = dth * n

    u = (th - th0) % period
    x = u / dth

    i = int(np.floor(x)) % n
    a = x - np.floor(x)
    i1 = (i + 1) % n

    return (1.0 - a) * arr[i] + a * arr[i1]


@njit(cache=True)
def gfield_scalar(x, z, sx, sz, quad_s, quad_w, field_coef):
    Ex = 0.0
    Ez = 0.0

    sx2 = sx * sx
    sz2 = sz * sz

    t0 = sx2 if sx2 > sz2 else sz2

    for iq in range(quad_s.shape[0]):
        s = quad_s[iq]
        w = quad_w[iq]

        t = t0 * s / (1.0 - s)
        dt = t0 / ((1.0 - s) * (1.0 - s))

        ax = sx2 + t
        az = sz2 + t

        root = np.sqrt(ax * az)
        expo = np.exp(-0.5 * (x * x / ax + z * z / az))

        Ex += w * dt * x / (ax * root) * expo
        Ez += w * dt * z / (az * root) * expo

    return field_coef * Ex, field_coef * Ez


@njit(cache=True)
def rhs_scalar(
    th, x, xp, z, zp,
    theta0, dtheta_grid,
    ds_dtheta, KR, KZ, sigmaR, sigmaZ,
    quad_s, quad_w, field_coef, force_factor,
):
    h = interp_uniform(th, theta0, dtheta_grid, ds_dtheta)
    kr = interp_uniform(th, theta0, dtheta_grid, KR)
    kz = interp_uniform(th, theta0, dtheta_grid, KZ)
    sx = interp_uniform(th, theta0, dtheta_grid, sigmaR)
    sz = interp_uniform(th, theta0, dtheta_grid, sigmaZ)

    Ex, Ez = gfield_scalar(x, z, sx, sz, quad_s, quad_w, field_coef)

    dx = h * xp
    dxp = h * (-kr * x + force_factor * Ex)

    dz = h * zp
    dzp = h * (-kz * z + force_factor * Ez)

    return dx, dxp, dz, dzp


@njit(parallel=True, cache=True)
def track_theory_numba(
    AR, AZ, theta_steps, theta0, dtheta_grid,
    ds_dtheta, KR, KZ, sigmaR, sigmaZ,
    quad_s, quad_w, field_coef, force_factor,
    n_cells, n_turns,
):
    n_part = AR.shape[0]
    n_steps = theta_steps.shape[0] - 1

    x_turn = np.zeros((n_turns, n_part))
    xp_turn = np.zeros((n_turns, n_part))
    z_turn = np.zeros((n_turns, n_part))
    zp_turn = np.zeros((n_turns, n_part))

    for ip in prange(n_part):
        x = AR[ip]
        xp = 0.0
        z = AZ[ip]
        zp = 0.0

        for it in range(n_turns):
            for ic in range(n_cells):
                for js in range(n_steps):
                    th = theta_steps[js]
                    dth = theta_steps[js + 1] - theta_steps[js]

                    k1x, k1xp, k1z, k1zp = rhs_scalar(
                        th, x, xp, z, zp,
                        theta0, dtheta_grid,
                        ds_dtheta, KR, KZ, sigmaR, sigmaZ,
                        quad_s, quad_w, field_coef, force_factor,
                    )

                    k2x, k2xp, k2z, k2zp = rhs_scalar(
                        th + 0.5 * dth,
                        x + 0.5 * dth * k1x,
                        xp + 0.5 * dth * k1xp,
                        z + 0.5 * dth * k1z,
                        zp + 0.5 * dth * k1zp,
                        theta0, dtheta_grid,
                        ds_dtheta, KR, KZ, sigmaR, sigmaZ,
                        quad_s, quad_w, field_coef, force_factor,
                    )

                    k3x, k3xp, k3z, k3zp = rhs_scalar(
                        th + 0.5 * dth,
                        x + 0.5 * dth * k2x,
                        xp + 0.5 * dth * k2xp,
                        z + 0.5 * dth * k2z,
                        zp + 0.5 * dth * k2zp,
                        theta0, dtheta_grid,
                        ds_dtheta, KR, KZ, sigmaR, sigmaZ,
                        quad_s, quad_w, field_coef, force_factor,
                    )

                    k4x, k4xp, k4z, k4zp = rhs_scalar(
                        th + dth,
                        x + dth * k3x,
                        xp + dth * k3xp,
                        z + dth * k3z,
                        zp + dth * k3zp,
                        theta0, dtheta_grid,
                        ds_dtheta, KR, KZ, sigmaR, sigmaZ,
                        quad_s, quad_w, field_coef, force_factor,
                    )

                    x += dth * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
                    xp += dth * (k1xp + 2.0 * k2xp + 2.0 * k3xp + k4xp) / 6.0

                    z += dth * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
                    zp += dth * (k1zp + 2.0 * k2zp + 2.0 * k3zp + k4zp) / 6.0

            x_turn[it, ip] = x
            xp_turn[it, ip] = xp
            z_turn[it, ip] = z
            zp_turn[it, ip] = zp

    return x_turn, xp_turn, z_turn, zp_turn


# ============================================================
# PyFFA tune 和振幅，读取 tracking 粒子
# ============================================================
paths = sorted(Path(DATA_DIR).glob("*.npz"), key=get_pid)

if MAX_FILES is not None and len(paths) > MAX_FILES:
    rng = np.random.default_rng(RNG_SEED)
    paths = sorted(list(rng.choice(paths, MAX_FILES, replace=False)), key=get_pid)

pid = []
QR_py = []
QZ_py = []
AR = []
AZ = []

t0 = time.time()

for i, path in enumerate(paths):
    arr = np.load(path)["particles"]
    arr = arr[arr[:, -1].argsort()]
    arr = arr[N_SKIP:]

    r = arr[:, COL_R].astype(float)
    pr = arr[:, COL_PR].astype(float)
    z = arr[:, COL_Z].astype(float)
    pz = arr[:, COL_PZ].astype(float)
    fi = arr[:, COL_FI].astype(float)
    Ek = arr[:, COL_EK].astype(float)

    if SUBTRACT_CLOSED_ORBIT:
        x, px = sub_co(r, pr, fi, Ek)
    else:
        x, px = r, pr

    qRf = cfft_frac_interp(x, px, COMPLEX_SIGN_R, PAD_FACTOR)
    qZf = cfft_frac_interp(z, pz, COMPLEX_SIGN_Z, PAD_FACTOR)

    if not (np.isfinite(qRf) and np.isfinite(qZf)):
        continue

    pid.append(get_pid(path))

    QR_py.append(abs_tune(qRf, QR_BARE_ABS, QR_CORE_ABS))
    QZ_py.append(abs_tune(qZf, QZ_BARE_ABS, QZ_CORE_ABS))

    AR.append(0.5 * (x.max() - x.min()))
    AZ.append(0.5 * (z.max() - z.min()))

    if (i + 1) % 1000 == 0:
        print(f"Read PyFFA particle {i + 1}/{len(paths)}")

pid = np.array(pid)
QR_py = np.array(QR_py)
QZ_py = np.array(QZ_py)
AR = np.array(AR)
AZ = np.array(AZ)

A_total_mm = np.sqrt((AR * 1e3)**2 + (AZ * 1e3)**2)
Np_py = len(pid)

print("\n" + "=" * 72)
print("PyFFA tune extracted from tracking")
print("=" * 72)
print(f"n_particles_pyffa = {Np_py}")
print(f"QR_py min/mean/max = {QR_py.min():.8f}, {QR_py.mean():.8f}, {QR_py.max():.8f}")
print(f"QZ_py min/mean/max = {QZ_py.min():.8f}, {QZ_py.mean():.8f}, {QZ_py.max():.8f}")
print(f"PyFFA data reading elapsed time = {time.time() - t0:.2f} s")


# ============================================================
# 从 PyFFA 点中抽样用于理论积分
# ============================================================
rng = np.random.default_rng(RNG_SEED + 1)

n_th = min(N_THEORY_PARTICLES, Np_py)
idx_th = rng.choice(Np_py, size=n_th, replace=False)

pid_th_ref = pid[idx_th]
QR_py_ref = QR_py[idx_th]
QZ_py_ref = QZ_py[idx_th]

AR_th = AR[idx_th]
AZ_th = AZ[idx_th]
A_total_th_mm = A_total_mm[idx_th]

print("\n" + "=" * 72)
print("Theory particles selected from PyFFA sample")
print("=" * 72)
print(f"n_particles_theory = {n_th}")


# ============================================================
# 作用量：J = JR + JZ，单位 mm mrad
# ============================================================
JR_py = AR**2 / (2.0 * betaR0)
JZ_py = AZ**2 / (2.0 * betaZ0)
Jtot_py = (JR_py + JZ_py) * 1e6

JR_th_ref = AR_th**2 / (2.0 * betaR0)
JZ_th_ref = AZ_th**2 / (2.0 * betaZ0)
Jtot_th_ref = (JR_th_ref + JZ_th_ref) * 1e6


# ============================================================
# Numba warm-up
# ============================================================
print("\nStart Numba warm-up...")
t0 = time.time()

_ = track_theory_numba(
    AR_th[:2].astype(np.float64).copy(),
    AZ_th[:2].astype(np.float64).copy(),
    theta_steps.astype(np.float64),
    theta0,
    dtheta_grid,
    ds_dtheta.astype(np.float64),
    KR.astype(np.float64),
    KZ.astype(np.float64),
    sigmaR.astype(np.float64),
    sigmaZ.astype(np.float64),
    quad_s.astype(np.float64),
    quad_w.astype(np.float64),
    field_coef,
    force_factor,
    N_CELLS,
    1,
)

print(f"Numba warm-up finished. Elapsed time = {time.time() - t0:.2f} s")


# ============================================================
# 正式理论积分
# ============================================================
print("\nStart full Numba theory tracking...")
t0 = time.time()

x_turn, xp_turn, z_turn, zp_turn = track_theory_numba(
    AR_th.astype(np.float64),
    AZ_th.astype(np.float64),
    theta_steps.astype(np.float64),
    theta0,
    dtheta_grid,
    ds_dtheta.astype(np.float64),
    KR.astype(np.float64),
    KZ.astype(np.float64),
    sigmaR.astype(np.float64),
    sigmaZ.astype(np.float64),
    quad_s.astype(np.float64),
    quad_w.astype(np.float64),
    field_coef,
    force_factor,
    N_CELLS,
    N_TURNS_THEORY,
)

print(f"Full Numba theory tracking finished. Elapsed time = {time.time() - t0:.2f} s")


# ============================================================
# 理论 tune 提取
# ============================================================
print("\nExtracting theory tunes...")
t0 = time.time()

QR_LOW = QR_CORE_ABS - 0.05
QR_HIGH = QR_BARE_ABS + 0.03

QZ_LOW = QZ_CORE_ABS - 0.05
QZ_HIGH = QZ_BARE_ABS + 0.03

QR_th_raw = np.array([
    fft_range(x_turn[:, i], xp_turn[:, i], betaR0, alphaR0, QR_LOW, QR_HIGH)
    for i in range(n_th)
])

QZ_th_raw = np.array([
    fft_range(z_turn[:, i], zp_turn[:, i], betaZ0, alphaZ0, QZ_LOW, QZ_HIGH)
    for i in range(n_th)
])

good_th = np.isfinite(QR_th_raw) & np.isfinite(QZ_th_raw)

pid_th_ref = pid_th_ref[good_th]

QR_py_ref = QR_py_ref[good_th]
QZ_py_ref = QZ_py_ref[good_th]

AR_th = AR_th[good_th]
AZ_th = AZ_th[good_th]

A_total_th_mm = A_total_th_mm[good_th]
Jtot_th_ref = Jtot_th_ref[good_th]

QR_th_raw = QR_th_raw[good_th]
QZ_th_raw = QZ_th_raw[good_th]

QR_th_plot = QR_th_raw + QX_DISPLAY_SHIFT
QZ_th_plot = QZ_th_raw

dQR_err_raw = QR_py_ref - QR_th_raw
dQZ_err_raw = QZ_py_ref - QZ_th_raw

dQR_err_plot = QR_py_ref - QR_th_plot
dQZ_err_plot = QZ_py_ref - QZ_th_plot

print(f"Theory tune extraction elapsed time = {time.time() - t0:.2f} s")

print("\n" + "=" * 72)
print("Theory tune from nonlinear Hill tracking")
print("=" * 72)
print(f"n_particles_theory_valid = {len(QR_th_raw)}")
print(f"QR_th_raw min/mean/max = {QR_th_raw.min():.8f}, {QR_th_raw.mean():.8f}, {QR_th_raw.max():.8f}")
print(f"QR_th_plot min/mean/max = {QR_th_plot.min():.8f}, {QR_th_plot.mean():.8f}, {QR_th_plot.max():.8f}")
print(f"QZ_th min/mean/max = {QZ_th_raw.min():.8f}, {QZ_th_raw.mean():.8f}, {QZ_th_raw.max():.8f}")

print("\n" + "=" * 72)
print("PyFFA selected particles - theory")
print("=" * 72)
print(f"dQR raw err min/mean/max = {dQR_err_raw.min():.8e}, {dQR_err_raw.mean():.8e}, {dQR_err_raw.max():.8e}")
print(f"dQR plot err min/mean/max = {dQR_err_plot.min():.8e}, {dQR_err_plot.mean():.8e}, {dQR_err_plot.max():.8e}")
print(f"dQZ err min/mean/max = {dQZ_err_raw.min():.8e}, {dQZ_err_raw.mean():.8e}, {dQZ_err_raw.max():.8e}")
print(f"QR RMSE raw  = {np.sqrt(np.mean(dQR_err_raw**2)):.8e}")
print(f"QR RMSE plot = {np.sqrt(np.mean(dQR_err_plot**2)):.8e}")
print(f"QZ RMSE      = {np.sqrt(np.mean(dQZ_err_raw**2)):.8e}")


# ============================================================
# 保存
# ============================================================
out_py = np.column_stack([
    pid,
    QR_py,
    QZ_py,
    AR * 1e3,
    AZ * 1e3,
    A_total_mm,
    Jtot_py,
])

np.savetxt(
    RESULTS_DIR / "pyffa_tracking_tune_20000_hist2d.csv",
    out_py,
    delimiter=",",
    header="pid,QR_py,QZ_py,AR_mm,AZ_mm,A_total_mm,J_total_mm_mrad",
    comments="",
)

out_th = np.column_stack([
    pid_th_ref,
    AR_th * 1e3,
    AZ_th * 1e3,
    A_total_th_mm,
    Jtot_th_ref,
    QR_py_ref,
    QZ_py_ref,
    QR_th_raw,
    QZ_th_raw,
    QR_th_plot,
    QZ_th_plot,
    QR_py_ref - QR_th_raw,
    QZ_py_ref - QZ_th_raw,
    QR_py_ref - QR_th_plot,
    QZ_py_ref - QZ_th_plot,
])

np.savetxt(
    RESULTS_DIR / "theory_tracking_tune_2000_scatter.csv",
    out_th,
    delimiter=",",
    header=(
        "pid,AR_mm,AZ_mm,A_total_mm,J_total_mm_mrad,"
        "QR_py_ref,QZ_py_ref,"
        "QR_th_raw,QZ_th_raw,"
        "QR_th_plot,QZ_th_plot,"
        "QR_py_minus_th_raw,QZ_py_minus_th_raw,"
        "QR_py_minus_th_plot,QZ_py_minus_th_plot"
    ),
    comments="",
)

print(f"\nSaved: {RESULTS_DIR / 'pyffa_tracking_tune_20000_hist2d.csv'}")
print(f"Saved: {RESULTS_DIR / 'theory_tracking_tune_2000_scatter.csv'}")


# ============================================================
# 图 1：PyFFA tracking，用 hist2d
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)

h = ax.hist2d(
    QR_py,
    QZ_py,
    bins=[256, 256],
    range=[QR_XLIM, QZ_XLIM],
    norm=LogNorm(),
    cmap="viridis",
)

ax.axvline(QR_BARE_PLOT, ls="--", lw=2.0, color="tab:blue")
ax.axhline(QZ_BARE_ABS, ls="--", lw=2.0, color="tab:blue")
ax.axvline(QR_CORE_PLOT, ls=":", lw=2.4, color="tab:blue")
ax.axhline(QZ_CORE_ABS, ls=":", lw=2.4, color="tab:blue")

ax.scatter(
    QR_BARE_PLOT,
    QZ_BARE_ABS,
    marker="o",
    s=160,
    facecolor="white",
    edgecolor="blue",
    linewidth=2.8,
    zorder=12,
    label="Bare tune",
)

ax.scatter(
    QR_CORE_PLOT,
    QZ_CORE_ABS,
    marker="*",
    s=300,
    color="red",
    edgecolor="black",
    linewidth=2.4,
    zorder=13,
    label="Max SC tune shift",
)
ax.text(
        0.04, 0.7,
        rf"({FigText1})",
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
            pad=2.0
        ),
        zorder=20,
    )

ax.set_xlabel(r"$Q_x$", fontsize=24)
ax.set_ylabel(r"$Q_z$", fontsize=24)
ax.set_xlim(QR_XLIM)
ax.set_ylim(QZ_XLIM)
ax.tick_params(axis="both", labelsize=20)
legend = ax.legend(fontsize=15, loc="upper left")
legend.set_zorder(30)
draw_resonances_up_to_order3(ax, QR_py, QZ_py, QR_XLIM, QZ_XLIM)

cb = plt.colorbar(h[3], ax=ax)
cb.set_label("Counts", fontsize=19)
cb.ax.tick_params(labelsize=12)

plt.savefig(
    OUTPUT_DIR / f"fig_tune_gauss_{int(EMITTANCE_40PI_MM_MRAD)}_pic.png",
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# 图 2：Theory tracking
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)

sc = ax.scatter(
    QR_th_plot,
    QZ_th_plot,
    c=Jtot_th_ref,
    cmap="plasma",
    s=18,
    alpha=0.75,
    linewidths=0,
    rasterized=True,
)

ax.axvline(QR_BARE_PLOT, ls="--", lw=2.0, color="tab:blue")
ax.axhline(QZ_BARE_ABS, ls="--", lw=2.0, color="tab:blue")
ax.axvline(QR_CORE_PLOT, ls=":", lw=2.4, color="tab:blue")
ax.axhline(QZ_CORE_ABS, ls=":", lw=2.4, color="tab:blue")

ax.scatter(
    QR_BARE_PLOT,
    QZ_BARE_ABS,
    marker="o",
    s=160,
    facecolor="white",
    edgecolor="blue",
    linewidth=2.8,
    zorder=12,
    label="Bare tune",
)

ax.scatter(
    QR_CORE_PLOT,
    QZ_CORE_ABS,
    marker="*",
    s=300,
    color="red",
    edgecolor="black",
    linewidth=2.4,
    zorder=13,
    label="Max SC tune shift",
)
ax.text(
        0.04, 0.7,
        rf"({FigText2})",
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
            pad=2.0
        ),
        zorder=20,
    )
ax.set_xlabel(r"$Q_x$", fontsize=24)
ax.set_ylabel(r"$Q_z$", fontsize=24)
ax.set_xlim(QR_XLIM)
ax.set_ylim(QZ_XLIM)
ax.tick_params(axis="both", labelsize=20)
legend = ax.legend(fontsize=15, loc="upper left")
legend.set_zorder(30)
draw_resonances_up_to_order3(ax, QR_th_plot, QZ_th_plot, QR_XLIM, QZ_XLIM)

cb = plt.colorbar(sc, ax=ax)
cb.set_label(r"$J_x + J_z$ [$\pi$ mm mrad]", fontsize=19)
cb.ax.tick_params(labelsize=12)

plt.savefig(
    OUTPUT_DIR / f"fig_tune_gauss_{int(EMITTANCE_40PI_MM_MRAD)}_theory.png",
    dpi=300,
    bbox_inches="tight",
)

if ARGS.no_show:
    plt.close("all")
else:
    plt.show()
