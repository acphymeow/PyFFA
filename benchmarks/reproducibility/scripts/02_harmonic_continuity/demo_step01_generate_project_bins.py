"""最小化演示：生成测试束团，然后投影到一圈并构造 histogram bins。

这个脚本只解释前三步，不计算 occupied interval：

1. 在原始纵向坐标上生成 h=2 的两个测试束团；
2. 用 phi_mod = phi mod 2*pi 投影到一圈；
3. 在 [0, 2*pi) 上构造等宽 histogram bins。
4. 平滑环周 histogram，寻找两个最宽低密度区段，并把最宽区段
   的中点设为公共切口。

默认依次弹出三张图。关闭当前窗口后，脚本才继续显示下一步。
无图形界面的批处理环境可加 --no-show。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


SHOW_FIGURES = "--no-show" not in sys.argv
if not SHOW_FIGURES:
    os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np


SEED = 20260821
TWO_PI = 2.0 * np.pi
N_BINS = 360
HARMONIC_NUMBER = 2
SMOOTH_SIGMA_BINS = 1.5
GAP_DENSITY_FRACTION = 0.03
MIN_GAP_BINS = 4
HERE = Path(__file__).resolve().parent
OUT_STEP1 = HERE / "demo_step01_generated_test_bunches.png"
OUT_STEP2 = HERE / "demo_step02_projected_histogram_bins.png"
OUT_STEP3 = HERE / "demo_step03_find_common_cut.png"

BLUE = "#3565f3"
ORANGE = "#d18b28"
GRAY = "#8f8f8f"


def generate_test_bunches(rng):
    """生成两个测试束团；bunch 1 故意跨越原来的 0/2pi 边界。"""
    # bunch 1 以 2*pi 为中心，因此一部分粒子位于 2*pi 左侧，
    # 另一部分粒子位于 2*pi 右侧。
    phi_bunch_1_raw = rng.normal(loc=TWO_PI, scale=0.22, size=6000)

    # bunch 2 位于一圈中部，不跨越周期边界。
    phi_bunch_2_raw = rng.normal(loc=np.pi, scale=0.28, size=4500)

    phi_raw = np.concatenate((phi_bunch_1_raw, phi_bunch_2_raw))
    bunch_id = np.concatenate((
        np.ones(phi_bunch_1_raw.size, dtype=int),
        np.full(phi_bunch_2_raw.size, 2, dtype=int),
    ))
    return phi_raw, bunch_id


def add_phase_ticks(ax):
    ax.set_xticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, TWO_PI])
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])


def show_or_close(fig, step_name):
    if SHOW_FIGURES:
        print(f"{step_name}: close this window to continue.")
        plt.show()
    plt.close(fig)


def plot_generated_bunches(phi_raw, bunch_id, rng):
    """第一张图：显示取模之前生成的粒子。"""
    fig, axes = plt.subplots(
        2, 1, figsize=(8.4, 5.7), sharex=True,
        gridspec_kw={"height_ratios": [0.72, 2.2], "hspace": 0.08},
    )

    # 只抽取部分粒子画散点，避免图上点过密；后面的直方图仍使用全部粒子。
    for bunch, color, y0 in ((1, BLUE, 1.0), (2, ORANGE, 2.0)):
        indices = np.flatnonzero(bunch_id == bunch)
        draw_n = min(900, indices.size)
        draw_indices = rng.choice(indices, size=draw_n, replace=False)
        jitter = rng.normal(0.0, 0.055, size=draw_n)
        axes[0].scatter(
            phi_raw[draw_indices], y0 + jitter,
            s=7, color=color, alpha=0.42, edgecolors="none",
        )

    axes[0].set_yticks([1.0, 2.0])
    axes[0].set_yticklabels(["bunch 1", "bunch 2"])
    axes[0].set_ylim(0.65, 2.35)
    axes[0].set_title("Step 1: generate two bunches in the raw coordinate")
    axes[0].grid(axis="x", alpha=0.16)

    raw_edges = np.linspace(0.0, TWO_PI + 0.85, 100)
    axes[1].hist(
        phi_raw[bunch_id == 1], bins=raw_edges,
        color=BLUE, alpha=0.78, label=r"bunch 1: crosses $2\pi$",
    )
    axes[1].hist(
        phi_raw[bunch_id == 2], bins=raw_edges,
        color=ORANGE, alpha=0.78, label="bunch 2",
    )
    axes[1].axvline(TWO_PI, color=GRAY, ls="--", lw=1.2)
    axes[1].text(
        TWO_PI - 0.04, axes[1].get_ylim()[1] * 0.92,
        r"original boundary $2\pi$", color=GRAY,
        ha="right", va="top", fontsize=9,
    )
    axes[1].set_xlim(0.0, TWO_PI + 0.85)
    axes[1].set_ylabel("particles / raw bin")
    axes[1].set_xlabel(r"raw longitudinal coordinate $\phi$")
    add_phase_ticks(axes[1])
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", alpha=0.16)

    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.91)
    fig.savefig(OUT_STEP1, dpi=220)
    show_or_close(fig, "Step 1")


def project_and_build_bins(phi_raw, bunch_id):
    """第二步的实际数值计算：取模，然后调用 np.histogram。"""
    phi_mod = np.mod(phi_raw, TWO_PI)
    bin_edges = np.linspace(0.0, TWO_PI, N_BINS + 1)

    counts_total, _ = np.histogram(phi_mod, bins=bin_edges)
    counts_bunch_1, _ = np.histogram(phi_mod[bunch_id == 1], bins=bin_edges)
    counts_bunch_2, _ = np.histogram(phi_mod[bunch_id == 2], bins=bin_edges)

    # 两个分束团 histogram 相加，应严格等于总 histogram。
    if not np.array_equal(counts_total, counts_bunch_1 + counts_bunch_2):
        raise AssertionError("Per-bunch bin counts do not reproduce total counts")
    if counts_total.sum() != phi_raw.size:
        raise AssertionError("Histogram lost particles during one-turn projection")

    return phi_mod, bin_edges, counts_total, counts_bunch_1, counts_bunch_2


def gaussian_smooth_periodic(values, sigma_bins=SMOOTH_SIGMA_BINS):
    """在环周方向平滑；数组首尾互相连接，不使用零填充。"""
    half_width = max(2, int(np.ceil(4.0 * sigma_bins)))
    offsets = np.arange(-half_width, half_width + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()

    smoothed = np.zeros_like(values, dtype=float)
    for shift, weight in zip(range(-half_width, half_width + 1), kernel):
        smoothed += weight * np.roll(values, shift)
    return smoothed


def circular_true_runs(mask):
    """合并连续 True bins；跨越 0/2pi 的首尾区段也会合并。"""
    mask = np.asarray(mask, dtype=bool)
    n_bins = mask.size
    if np.all(mask):
        return [(0, n_bins - 1)]
    if not np.any(mask):
        return []

    # 从一个 False bin 后面开始扫描，避免把跨越数组首尾的同一个
    # 低密度区段错误地拆成两段。
    first_false = int(np.flatnonzero(~mask)[0])
    scan_order = (first_false + 1 + np.arange(n_bins)) % n_bins
    runs = []
    start = previous = None
    for index in scan_order:
        if mask[index]:
            if start is None:
                start = int(index)
            previous = int(index)
        elif start is not None:
            runs.append((start, previous))
            start = previous = None
    if start is not None:
        runs.append((start, previous))
    return runs


def find_common_cut(counts_total, bin_edges, h=HARMONIC_NUMBER):
    """找 h 个最宽低密度区段，并把最宽区段中点作为公共切口。"""
    smoothed = gaussian_smooth_periodic(counts_total)
    threshold = GAP_DENSITY_FRACTION * float(smoothed.max())
    low_density_mask = smoothed <= threshold
    bin_width = bin_edges[1] - bin_edges[0]

    candidates = []
    for start_bin, end_bin in circular_true_runs(low_density_mask):
        length_bins = (end_bin - start_bin) % counts_total.size + 1
        if length_bins < MIN_GAP_BINS:
            continue
        start = float(bin_edges[start_bin])
        length = float(length_bins * bin_width)
        candidates.append({
            "start_bin": int(start_bin),
            "end_bin": int(end_bin),
            "length_bins": int(length_bins),
            "start": start,
            "length": length,
            "midpoint": float((start + 0.5 * length) % TWO_PI),
        })

    if len(candidates) < h:
        raise ValueError(
            f"Only {len(candidates)} low-density gaps found, but h={h}"
        )

    selected_gaps = sorted(
        candidates, key=lambda gap: gap["length"], reverse=True
    )[:h]
    return {
        "smoothed": smoothed,
        "threshold": threshold,
        "low_density_mask": low_density_mask,
        "candidate_gaps": candidates,
        "selected_gaps": selected_gaps,
        "common_cut": selected_gaps[0]["midpoint"],
    }


def plot_projected_bins(
    phi_mod, bunch_id, bin_edges, counts_total, counts_bunch_1, counts_bunch_2, rng
):
    """第二张图：显示取模后的粒子位置和最终 bins。"""
    fig, axes = plt.subplots(
        2, 1, figsize=(8.4, 5.9), sharex=True,
        gridspec_kw={"height_ratios": [0.72, 2.35], "hspace": 0.08},
    )

    for bunch, color, y0 in ((1, BLUE, 1.0), (2, ORANGE, 2.0)):
        indices = np.flatnonzero(bunch_id == bunch)
        draw_n = min(900, indices.size)
        draw_indices = rng.choice(indices, size=draw_n, replace=False)
        jitter = rng.normal(0.0, 0.055, size=draw_n)
        axes[0].scatter(
            phi_mod[draw_indices], y0 + jitter,
            s=7, color=color, alpha=0.42, edgecolors="none",
        )

    axes[0].set_yticks([1.0, 2.0])
    axes[0].set_yticklabels(["bunch 1", "bunch 2"])
    axes[0].set_ylim(0.65, 2.35)
    axes[0].set_title(r"Step 2: project with $\phi_{\rm mod}=\phi\ {\rm mod}\ 2\pi$")
    axes[0].grid(axis="x", alpha=0.16)

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    axes[1].bar(
        centers, counts_bunch_1, width=0.94 * bin_width,
        color=BLUE, edgecolor="white", linewidth=0.45,
        label="bunch 1",
    )
    axes[1].bar(
        centers, counts_bunch_2, bottom=counts_bunch_1,
        width=0.94 * bin_width, color=ORANGE,
        edgecolor="white", linewidth=0.45, label="bunch 2",
    )
    for edge in bin_edges:
        axes[1].axvline(edge, color=GRAY, lw=0.35, alpha=0.18, zorder=0)

    axes[1].text(
        0.5 * np.pi, counts_total.max() * 0.88,
        rf"{N_BINS} equal bins, $\Delta\phi=2\pi/{N_BINS}$",
        ha="center", fontsize=9, color=GRAY,
    )
    axes[1].set_xlim(0.0, TWO_PI)
    axes[1].set_ylim(0.0, counts_total.max() * 1.08)
    axes[1].set_ylabel("particles / bin")
    axes[1].set_xlabel(r"one-turn coordinate $\phi_{\rm mod}$")
    add_phase_ticks(axes[1])
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    axes[1].grid(axis="y", alpha=0.16)

    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.91)
    fig.savefig(OUT_STEP2, dpi=220)
    show_or_close(fig, "Step 2")


def add_periodic_gap_span(ax, start, length, color, alpha):
    """在 0/2pi 横坐标上画可能跨越边界的环形区段。"""
    end = start + length
    if end <= TWO_PI:
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0)
    else:
        ax.axvspan(start, TWO_PI, color=color, alpha=alpha, lw=0)
        ax.axvspan(0.0, end - TWO_PI, color=color, alpha=alpha, lw=0)


def plot_common_cut(phi_mod, bunch_id, bin_edges, counts_total, gap_result, rng):
    """第三张图：显示低密度区段以及由其得到的公共切口。"""
    fig = plt.figure(figsize=(9.6, 4.8))
    grid = fig.add_gridspec(
        1, 2, left=0.06, right=0.97, bottom=0.14, top=0.88,
        width_ratios=[0.88, 1.75], wspace=0.22,
    )
    ring = fig.add_subplot(grid[0, 0])
    hist = fig.add_subplot(grid[0, 1])

    # 左图：把投影后的粒子画在圆环上。
    theta = np.linspace(0.0, TWO_PI, 500)
    ring.plot(np.cos(theta), np.sin(theta), color="#c6c6c6", lw=1.2)
    for bunch, color in ((1, BLUE), (2, ORANGE)):
        indices = np.flatnonzero(bunch_id == bunch)
        draw_n = min(1200, indices.size)
        draw_indices = rng.choice(indices, size=draw_n, replace=False)
        radius = 1.0 + rng.normal(0.0, 0.025, size=draw_n)
        angles = phi_mod[draw_indices]
        ring.scatter(
            radius * np.cos(angles), radius * np.sin(angles),
            s=6, color=color, alpha=0.28, edgecolors="none",
        )

    for gap_index, gap in enumerate(gap_result["selected_gaps"]):
        angle = gap["midpoint"]
        color = "#c73832" if gap_index == 0 else ORANGE
        label = "common cut" if gap_index == 0 else "other gap"
        ring.plot(
            [0.72 * np.cos(angle), 1.25 * np.cos(angle)],
            [0.72 * np.sin(angle), 1.25 * np.sin(angle)],
            color=color, ls="--", lw=1.4,
        )
        ring.text(
            1.37 * np.cos(angle), 1.37 * np.sin(angle), label,
            color=color, ha="center", va="center", fontsize=8.5,
        )
    ring.set_xlim(-1.48, 1.48)
    ring.set_ylim(-1.48, 1.48)
    ring.set_aspect("equal")
    ring.axis("off")
    ring.set_title("one-turn particle distribution", fontsize=10)

    # 右图：显示 bins、周期平滑密度，以及两个最宽低密度区段。
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    hist.bar(
        centers, counts_total, width=0.92 * bin_width,
        color="#c7c7c7", edgecolor="none", label="bin counts",
    )
    hist.plot(
        centers, gap_result["smoothed"], color=BLUE, lw=1.6,
        label="periodically smoothed counts",
    )
    hist.axhline(
        gap_result["threshold"], color=GRAY, ls="--", lw=1.1,
        label="3% of peak",
    )

    for gap_index, gap in enumerate(gap_result["selected_gaps"]):
        color = "#c73832" if gap_index == 0 else ORANGE
        alpha = 0.12 if gap_index == 0 else 0.09
        add_periodic_gap_span(hist, gap["start"], gap["length"], color, alpha)
        hist.axvline(gap["midpoint"], color=color, ls="--", lw=1.3)
        label = "widest gap\ncommon cut" if gap_index == 0 else "second gap"
        hist.annotate(
            label,
            xy=(gap["midpoint"], gap_result["threshold"]),
            xytext=(
                gap["midpoint"],
                counts_total.max() * (0.48 if gap_index == 0 else 0.30),
            ),
            ha="center", va="center", fontsize=8.5, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )

    hist.set_xlim(0.0, TWO_PI)
    hist.set_ylim(0.0, counts_total.max() * 1.08)
    add_phase_ticks(hist)
    hist.set_xlabel(r"one-turn coordinate $\phi_{\rm mod}$")
    hist.set_ylabel("particles / bin")
    hist.set_title("select the widest low-density gap", fontsize=10)
    hist.legend(frameon=False, fontsize=8, loc="upper right")
    hist.grid(axis="y", alpha=0.15)

    fig.suptitle("Step 3: find the common cut", fontsize=13)
    fig.savefig(OUT_STEP3, dpi=220)
    show_or_close(fig, "Step 3")


def main():
    rng = np.random.default_rng(SEED)

    # 第一步：生成测试粒子。
    phi_raw, bunch_id = generate_test_bunches(rng)
    print("Step 1: generated test particles")
    print(f"  bunch 1 particles = {np.count_nonzero(bunch_id == 1)}")
    print(f"  bunch 2 particles = {np.count_nonzero(bunch_id == 2)}")
    print(f"  total particles   = {phi_raw.size}")
    plot_generated_bunches(phi_raw, bunch_id, rng)

    # 第二步：投影到 [0, 2*pi)，然后构造 bins。
    result = project_and_build_bins(phi_raw, bunch_id)
    phi_mod, bin_edges, counts_total, counts_bunch_1, counts_bunch_2 = result
    print("Step 2: projected particles and built histogram bins")
    print(f"  phi_mod range     = [{phi_mod.min():.6f}, {phi_mod.max():.6f}) rad")
    print(f"  number of bins    = {N_BINS}")
    print(f"  bin width         = {bin_edges[1] - bin_edges[0]:.6f} rad")
    print(f"  histogram sum     = {counts_total.sum()}")
    print(f"  first 5 bins      = {counts_total[:5].tolist()}")
    print(f"  last 5 bins       = {counts_total[-5:].tolist()}")
    plot_projected_bins(
        phi_mod, bunch_id, bin_edges, counts_total,
        counts_bunch_1, counts_bunch_2, rng,
    )

    # 第三步：选择 h 个最宽低密度区段；最宽区段中点是公共切口。
    gap_result = find_common_cut(counts_total, bin_edges)
    print("Step 3: found low-density gaps and selected the common cut")
    for index, gap in enumerate(gap_result["selected_gaps"], start=1):
        print(
            f"  selected gap {index}: width={gap['length']:.6f} rad, "
            f"midpoint={gap['midpoint']:.6f} rad"
        )
    print(f"  common cut        = {gap_result['common_cut']:.6f} rad")
    plot_common_cut(
        phi_mod, bunch_id, bin_edges, counts_total, gap_result, rng
    )

    print(f"saved -> {OUT_STEP1}")
    print(f"saved -> {OUT_STEP2}")
    print(f"saved -> {OUT_STEP3}")


if __name__ == "__main__":
    main()
