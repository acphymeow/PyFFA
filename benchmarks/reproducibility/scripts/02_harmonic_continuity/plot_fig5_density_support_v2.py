"""Constructive Fig. 5 schematic based only on density support.

This revision removes the old min/max comparison and directly illustrates:
  (a) periodic density support across the 0/2pi boundary;
  (b) unwrapping one occupied arc into a continuous local coordinate;
  (c) independent coordinates for two populated RF buckets at h=2;
  (d) robust fine-grid support in the presence of longitudinal halo.

All green support arcs are calculated from the plotted synthetic samples by a
periodically smoothed density threshold; no support endpoints are hard-coded.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# CONFIG -- edit here
# ================================================================

SEED = 7
PERIOD_DEG = 360.0
NBINS = 512
SMOOTH_SIGMA_BINS = 2.0
EPS = 1.0e-2
SHOW_FIGS = False
SAVE_PDF = True

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "fig5_density_support_v2.png"
OUT_PDF = HERE / "fig5_density_support_v2.pdf"

C_BUNCH = "#2657ff"
C_BUNCH_2 = "#527bdc"
C_SUPPORT = "#07833d"
C_THRESHOLD = "#e68a00"
C_HALO = "#8a8a8a"
C_RING = "#cccccc"


def smooth_periodic(values, sigma_bins):
    half = max(2, int(np.ceil(4.0 * sigma_bins)))
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    result = np.zeros_like(values, dtype=np.float64)
    for shift, weight in zip(range(-half, half + 1), kernel):
        result += weight * np.roll(values, shift)
    return result


def circular_runs(mask):
    n = len(mask)
    if np.all(mask):
        return [(0, n - 1)]
    if not np.any(mask):
        return []
    invalid = int(np.flatnonzero(~mask)[0])
    order = (invalid + 1 + np.arange(n)) % n
    runs = []
    start = previous = None
    for index in order:
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


def density_support(samples, eps=EPS):
    counts, edges = np.histogram(
        np.mod(samples, PERIOD_DEG), bins=NBINS, range=(0.0, PERIOD_DEG)
    )
    density = smooth_periodic(counts.astype(float), SMOOTH_SIGMA_BINS)
    threshold = eps * density.max()
    valid = density > threshold
    bw = PERIOD_DEG / NBINS
    arcs = []
    for b0, b1 in circular_runs(valid):
        length_bins = (b1 - b0) % NBINS + 1
        start = edges[b0]
        length = length_bins * bw
        arcs.append((float(start), float(length)))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return arcs, centers, density, threshold


def setup_ring(ax):
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    ax.plot(np.cos(theta), np.sin(theta), color=C_RING, lw=1.25)
    ax.set_xlim(-1.46, 1.46)
    ax.set_ylim(-1.46, 1.46)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_bunch(ax, center_deg, sigma_deg, color=C_BUNCH, radius=1.0, lw=7):
    local = np.linspace(-3.2, 3.2, 240) * sigma_deg
    angle = np.deg2rad(center_deg + local)
    density = np.exp(-0.5 * (local / sigma_deg) ** 2) ** 1.35
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    for i in range(len(local) - 1):
        ax.plot(
            x[i:i + 2], y[i:i + 2], color=color,
            alpha=0.04 + 0.92 * density[i], lw=lw,
            solid_capstyle="round", zorder=3,
        )


def draw_support_arc(ax, start_deg, length_deg, radius=1.20, lw=3.0):
    angle = np.deg2rad(np.linspace(start_deg, start_deg + length_deg, 300))
    ax.plot(
        radius * np.cos(angle), radius * np.sin(angle),
        color=C_SUPPORT, lw=lw, solid_capstyle="butt", zorder=4,
    )


def draw_radial_boundary(ax, angle_deg, label=None, label_xy=None):
    angle = np.deg2rad(angle_deg)
    ax.plot(
        [0.82 * np.cos(angle), 1.28 * np.cos(angle)],
        [0.82 * np.sin(angle), 1.28 * np.sin(angle)],
        color="0.15", lw=1.2, ls=(0, (3, 3)), zorder=5,
    )
    if label is not None:
        x, y = label_xy or (1.40 * np.cos(angle), 1.40 * np.sin(angle))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8.2, style="italic")


def draw_halo(ax, samples_deg):
    angle = np.deg2rad(samples_deg)
    ax.scatter(
        np.cos(angle), np.sin(angle), s=17, color=C_HALO,
        edgecolors="none", zorder=5,
    )


def setup_strip(ax, xlim=(0.0, 360.0), local=False):
    ax.set_xlim(*xlim)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, length=2)
    if local:
        ax.set_xlabel("local continuous coordinate", fontsize=8)
    else:
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xticklabels(["0", "", r"$\pi$", "", r"$2\pi$"])


def support_segments(start, length):
    end = start + length
    if end <= PERIOD_DEG:
        return [(start, end)]
    return [(start, PERIOD_DEG), (0.0, end - PERIOD_DEG)]


def plot_periodic_density(ax, samples, color=C_BUNCH, log=False, halo_floor=None):
    arcs, centers, density, threshold = density_support(samples)
    if log:
        floor = max(0.25, 0.2 * threshold)
        ax.semilogy(centers, np.maximum(density, floor), color=color, lw=1.3)
        ax.axhline(threshold, color=C_THRESHOLD, lw=1.1, ls=(0, (4, 2)))
        ax.set_ylim(floor, density.max() * 1.8)
    else:
        ax.fill_between(centers, density, color=color, alpha=0.78, lw=0)
        ax.axhline(threshold, color=C_THRESHOLD, lw=1.1, ls=(0, (4, 2)))
        ax.set_ylim(0.0, density.max() * 1.55)
    if halo_floor is not None:
        ax.scatter(halo_floor, np.full(len(halo_floor), threshold * 0.45),
                   marker="v", s=12, color=C_HALO, edgecolors="none")
    return arcs, density.max(), threshold


def draw_support_segments(ax, arcs, y, lw=3.1):
    for start, length in arcs:
        for x0, x1 in support_segments(start, length):
            ax.plot([x0, x1], [y, y], color=C_SUPPORT, lw=lw,
                    solid_capstyle="butt")


def local_profile(samples, arc):
    start, length = arc
    coordinate = np.mod(samples - start, PERIOD_DEG)
    center = np.median(coordinate[coordinate <= length])
    local = coordinate - center
    keep = np.abs(local) < 0.5 * PERIOD_DEG
    return local[keep], length


def plot_local_density(ax, local, color, label):
    counts, edges = np.histogram(local, bins=100, range=(-85.0, 85.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.fill_between(centers, counts, color=color, alpha=0.80, lw=0)
    ax.set_xlim(-85, 85)
    ax.set_ylim(0, max(1, counts.max()) * 1.40)
    ax.set_yticks([])
    ax.set_xticks([-60, 0, 60])
    ax.set_xticklabels(["", "0", ""])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, length=2)
    ax.set_xlabel(label, fontsize=9)


def main():
    rng = np.random.default_rng(SEED)

    # Shared single bunch crossing the periodic boundary.
    single = rng.normal(0.0, 24.0, 10_000)
    arcs_single, _, _, _ = density_support(single)

    # h=2, two populated RF buckets.
    bunch_1 = rng.normal(38.0, 17.0, 7000)
    bunch_2 = rng.normal(218.0, 20.0, 7000)
    double = np.r_[bunch_1, bunch_2]
    arcs_double, _, _, _ = density_support(double)

    # Core plus a visible diffuse longitudinal halo.
    core_halo = rng.normal(88.0, 23.0, 12_000)
    halo = rng.uniform(0.0, PERIOD_DEG, 240)
    with_halo = np.r_[core_halo, halo]
    arcs_halo, _, _, _ = density_support(with_halo)

    fig = plt.figure(figsize=(7.6, 8.4))
    grid = fig.add_gridspec(
        4, 2, height_ratios=[3.0, 1.16, 3.0, 1.28],
        hspace=0.55, wspace=0.16,
        left=0.07, right=0.97, top=0.94, bottom=0.06,
    )

    # ---------------- (a) periodic density support ----------------
    ax = fig.add_subplot(grid[0, 0])
    setup_ring(ax)
    draw_bunch(ax, 0.0, 24.0)
    for arc in arcs_single:
        draw_support_arc(ax, *arc)
    draw_radial_boundary(
        ax, 0.0, "periodic boundary at $0/2\\pi$", label_xy=(0.0, -1.31)
    )
    ax.set_title("(a) periodic density support", fontsize=10.5, pad=6)

    ax = fig.add_subplot(grid[1, 0])
    setup_strip(ax)
    arcs, ymax, threshold = plot_periodic_density(ax, single)
    draw_support_segments(ax, arcs, ymax * 1.20)
    ax.text(180, ymax * 1.35, "one connected wrap-around support",
            color=C_SUPPORT, fontsize=8, ha="center")
    ax.text(187, threshold * 1.22, r"$\epsilon\lambda_{\rm peak}$",
            color=C_THRESHOLD, fontsize=7.5, va="bottom")

    # ---------------- (b) unwrap one arc ---------------------------
    ax = fig.add_subplot(grid[0, 1])
    setup_ring(ax)
    draw_bunch(ax, 0.0, 24.0)
    for arc in arcs_single:
        draw_support_arc(ax, *arc)
    draw_radial_boundary(
        ax, 180.0, "branch cut\nin empty region", label_xy=(-1.12, -0.43)
    )
    ax.text(
        0.0, -1.31,
        r"$s_1=(s_{\rm mod}-s_{\rm start})\ {\rm mod}\ C$",
        fontsize=8.2, ha="center",
    )
    ax.set_title("(b) continuous coordinate on one arc", fontsize=10.5, pad=6)

    ax = fig.add_subplot(grid[1, 1])
    local_single, _ = local_profile(single, arcs_single[0])
    plot_local_density(ax, local_single, C_BUNCH, r"$s_1$")
    counts, _ = np.histogram(local_single, bins=100, range=(-85, 85))
    y = counts.max() * 1.16
    lo, hi = np.quantile(local_single, [0.001, 0.999])
    ax.plot([lo, hi], [y, y], color=C_SUPPORT, lw=3.1)
    ax.text(0, counts.max() * 1.30, "continuous fine-grid support",
            color=C_SUPPORT, fontsize=8, ha="center")

    # ---------------- (c) h=2, two occupied arcs ------------------
    ax = fig.add_subplot(grid[2, 0])
    setup_ring(ax)
    draw_bunch(ax, 38.0, 17.0, C_BUNCH)
    draw_bunch(ax, 218.0, 20.0, C_BUNCH_2)
    for arc in arcs_double:
        draw_support_arc(ax, *arc)
    ax.text(0.0, 0.0, "two independent\noccupied arcs",
            ha="center", va="center", fontsize=8.5, color=C_SUPPORT)
    ax.set_title(r"(c) two populated RF buckets ($h=2$)", fontsize=10.5, pad=6)

    sub = grid[3, 0].subgridspec(1, 2, wspace=0.20)
    # Match arcs by circular center to the two synthetic bunch centers.
    arc_centers = [np.mod(a + 0.5 * length, 360.0) for a, length in arcs_double]
    i1 = int(np.argmin([abs(np.mod(c - 38 + 180, 360) - 180) for c in arc_centers]))
    i2 = 1 - i1
    local_1, _ = local_profile(bunch_1, arcs_double[i1])
    local_2, _ = local_profile(bunch_2, arcs_double[i2])
    ax1 = fig.add_subplot(sub[0, 0])
    ax2 = fig.add_subplot(sub[0, 1])
    plot_local_density(ax1, local_1, C_BUNCH, r"local coordinate $s_1$")
    plot_local_density(ax2, local_2, C_BUNCH_2, r"local coordinate $s_2$")
    for local, axis in ((local_1, ax1), (local_2, ax2)):
        counts, _ = np.histogram(local, bins=100, range=(-85, 85))
        lo, hi = np.quantile(local, [0.001, 0.999])
        axis.plot([lo, hi], [counts.max() * 1.16] * 2,
                  color=C_SUPPORT, lw=3.0)

    # ---------------- (d) density support with halo ---------------
    ax = fig.add_subplot(grid[2, 1])
    setup_ring(ax)
    draw_bunch(ax, 88.0, 23.0)
    shown_halo = halo[::12]
    draw_halo(ax, shown_halo)
    for arc in arcs_halo:
        draw_support_arc(ax, *arc)
    ax.text(-0.78, -0.83, "halo", color=C_HALO, fontsize=8.5, style="italic")
    ax.set_title("(d) density support with longitudinal halo", fontsize=10.5, pad=6)

    ax = fig.add_subplot(grid[3, 1])
    setup_strip(ax)
    arcs, ymax, threshold = plot_periodic_density(
        ax, with_halo, log=True, halo_floor=shown_halo
    )
    draw_support_segments(ax, arcs, threshold * 2.5)
    ax.text(88, threshold * 3.15, "fine-grid support", color=C_SUPPORT,
            fontsize=8, ha="center")
    ax.text(350, threshold * 12.0, "gray markers: halo/background",
            color=C_HALO, fontsize=7.2, ha="right", va="center", style="italic")
    ax.text(205, threshold * 1.14, r"$\epsilon\lambda_{\rm peak}$",
            color=C_THRESHOLD, fontsize=7.5, va="bottom")

    fig.savefig(OUT_PNG, dpi=220)
    if SAVE_PDF:
        fig.savefig(OUT_PDF)
    print(f"saved -> {OUT_PNG}")
    if SAVE_PDF:
        print(f"saved -> {OUT_PDF}")
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
