"""Paper schematic for periodic continuity, h>1, and halo-robust length.

The figure intentionally contains no comparison with the former particle
min/max construction.  Every green envelope is calculated from density-ranked
histogram bins rather than hand-placed endpoints.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np


SEED = 20260811
HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "fig5_core_bin_method.png"
OUT_PDF = HERE / "fig5_core_bin_method.pdf"

BLUE = "#3264f5"
BLUE2 = "#6d8fdc"
GREEN = "#07833d"
GRAY = "#969696"
LIGHT_GRAY = "#d0d0d0"
CUT = "#c24a3a"
BOUNDARY = "#d18b28"
PERIOD = 360.0


def smooth(values, sigma=1.25):
    half = max(2, int(np.ceil(4 * sigma)))
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values.astype(float), kernel, mode="same")


def select_dense_bins(counts, q):
    """Densest non-empty bins carrying q of charge, plus their envelope."""
    score = smooth(counts)
    candidates = np.flatnonzero(counts > 0)
    order = candidates[np.lexsort((-counts[candidates], -score[candidates]))]
    stop = np.searchsorted(np.cumsum(counts[order]), q * counts.sum(), side="left") + 1
    selected = np.sort(order[:stop])
    return selected, int(selected[0]), int(selected[-1])


def setup_ring(ax):
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(theta), np.sin(theta), color=LIGHT_GRAY, lw=1.4)
    ax.set_xlim(-1.43, 1.43)
    ax.set_ylim(-1.43, 1.43)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_bunch_arc(ax, center_deg, sigma_deg, color=BLUE, radius=1.0, lw=7):
    local = np.linspace(-3.3, 3.3, 260) * sigma_deg
    angle = np.deg2rad(center_deg + local)
    density = np.exp(-0.5 * (local / sigma_deg) ** 2) ** 1.25
    for i in range(len(local) - 1):
        ax.plot(radius * np.cos(angle[i:i + 2]), radius * np.sin(angle[i:i + 2]),
                color=color, alpha=0.035 + 0.94 * density[i], lw=lw,
                solid_capstyle="round", zorder=3)


def draw_arc(ax, start_deg, length_deg, radius=1.19, color=GREEN, lw=3.0):
    a = np.deg2rad(np.linspace(start_deg, start_deg + length_deg, 240))
    ax.plot(radius * np.cos(a), radius * np.sin(a), color=color, lw=lw,
            solid_capstyle="butt", zorder=4)


def radial_mark(ax, angle_deg, color, label=None, label_xy=None, lw=1.3):
    a = np.deg2rad(angle_deg)
    ax.plot([0.80 * np.cos(a), 1.29 * np.cos(a)],
            [0.80 * np.sin(a), 1.29 * np.sin(a)],
            color=color, lw=lw, ls=(0, (4, 3)), zorder=5)
    if label:
        x, y = label_xy or (1.38 * np.cos(a), 1.38 * np.sin(a))
        ax.text(x, y, label, color=color, ha="center", va="center",
                fontsize=8.1, style="italic")


def clean_density_axis(ax):
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, length=2)


def local_hist(samples, bins=90, extent=(-80, 80)):
    counts, edges = np.histogram(samples, bins=bins, range=extent)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts, edges, centers


def plot_local_profile(ax, samples, color, label, q=0.995):
    counts, edges, centers = local_hist(samples)
    selected, left, right = select_dense_bins(counts, q)
    ax.fill_between(centers, counts, color=color, alpha=0.84, lw=0)
    y = counts.max() * 1.17
    ax.plot([edges[left], edges[right + 1]], [y, y], color=GREEN, lw=3.2)
    ax.text(0.5 * (edges[left] + edges[right + 1]), counts.max() * 1.29,
            label, color=GREEN, fontsize=8.2, ha="center")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, counts.max() * 1.43)
    ax.set_xticks([-60, 0, 60])
    ax.set_xticklabels(["", "0", ""])
    clean_density_axis(ax)


def panel_a(container, samples):
    """Project the beam onto one turn; the original boundary splits it."""
    grid = container.subgridspec(2, 1, height_ratios=[2.15, 1.0], hspace=0.05)

    ring = plt.subplot(grid[0])
    setup_ring(ring)
    draw_bunch_arc(ring, 0, 23)
    radial_mark(ring, 0, "0.25", r"$0/2\pi$", label_xy=(1.34, 0.22))
    ring.text(0, 0.0,
              r"$\phi_{\rm mod}=\phi\ {\rm mod}\ 2\pi$", color="0.30",
              ha="center", va="center", fontsize=8.2)
    ring.set_title("(a) one-turn projection",
                   fontsize=10.2, pad=5)

    periodic = np.mod(samples, PERIOD)
    counts, edges = np.histogram(periodic, bins=72, range=(0.0, PERIOD))
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = plt.subplot(grid[1])
    hist.bar(centers, counts, width=0.92 * (edges[1] - edges[0]),
             color=BLUE, edgecolor="none", alpha=0.88)
    hist.text(PERIOD / 2, counts.max() * 1.08,
              r"split at $0/2\pi$", color="0.35",
              fontsize=8.0, ha="center")
    hist.set_xlim(0.0, PERIOD)
    hist.set_ylim(0.0, counts.max() * 1.25)
    hist.set_xticks([0.0, PERIOD])
    hist.set_xticklabels(["0", r"$2\pi$"])
    hist.set_xlabel(r"$\phi_{\rm mod}$", fontsize=8.5, labelpad=1)
    clean_density_axis(hist)


def panel_b(container, samples):
    """Move the branch cut into the empty arc and recover continuity."""
    grid = container.subgridspec(2, 1, height_ratios=[2.15, 1.0], hspace=0.05)

    ring = plt.subplot(grid[0])
    setup_ring(ring)
    draw_bunch_arc(ring, 0, 23)
    radial_mark(ring, 180, CUT, "common cut",
                label_xy=(-1.02, -0.42))
    ring.text(0, -1.18,
              r"$u=(\phi_{\rm mod}-\phi_{\rm cut})\ {\rm mod}\ 2\pi$",
              color="0.30", ha="center", va="center", fontsize=8.0)
    ring.set_title("(b) restore continuity",
                   fontsize=10.2, pad=5)

    periodic = np.mod(samples, PERIOD)
    local = np.mod(periodic - 180.0, PERIOD)
    counts, edges = np.histogram(local, bins=72, range=(0.0, PERIOD))
    centers = 0.5 * (edges[:-1] + edges[1:])
    _, left, right = select_dense_bins(counts, 0.995)
    hist = plt.subplot(grid[1])
    hist.bar(centers, counts, width=0.92 * (edges[1] - edges[0]),
             color=BLUE, edgecolor="none", alpha=0.88)
    y = counts.max() * 1.10
    hist.plot([edges[left], edges[right + 1]], [y, y], color=GREEN, lw=3.0)
    hist.text(0.5 * (edges[left] + edges[right + 1]), counts.max() * 1.19,
              "continuous bunch", color=GREEN, fontsize=8.0,
              ha="center")
    hist.set_xlim(0.0, PERIOD)
    hist.set_ylim(0.0, counts.max() * 1.34)
    hist.set_xticks([0.0, PERIOD])
    hist.set_xticklabels(["0", r"$2\pi$"])
    hist.set_xlabel(r"$u$", fontsize=8.5, labelpad=1)
    clean_density_axis(hist)


def panel_h2(container, rng):
    """Use one common cut and one bunch boundary for the h=2 case."""
    grid = container.subgridspec(2, 1, height_ratios=[2.15, 1.12], hspace=0.08)

    center_1, center_2 = 20.0, 200.0
    common_cut, bunch_boundary = 100.0, 290.0
    samples_1 = rng.normal(0, 15, 9000)
    samples_2 = rng.normal(0, 18, 7000)
    angles_1 = np.mod(center_1 + samples_1, PERIOD)
    angles_2 = np.mod(center_2 + samples_2, PERIOD)
    envelopes = []
    for samples in (samples_1, samples_2):
        counts, edges, _ = local_hist(samples)
        _, left, right = select_dense_bins(counts, 0.995)
        envelopes.append((edges[left], edges[right + 1] - edges[left]))

    ring = plt.subplot(grid[0])
    setup_ring(ring)
    draw_bunch_arc(ring, center_1, 15, BLUE)
    draw_bunch_arc(ring, center_2, 18, BLUE2)
    draw_arc(ring, center_1 + envelopes[0][0], envelopes[0][1])
    draw_arc(ring, center_2 + envelopes[1][0], envelopes[1][1])
    radial_mark(ring, common_cut, CUT, "common cut", label_xy=(-0.44, 1.30))
    radial_mark(ring, bunch_boundary, BOUNDARY, "bunch boundary",
                label_xy=(0.72, -1.28))
    ring.text(0.47, 0.34, "bunch 1", color=BLUE,
              ha="center", va="center", fontsize=8.4)
    ring.text(-0.47, -0.34, "bunch 2", color=BLUE2,
              ha="center", va="center", fontsize=8.4)
    ring.set_title(r"(d) $h=2$",
                   fontsize=10.2, pad=5)

    u_1 = np.mod(angles_1 - common_cut, PERIOD)
    u_2 = np.mod(angles_2 - common_cut, PERIOD)
    edges = np.linspace(0.0, PERIOD, 91)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts_1, _ = np.histogram(u_1, bins=edges)
    counts_2, _ = np.histogram(u_2, bins=edges)
    hist = plt.subplot(grid[1])
    boundary_u = np.mod(bunch_boundary - common_cut, PERIOD)
    hist.axvspan(0.0, boundary_u, color=BLUE2, alpha=0.045, lw=0)
    hist.axvspan(boundary_u, PERIOD, color=BLUE, alpha=0.045, lw=0)
    width = 0.92 * (edges[1] - edges[0])
    hist.bar(centers, counts_1, width=width, color=BLUE,
             edgecolor="none", alpha=0.90)
    hist.bar(centers, counts_2, width=width, color=BLUE2,
             edgecolor="none", alpha=0.90)
    ymax = max(counts_1.max(), counts_2.max())
    hist.axvline(0.0, color=CUT, ls="--", lw=1.2)
    hist.axvline(PERIOD, color=CUT, ls="--", lw=1.2)
    hist.axvline(boundary_u, color=BOUNDARY, ls="--", lw=1.3)
    hist.text(4.0, ymax * 1.06, "common cut", color=CUT,
              fontsize=7.5, ha="left")
    hist.text(boundary_u, ymax * 1.06, "boundary", color=BOUNDARY,
              fontsize=7.5, ha="center")
    for index, (center, envelope, color, label) in enumerate((
            (center_1, envelopes[0], BLUE, r"$L_{{\rm core},1}$"),
            (center_2, envelopes[1], BLUE2, r"$L_{{\rm core},2}$"))):
        start_u = np.mod(center + envelope[0] - common_cut, PERIOD)
        end_u = start_u + envelope[1]
        hist.plot([start_u, end_u], [ymax * 0.86, ymax * 0.86],
                  color=GREEN, lw=3.0)
        hist.text(0.5 * (start_u + end_u), ymax * 0.93, label,
                  color=GREEN, fontsize=7.8, ha="center")
    hist.set_xlim(0.0, PERIOD)
    hist.set_ylim(0.0, ymax * 1.22)
    hist.set_xticks([0.0, PERIOD])
    hist.set_xticklabels(["0", r"$2\pi$"])
    hist.set_xlabel(r"$u$", fontsize=8.5, labelpad=1)
    clean_density_axis(hist)


def panel_d(ax):
    edges = np.linspace(-90, 90, 62)
    x = 0.5 * (edges[:-1] + edges[1:])
    core = 100 * np.exp(-0.5 * (x / 27) ** 2)
    counts = np.rint(core).astype(int)
    # Only a few sparse, remote bins represent longitudinal halo.
    halo_indices = np.array([1, 4, 7, len(x) - 8, len(x) - 5, len(x) - 2])
    counts[halo_indices] += np.array([1, 2, 1, 1, 2, 1])
    selected, left, right = select_dense_bins(counts, 0.985)
    selected_mask = np.zeros_like(counts, dtype=bool)
    selected_mask[selected] = True
    width = edges[1] - edges[0]

    ax.axvspan(edges[left], edges[right + 1], color=GREEN, alpha=0.10, lw=0)
    ax.bar(x[selected_mask], counts[selected_mask], width=0.92 * width,
           color=BLUE, edgecolor="none", label="core-density bins")
    ax.bar(x[~selected_mask], counts[~selected_mask], width=0.92 * width,
           color=GRAY, alpha=0.75, edgecolor="none", label="low-density/halo bins")
    y = counts.max() * 1.15
    ax.plot([edges[left], edges[right + 1]], [y, y], color=GREEN, lw=3.3)
    ax.text(0.5 * (edges[left] + edges[right + 1]), counts.max() * 1.27,
            r"$L_{{\rm core},k}$", color=GREEN,
            fontsize=8.2, ha="center")
    note_box = dict(facecolor="white", alpha=0.76, edgecolor="none", pad=1.0)
    halo_pointer = len(x) - 5
    ax.annotate("halo does not\nset the length",
                xy=(x[halo_pointer], counts[halo_pointer]),
                xytext=(57, counts.max() * 0.61), fontsize=7.4,
                color="0.35", ha="center", bbox=note_box,
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
    ax.text(0, counts.max() * 1.02,
            r"retain $qQ_k$", fontsize=8.0,
            ha="center", va="bottom", bbox=note_box)
    ax.set_xlim(-90, 90)
    ax.set_ylim(0, counts.max() * 1.43)
    ax.set_xticks([-60, 0, 60])
    ax.set_xticklabels(["", "0", ""])
    ax.set_xlabel(r"$s_k$", fontsize=8.5)
    clean_density_axis(ax)
    ax.set_title("(c) halo-robust length",
                 fontsize=10.2, pad=5)


def main():
    rng = np.random.default_rng(SEED)
    single_bunch = rng.normal(0.0, 22.0, 15000)
    fig = plt.figure(figsize=(8.0, 7.8))
    outer = fig.add_gridspec(2, 2, left=0.07, right=0.97, bottom=0.08,
                             top=0.95, wspace=0.20, hspace=0.28,
                             height_ratios=[1.0, 1.08])
    panel_a(outer[0, 0], single_bunch)
    panel_b(outer[0, 1], single_bunch)
    panel_d(fig.add_subplot(outer[1, 0]))
    panel_h2(outer[1, 1], rng)
    fig.savefig(OUT_PNG, dpi=230)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"saved -> {OUT_PNG}")
    print(f"saved -> {OUT_PDF}")


if __name__ == "__main__":
    main()
