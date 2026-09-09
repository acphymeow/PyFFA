"""Unified schematic for periodic longitudinal intervals, h>1, and halo.

The figure replaces the older, partly overlapping Fig. 5/Fig. 6 concepts with
one 2x2 workflow:

  (a) build one periodic histogram on the full ring;
  (b) use one common cut and one internal boundary for an h=2 example;
  (c) reuse the same bins to construct one core-density envelope per bunch;
  (d) reserve 126 fine core slices plus two halo overflow slices.

The numerical distributions are synthetic and are used only to make the
algorithmic construction visually explicit.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


# ================================================================
# CONFIG -- edit here
# ================================================================

SEED = 20260821
PERIOD = 360.0
HARMONIC_NUMBER = 2
FULL_RING_BINS = 180
SMOOTH_SIGMA_BINS = 1.5
GAP_DENSITY_FRACTION = 0.03
CORE_CHARGE_FRACTION = 0.995
HALO_FRACTION = 0.005

SHOW_FIGS = False
SAVE_FIGS = True

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "fig_longitudinal_interval_unified.png"
OUT_PDF = HERE / "fig_longitudinal_interval_unified.pdf"


# A restrained paper palette shared by all panels.
BLUE = "#3565f3"
BLUE_LIGHT = "#7898df"
TEAL = "#1c9a91"
GREEN = "#07833d"
RED = "#c73832"
ORANGE = "#d18419"
GRAY = "#969696"
LIGHT_GRAY = "#dedede"
VERY_LIGHT_GRAY = "#f1f1f1"


def smooth_periodic(values, sigma_bins=SMOOTH_SIGMA_BINS):
    half = max(2, int(np.ceil(4.0 * sigma_bins)))
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    out = np.zeros_like(values, dtype=float)
    for shift, weight in zip(range(-half, half + 1), kernel):
        out += weight * np.roll(values, shift)
    return out


def circular_true_runs(mask):
    """Return inclusive circular runs; a wrapping run has end < start."""
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    if np.all(mask):
        return [(0, n - 1)]
    if not np.any(mask):
        return []
    first_false = int(np.flatnonzero(~mask)[0])
    order = (first_false + 1 + np.arange(n)) % n
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


def add_periodic_span(ax, start, length, **kwargs):
    end = start + length
    if end <= PERIOD:
        ax.axvspan(start, end, **kwargs)
    else:
        ax.axvspan(start, PERIOD, **kwargs)
        second = dict(kwargs)
        second.pop("label", None)
        ax.axvspan(0.0, end - PERIOD, **second)


def generate_distribution():
    rng = np.random.default_rng(SEED)
    # Bunch 1 crosses the original 0/360 cut.  Bunch 2 deliberately has an
    # internal density valley to show that the final envelope remains continuous.
    bunch_1 = rng.normal(3.0, 14.0, 16_000) % PERIOD
    bunch_2 = np.concatenate(
        (rng.normal(168.0, 7.0, 5_500), rng.normal(193.0, 8.0, 4_500))
    ) % PERIOD
    n_core = len(bunch_1) + len(bunch_2)
    n_halo = int(round(n_core * HALO_FRACTION / (1.0 - HALO_FRACTION)))
    halo = rng.uniform(0.0, PERIOD, n_halo)
    return bunch_1, bunch_2, halo


def select_gap_geometry(total_counts, edges):
    density = smooth_periodic(total_counts)
    threshold = GAP_DENSITY_FRACTION * density.max()
    low = density <= threshold
    bin_width = edges[1] - edges[0]
    gaps = []
    for start_bin, end_bin in circular_true_runs(low):
        length_bins = (end_bin - start_bin) % len(total_counts) + 1
        if length_bins < 2:
            continue
        start = float(edges[start_bin])
        length = float(length_bins * bin_width)
        midpoint = (start + 0.5 * length) % PERIOD
        gaps.append(
            {
                "start": start,
                "length": length,
                "midpoint": midpoint,
            }
        )
    if len(gaps) < HARMONIC_NUMBER:
        raise RuntimeError("the synthetic distribution did not produce two gaps")
    selected = sorted(gaps, key=lambda item: item["length"], reverse=True)[
        :HARMONIC_NUMBER
    ]

    # Snap the cuts to existing global bin edges so the same full-ring bins can
    # be reused after unfolding without re-binning.
    cut_bin = int(np.round(selected[0]["midpoint"] / bin_width)) % len(total_counts)
    common_cut = float(edges[cut_bin])
    other_bin = int(np.round(selected[1]["midpoint"] / bin_width)) % len(total_counts)
    other_cut = float(edges[other_bin])
    boundary_u = float((other_cut - common_cut) % PERIOD)
    return density, threshold, selected, cut_bin, common_cut, boundary_u


def select_core_bins(counts, score, region_mask, q):
    candidates = np.flatnonzero(region_mask & (counts > 0))
    if candidates.size == 0:
        raise RuntimeError("empty bunch interval")
    # Primary key: decreasing smoothed density; secondary key: raw charge.
    order = candidates[np.lexsort((-counts[candidates], -score[candidates]))]
    target = q * counts[region_mask].sum()
    stop = int(np.searchsorted(np.cumsum(counts[order]), target, side="left")) + 1
    return np.sort(order[:stop])


def build_state():
    bunch_1, bunch_2, halo = generate_distribution()
    edges = np.linspace(0.0, PERIOD, FULL_RING_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    counts_1, _ = np.histogram(bunch_1, bins=edges)
    counts_2, _ = np.histogram(bunch_2, bins=edges)
    counts_halo, _ = np.histogram(halo, bins=edges)
    total_counts = counts_1 + counts_2 + counts_halo

    density, threshold, gaps, cut_bin, common_cut, boundary_u = (
        select_gap_geometry(total_counts, edges)
    )

    # Reuse the exact same global bins by rolling them so the snapped common
    # cut becomes u=0.  No second histogram is constructed.
    roll = -cut_bin
    u_counts_1 = np.roll(counts_1, roll)
    u_counts_2 = np.roll(counts_2, roll)
    u_counts_halo = np.roll(counts_halo, roll)
    u_counts = np.roll(total_counts, roll)
    u_density = smooth_periodic(u_counts)
    u_edges = np.linspace(0.0, PERIOD, FULL_RING_BINS + 1)
    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])

    region_1 = u_centers < boundary_u
    region_2 = ~region_1
    selected_1 = select_core_bins(
        u_counts, u_density, region_1, CORE_CHARGE_FRACTION
    )
    selected_2 = select_core_bins(
        u_counts, u_density, region_2, CORE_CHARGE_FRACTION
    )

    def envelope(selected):
        left = int(selected.min())
        right = int(selected.max())
        return float(u_edges[left]), float(u_edges[right + 1])

    envelope_1 = envelope(selected_1)
    envelope_2 = envelope(selected_2)
    return {
        "edges": edges,
        "centers": centers,
        "width": width,
        "counts_1": counts_1,
        "counts_2": counts_2,
        "counts_halo": counts_halo,
        "total_counts": total_counts,
        "density": density,
        "threshold": threshold,
        "gaps": gaps,
        "common_cut": common_cut,
        "boundary_u": boundary_u,
        "u_edges": u_edges,
        "u_centers": u_centers,
        "u_counts_1": u_counts_1,
        "u_counts_2": u_counts_2,
        "u_counts_halo": u_counts_halo,
        "u_counts": u_counts,
        "u_density": u_density,
        "region_1": region_1,
        "region_2": region_2,
        "selected_1": selected_1,
        "selected_2": selected_2,
        "envelope_1": envelope_1,
        "envelope_2": envelope_2,
    }


def clean_density_axis(ax, coordinate):
    ax.set_xlim(0.0, PERIOD)
    ax.set_xticks([0.0, 90.0, 180.0, 270.0, 360.0])
    ax.set_xlabel(coordinate)
    ax.set_ylabel("particles / bin")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16)
    ax.tick_params(direction="in")


def panel_a(ax, state):
    centers = state["centers"]
    width = 0.90 * state["width"]
    ax.bar(centers, state["counts_1"], width=width, color=BLUE, edgecolor="none")
    ax.bar(centers, state["counts_2"], width=width, color=TEAL, edgecolor="none")
    ax.bar(
        centers,
        state["counts_halo"],
        bottom=state["counts_1"] + state["counts_2"],
        width=width,
        color=GRAY,
        edgecolor="none",
        alpha=0.75,
    )
    halo_bins = state["counts_halo"] > 0
    ax.scatter(
        centers[halo_bins],
        np.full(np.count_nonzero(halo_bins), 12.0),
        s=7,
        marker="s",
        color=GRAY,
        alpha=0.85,
        linewidths=0,
        zorder=4,
    )
    ax.axvline(0.0, color="0.25", ls="--", lw=1.1)
    ax.axvline(PERIOD, color="0.25", ls="--", lw=1.1)
    ymax = max(state["counts_1"].max(), state["counts_2"].max())
    ax.text(
        35.0,
        0.90 * ymax,
        "bunch 1 split\nat the original cut",
        color=BLUE,
        fontsize=8,
        ha="center",
        va="top",
    )
    ax.text(181.0, 0.76 * ymax, "bunch 2", color=TEAL, fontsize=8, ha="center")
    ax.set_title(r"(a) One full-ring histogram for $h=2$")
    clean_density_axis(ax, r"periodic coordinate $s_{\rm mod}$ [deg]")
    ax.legend(
        handles=[
            Patch(color=BLUE, label="bunch 1"),
            Patch(color=TEAL, label="bunch 2"),
            Patch(color=GRAY, label="low-density halo"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper right",
        ncol=3,
    )


def panel_b(ax, state):
    centers = state["u_centers"]
    width = 0.90 * state["width"]
    ax.bar(centers, state["u_counts_1"], width=width, color=BLUE, edgecolor="none")
    ax.bar(centers, state["u_counts_2"], width=width, color=TEAL, edgecolor="none")
    ax.bar(
        centers,
        state["u_counts_halo"],
        bottom=state["u_counts_1"] + state["u_counts_2"],
        width=width,
        color=GRAY,
        edgecolor="none",
        alpha=0.72,
    )
    ax.axvline(0.0, color=RED, ls="--", lw=1.3)
    ax.axvline(PERIOD, color=RED, ls="--", lw=1.3)
    ax.axvline(state["boundary_u"], color=ORANGE, ls="--", lw=1.3)
    ax.text(5.0, ax.get_ylim()[1] * 0.90, "common cut", color=RED, fontsize=8)
    ax.text(
        state["boundary_u"],
        ax.get_ylim()[1] * 0.90,
        "bunch boundary",
        color=ORANGE,
        fontsize=8,
        ha="center",
    )
    ax.set_title("(b) One common cut restores continuity")
    clean_density_axis(ax, r"common unfolded coordinate $u$ [deg]")


def panel_c(ax, state):
    centers = state["u_centers"]
    width = 0.90 * state["width"]
    configurations = (
        (
            state["region_1"],
            state["selected_1"],
            state["envelope_1"],
            BLUE,
            1.25,
            "bunch 1",
        ),
        (
            state["region_2"],
            state["selected_2"],
            state["envelope_2"],
            TEAL,
            0.00,
            "bunch 2",
        ),
    )
    for region, selected, envelope, color, baseline, label in configurations:
        counts = np.where(region, state["u_counts"], 0)
        peak = max(1, counts.max())
        normalized = counts / peak
        selected_mask = np.zeros_like(region, dtype=bool)
        selected_mask[selected] = True
        rejected = region & (counts > 0) & ~selected_mask
        ax.bar(
            centers[rejected],
            normalized[rejected],
            bottom=baseline,
            width=width,
            color=GRAY,
            alpha=0.70,
            edgecolor="none",
        )
        ax.scatter(
            centers[rejected],
            np.full(np.count_nonzero(rejected), baseline + 0.035),
            s=7,
            marker="s",
            color=GRAY,
            alpha=0.88,
            linewidths=0,
            zorder=4,
        )
        ax.bar(
            centers[selected_mask],
            normalized[selected_mask],
            bottom=baseline,
            width=width,
            color=color,
            edgecolor="none",
        )
        ax.axhline(baseline, color="0.55", lw=0.6)
        ax.plot(
            [envelope[0], envelope[1]],
            [baseline + 1.08, baseline + 1.08],
            color=GREEN,
            lw=4,
        )
        ax.text(
            0.5 * (envelope[0] + envelope[1]),
            baseline + 1.14,
            rf"$L_{{\rm core,{1 if label == 'bunch 1' else 2}}}$",
            color=GREEN,
            fontsize=8,
            ha="center",
        )
    ax.set_xlim(0.0, PERIOD)
    ax.set_ylim(-0.05, 2.45)
    ax.set_yticks([0.45, 1.70])
    ax.set_yticklabels(["bunch 2", "bunch 1"])
    ax.set_xticks([0.0, 90.0, 180.0, 270.0, 360.0])
    ax.set_xlabel(r"the same full-ring bins in $u$ [deg]")
    ax.set_title(r"(c) Core-density bins carrying $qQ_k$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.14)
    ax.tick_params(direction="in")
    ax.legend(
        handles=[
            Patch(color=BLUE, label="selected core-density bins"),
            Patch(color=GRAY, label="rejected low-density bins"),
            Line2D([0], [0], color=GREEN, lw=4, label="continuous envelope"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )


def draw_slice_band(ax, y, label):
    left = 0.07
    overflow_width = 0.095
    core_left = left + overflow_width
    core_right = 0.93 - overflow_width
    height = 0.18
    ax.add_patch(
        Rectangle(
            (left, y - height / 2),
            overflow_width,
            height,
            facecolor=GRAY,
            edgecolor="none",
            alpha=0.78,
        )
    )
    ax.add_patch(
        Rectangle(
            (core_left, y - height / 2),
            core_right - core_left,
            height,
            facecolor=BLUE_LIGHT,
            edgecolor="none",
            alpha=0.34,
        )
    )
    ax.add_patch(
        Rectangle(
            (core_right, y - height / 2),
            overflow_width,
            height,
            facecolor=GRAY,
            edgecolor="none",
            alpha=0.78,
        )
    )
    # Representative slice boundaries; the label states the actual count.
    for x in np.linspace(core_left, core_right, 25):
        ax.plot([x, x], [y - height / 2, y + height / 2], color=BLUE, lw=0.55)
    ax.text(0.02, y, label, ha="right", va="center", fontsize=8)
    return left, core_left, core_right, 0.93


def panel_d(ax):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("(d) Fixed 128-slice allocation in every bunch")
    left, core_left, core_right, right = draw_slice_band(ax, 0.67, "bunch $k$")

    ax.text(
        0.5 * (left + core_left),
        0.82,
        "1 left\noverflow",
        color="0.30",
        fontsize=8,
        ha="center",
    )
    ax.text(
        0.5 * (core_left + core_right),
        0.82,
        "126 uniform core slices",
        color=BLUE,
        fontsize=8,
        ha="center",
    )
    ax.text(
        0.5 * (core_right + right),
        0.82,
        "1 right\noverflow",
        color="0.30",
        fontsize=8,
        ha="center",
    )

    # A compact field-treatment summary directly below the slice band.
    ax.annotate(
        "",
        xy=(0.50, 0.49),
        xytext=(0.50, 0.58),
        arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0),
    )
    ax.text(
        0.50,
        0.40,
        "all 128 slices retain charge and receive fields",
        ha="center",
        fontsize=9,
        color="0.20",
    )
    ax.text(
        0.50,
        0.27,
        "transverse: one 2D solve per slice",
        ha="center",
        fontsize=8,
        color=BLUE,
    )
    ax.text(
        0.50,
        0.17,
        "longitudinal: slice charges coupled within each bunch",
        ha="center",
        fontsize=8,
        color=GREEN,
    )
    ax.legend(
        handles=[
            Patch(color=BLUE_LIGHT, alpha=0.5, label="fine core grid"),
            Patch(color=GRAY, label="halo overflow slices"),
        ],
        frameon=False,
        fontsize=8,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.01),
    )


def main():
    state = build_state()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    panel_a(axes[0, 0], state)
    panel_b(axes[0, 1], state)
    panel_c(axes[1, 0], state)
    panel_d(axes[1, 1])
    fig.tight_layout(w_pad=2.0, h_pad=2.2)

    if SAVE_FIGS:
        fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
        fig.savefig(OUT_PDF, bbox_inches="tight")
        print(f"saved -> {OUT_PNG}")
        print(f"saved -> {OUT_PDF}")
        print(f"common cut = {state['common_cut']:.3f} deg")
        print(f"boundary u = {state['boundary_u']:.3f} deg")
        for index, envelope in enumerate(
            (state["envelope_1"], state["envelope_2"]), start=1
        ):
            print(
                f"bunch {index}: core envelope "
                f"[{envelope[0]:.3f}, {envelope[1]:.3f}] deg, "
                f"length={envelope[1]-envelope[0]:.3f} deg"
            )
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
