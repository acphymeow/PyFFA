"""生成论文使用的纵向坐标连续化四联图。

这个脚本复用 ``test_core_bin_enclosing_arc.py`` 中当前实际验证的算法，
依次展示：

1. 多圈散点纵向坐标；
2. 在同一 panel 中展示周期 histogram、密度区域、尾部识别、公共切口和束团边界；
3. 展开坐标并确定每个束团的主要切片范围；
4. 将范围外粒子保留在头、尾边缘切片中。

注意：halo 粒子没有被删除。密度筛选只决定束团长度和常规切片范围。
全环密度 histogram 只建立一次；展开和逐束团 core-bin 选择都直接复用
同一组 counts 和 bin edges。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ================================================================
# CONFIG -- edit here
# ================================================================

HARMONIC_NUMBER = 2
CORE_CHARGE_FRACTION = 0.995
HALO_FRACTION = 0.005
RANDOM_SEED = 20260821
TOTAL_SC_SLICES = 128
CORE_SC_SLICES = 126

SAVE_FIGS = True
SAVE_INDIVIDUAL_STEPS = True
SHOW_FIGS = True

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "demo_core_bin_continuity_4steps_output"


if not SHOW_FIGS:
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import test_core_bin_enclosing_arc as reference


PERIOD = reference.PERIOD
# Muted paper palette.  Adjust these values here if a different tone is
# preferred; all panels use the same shared colors.
BLUE_1 = "#4c72b0"     # balanced deep blue
BLUE_2 = "#7189b8"     # softer secondary blue
GREEN = "#3f8f5f"      # balanced green
RED = "#c44e52"        # brick red
ORANGE = "#d28e45"     # warm orange
GRAY = "#6b6b6b"
LIGHT_GRAY = "#d9d9d9"

# Consistent horizontal-coordinate labels for panels (a)--(c).
PHI_LABEL = r"$\phi$"
PHI_MOD_LABEL = r"$\phi_{\mathrm{mod}}$"
UNWRAPPED_LABEL = r"$\phi_{\mathrm{unwrapped}}$"

# Vertical centers of the two slice bars in panel (d).
SLICE_ROW_Y = (0.70, 0.34)


@dataclass
class BucketView:
    index: int
    left_u: float
    right_u: float
    member: np.ndarray
    counts: np.ndarray
    edges: np.ndarray
    envelope: dict


@dataclass
class DemoState:
    phi_raw: np.ndarray
    phi_mod: np.ndarray
    source_label: np.ndarray
    display_y: np.ndarray
    partition: dict
    common_cut: float
    u: np.ndarray
    unfolded_counts: np.ndarray
    unfolded_edges: np.ndarray
    buckets: list[BucketView]
    shown: np.ndarray


def generate_demo_distribution(rng: np.random.Generator):
    """Generate two unequal bunches plus a small ring-wide halo."""
    if HARMONIC_NUMBER != 2:
        raise ValueError("This visual demo currently uses the h=2 layout")

    # Bunch 1 crosses the original 0/360-degree cut.  Bunch 2 contains a
    # shallow internal density valley so that the continuous-envelope step is
    # visible rather than trivial.
    bunch_1 = rng.normal(4.0, 14.0, 16_000)
    bunch_2 = np.concatenate(
        (
            rng.normal(168.0, 7.0, 5_500),
            rng.normal(193.0, 8.0, 4_500),
        )
    )
    core = np.concatenate((bunch_1, bunch_2)) % PERIOD
    labels = np.concatenate(
        (
            np.zeros(len(bunch_1), dtype=np.int32),
            np.ones(len(bunch_2), dtype=np.int32),
        )
    )

    n_halo = int(round(len(core) * HALO_FRACTION / (1.0 - HALO_FRACTION)))
    halo = rng.uniform(0.0, PERIOD, n_halo)
    phi_mod = np.concatenate((core, halo))
    source_label = np.concatenate(
        (labels, np.full(n_halo, -1, dtype=np.int32))
    )

    # Add integer turns only for the first panel.  The continuity algorithm
    # receives phi_mod, not phi_raw.
    turns = rng.integers(-1, 2, len(phi_mod))
    phi_raw = phi_mod + turns * PERIOD
    display_y = rng.normal(0.0, 0.11, len(phi_mod))
    return phi_raw, phi_mod, source_label, display_y


def build_state() -> DemoState:
    if TOTAL_SC_SLICES != CORE_SC_SLICES + 2:
        raise ValueError("TOTAL_SC_SLICES must equal CORE_SC_SLICES + 2 overflow slices")
    rng = np.random.default_rng(RANDOM_SEED)
    phi_raw, phi_mod, source_label, display_y = generate_demo_distribution(rng)

    partition_raw = reference.find_h_gap_partition(phi_mod, HARMONIC_NUMBER)
    counts_ring = partition_raw["counts"]
    edges_ring = partition_raw["edges"]
    n_bins = len(counts_ring)
    bin_width = float(edges_ring[1] - edges_ring[0])

    # Keep one and only one density histogram.  Snap every cut to an existing
    # full-ring bin edge so that unwrapping is an exact cyclic permutation of
    # the original counts rather than a second histogram operation.
    selected_gaps = [dict(gap) for gap in partition_raw["selected_gaps"]]
    common_edge_index = int(
        np.floor(selected_gaps[0]["midpoint_deg"] / bin_width + 0.5)
    ) % n_bins
    common_cut = common_edge_index * bin_width
    selected_gaps[0]["midpoint_deg"] = float(common_cut)

    internal_edge_indices = []
    for gap in selected_gaps[1:]:
        phi_edge_index = int(
            np.floor(gap["midpoint_deg"] / bin_width + 0.5)
        ) % n_bins
        gap["midpoint_deg"] = float(phi_edge_index * bin_width)
        u_edge_index = (phi_edge_index - common_edge_index) % n_bins
        if u_edge_index == 0:
            raise RuntimeError("an internal bunch boundary coincides with the common cut")
        internal_edge_indices.append(u_edge_index)

    internal_edge_indices = sorted(set(internal_edge_indices))
    if len(internal_edge_indices) != HARMONIC_NUMBER - 1:
        raise RuntimeError("selected gaps do not produce h distinct bunch intervals")

    boundary_bin_indices = [0, *internal_edge_indices, n_bins]
    internal_cuts = [index * bin_width for index in internal_edge_indices]
    boundaries = [index * bin_width for index in boundary_bin_indices]

    partition = dict(partition_raw)
    partition.update(
        {
            "selected_gaps": selected_gaps,
            "common_cut_deg": float(common_cut),
            "internal_cuts_u_deg": internal_cuts,
            "interval_boundaries_u_deg": boundaries,
        }
    )

    u = (phi_mod - common_cut) % PERIOD
    unfolded_counts = np.roll(counts_ring, -common_edge_index)
    unfolded_edges = np.linspace(0.0, PERIOD, n_bins + 1)

    buckets = []
    for index, (left_bin, right_bin) in enumerate(
        zip(boundary_bin_indices[:-1], boundary_bin_indices[1:])
    ):
        left_u = float(unfolded_edges[left_bin])
        right_u = float(unfolded_edges[right_bin])
        if index == len(boundaries) - 2:
            member = (u >= left_u) & (u <= right_u)
        else:
            member = (u >= left_u) & (u < right_u)

        # Reuse the corresponding full-ring bins directly.  The only local
        # operation is slicing the already-built counts/edges arrays.
        counts = unfolded_counts[left_bin:right_bin]
        edges = unfolded_edges[left_bin:right_bin + 1]
        if int(counts.sum()) != int(np.count_nonzero(member)):
            raise RuntimeError("rotated full-ring bins lost particles at a boundary")
        envelope = reference.select_core_bin_envelope(
            counts, edges, CORE_CHARGE_FRACTION
        )
        if envelope is None:
            raise RuntimeError(f"coarse interval {index} contains no particles")
        buckets.append(
            BucketView(index, left_u, right_u, member, counts, edges, envelope)
        )

    shown = rng.choice(len(phi_mod), min(12_000, len(phi_mod)), replace=False)
    return DemoState(
        phi_raw,
        phi_mod,
        source_label,
        display_y,
        partition,
        common_cut,
        u,
        unfolded_counts,
        unfolded_edges,
        buckets,
        shown,
    )


def add_periodic_span(ax, start, length, **kwargs):
    """Shade a circular span, splitting it when it crosses 0/PERIOD."""
    end = start + length
    if end <= PERIOD:
        ax.axvspan(start, end, **kwargs)
    else:
        ax.axvspan(start, PERIOD, **kwargs)
        second = dict(kwargs)
        second.pop("label", None)
        ax.axvspan(0.0, end - PERIOD, **second)


def style_phase_axis(
    ax,
    coordinate=PHI_MOD_LABEL,
    *,
    label_fontsize,
    tick_fontsize,
):
    ax.set_xlim(0.0, PERIOD)
    ax.set_xticks([0.0, 90.0, 180.0, 270.0, 360.0])
    ax.set_xlabel(f"{coordinate} [deg]", fontsize=label_fontsize)
    ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)
    ax.grid(axis="y", alpha=0.18)


def draw_step_1(ax, state: DemoState):
    # Typography for panel 1 -- edit here.
    # title_fontsize = 16
    # label_fontsize = 15
    # tick_fontsize = 14
    # bunch_text_fontsize = 17
    # turn_text_fontsize = 18

    shown = state.shown
    halo = state.source_label[shown] < 0
    colors = np.where(
        state.source_label[shown] == 0,
        BLUE_1,
        np.where(state.source_label[shown] == 1, BLUE_2, GRAY),
    )
    # The first panel only introduces the multi-turn sub-bunch pattern.  Halo
    # appears later, where its effect on the occupied interval is discussed.
    core = ~halo
    ax.scatter(
        state.phi_raw[shown][core],
        state.display_y[shown][core],
        s=4.5,
        c=colors[core],
        alpha=0.25,
        linewidths=0,
        rasterized=True,
    )
    for turn in (-1, 0, 1, 2):
        ax.axvline(turn * PERIOD, color="0.55", lw=1.7, ls="--")
    ax.set_xlim(-PERIOD, 2.0 * PERIOD)
    ax.set_ylim(-0.52, 0.52)
    ax.set_yticks([])
    ax.set_xlabel(f"{PHI_LABEL} [deg]", fontsize=18)
    ax.set_title(
        "(a) Multi-turn coordinates",
        fontsize=14,
    )
    ax.tick_params(axis="both", which="both", labelsize=18)
    ax.annotate(
        "bunch 1",
        xy=(0.0, 0.25),
        xytext=(-100.0, 0.40),
        color=BLUE_1,
        fontsize=18,
        arrowprops=dict(arrowstyle="->", color=BLUE_1, lw=1.3),
    )
    ax.annotate(
        "bunch 2",
        xy=(180.0, 0.18),
        xytext=(100.0, 0.40),
        color=BLUE_2,
        fontsize=18,
        arrowprops=dict(arrowstyle="->", color=BLUE_2, lw=1.3),
    )
    for text_x, text_y, label in (
        (-180.0, -0.47, r"turn $n-1$"),
        (180.0, -0.47, r"turn $n$"),
        (540.0, -0.47, r"turn $n+1$"),
    ):
        ax.text(
            x=text_x,
            y=text_y,
            s=label,
            color="0.20",
            fontsize=18,
            ha="center",
            va="bottom",
        )


def draw_step_2(ax, state: DemoState):
    # Typography for panel 2 -- edit here.
    # title_fontsize = 16
    # label_fontsize = 15
    # tick_fontsize = 14
    # text_fontsize = 13

    counts = state.partition["counts"]
    edges = state.partition["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = 0.92 * (edges[1] - edges[0])
    ax.bar(centers, counts, width=width, color=BLUE_1, edgecolor="none")
    ax.axvline(0.0, color=RED, ls="--", lw=1.0)
    ax.axvline(PERIOD, color=RED, ls="--", lw=1.0)
    ax.set_ylabel("particles / bin", fontsize=18)
    ax.set_title(
        "(b) Periodic density",
        fontsize=18,
    )
    style_phase_axis(
        ax,
        label_fontsize=18,
        tick_fontsize=18,
    )


def _draw_unfolded_bunch_histogram(
    ax,
    state: DemoState,
    *,
    title_fontsize=16,
    label_fontsize=15,
    tick_fontsize=14,
):
    colors = (BLUE_1, BLUE_2)
    ymax = 0
    for bucket, color in zip(state.buckets, colors):
        centers = 0.5 * (bucket.edges[:-1] + bucket.edges[1:])
        width = 0.92 * (bucket.edges[1] - bucket.edges[0])
        ymax = max(ymax, int(bucket.counts.max()))
        ax.bar(
            centers,
            bucket.counts,
            width=width,
            color=color,
            edgecolor="none",
            alpha=0.86,
            label=f"coarse bunch interval {bucket.index + 1}",
        )
        ax.axvspan(bucket.left_u, bucket.right_u, color=color, alpha=0.035, lw=0)

    ax.axvline(0.0, color=RED, ls="--", lw=1.2)
    ax.axvline(PERIOD, color=RED, ls="--", lw=1.2)
    for cut in state.partition["internal_cuts_u_deg"]:
        ax.axvline(cut, color=ORANGE, ls="--", lw=1.3)
    ax.set_ylabel("particles / bin", fontsize=label_fontsize)
    ax.set_title(
        "4. Unwrap by rotating the same full-ring bins",
        fontsize=title_fontsize,
    )
    style_phase_axis(
        ax,
        coordinate=UNWRAPPED_LABEL,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
    )
    return ymax


def draw_step_3_combined(ax, state: DemoState):
    """Distinguish continuous bunch regions from sparse tails."""
    edges = state.partition["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = state.partition["density"]
    threshold = state.partition["threshold"]

    ax.plot(centers, density, color=BLUE_1, lw=1.6)
    ax.axhline(threshold, color=RED, lw=1.0, ls=":")

    for gap in state.partition["candidate_gaps"]:
        add_periodic_span(
            ax,
            gap["start_deg"],
            gap["length_deg"],
            color=LIGHT_GRAY,
            alpha=0.34,
            lw=0,
        )

    ymax = density.max()
    ax.annotate(
        "bunch density",
        xy=(176.0, 0.72 * ymax),
        xytext=(92.0, 0.90 * ymax),
        color=BLUE_1,
        fontsize=17,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=BLUE_1, lw=1.4),
    )
    ax.annotate(
        "tails / halo",
        xy=(145.0, max(0.9 * threshold, 0.5)),
        xytext=(88.0, 0.36 * ymax),
        color=GRAY,
        fontsize=16,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.3),
    )

    ax.set_ylabel("smoothed particles / bin", fontsize=18)
    ax.set_title(
        "(c) Density-based bunch regions",
        fontsize=16,
    )
    style_phase_axis(
        ax,
        label_fontsize=18,
        tick_fontsize=18,
    )


def draw_step_4_unfold(ax, state: DemoState):
    """Select the common cut and the remaining bunch boundaries."""
    edges = state.partition["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = state.partition["density"]
    threshold = state.partition["threshold"]
    ax.plot(centers, density, color=BLUE_1, lw=1.6)
    ax.axhline(threshold, color=RED, lw=1.0, ls=":")

    for index, gap in enumerate(state.partition["selected_gaps"]):
        is_common = index == 0
        color = RED if is_common else ORANGE
        add_periodic_span(
            ax,
            gap["start_deg"],
            gap["length_deg"],
            color=color,
            alpha=0.14,
            lw=0,
        )
        ax.axvline(gap["midpoint_deg"], color=color, ls="--", lw=1.5)

    ymax = density.max()
    ax.text(
        state.common_cut,
        0.46 * ymax,
        "common cut",
        color=RED,
        fontsize=18,
        ha="center",
    )
    if len(state.partition["selected_gaps"]) > 1:
        boundary_x = (
            state.common_cut + state.partition["internal_cuts_u_deg"][0]
        ) % PERIOD
        ax.text(
            boundary_x,
            0.46 * ymax,
            "bunch boundary",
            color=ORANGE,
            fontsize=18,
            ha="center",
        )

    ax.set_ylabel("smoothed particles / bin", fontsize=18)
    ax.set_title(
        r"(d) Common cut and bunch boundaries",
        fontsize=16,
    )
    style_phase_axis(
        ax,
        label_fontsize=18,
        tick_fontsize=18,
    )


def draw_steps_2_to_4_combined(ax, state: DemoState):
    """Show the one-turn density, halo-robust regions, and periodic cuts."""
    counts = state.partition["counts"]
    edges = state.partition["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = 0.92 * (edges[1] - edges[0])
    density = state.partition["density"]
    threshold = state.partition["threshold"]

    ax.bar(
        centers,
        counts,
        width=width,
        color=BLUE_1,
        edgecolor="none",
        alpha=0.20,
        zorder=1,
    )
    ax.plot(centers, density, color=BLUE_1, lw=1.8, zorder=3)
    ax.axhline(threshold, color=RED, lw=1.0, ls=":", zorder=2)

    for gap in state.partition["candidate_gaps"]:
        add_periodic_span(
            ax,
            gap["start_deg"],
            gap["length_deg"],
            color=LIGHT_GRAY,
            alpha=0.22,
            lw=0,
            zorder=0,
        )

    for index, gap in enumerate(state.partition["selected_gaps"]):
        is_common = index == 0
        color = RED if is_common else ORANGE
        add_periodic_span(
            ax,
            gap["start_deg"],
            gap["length_deg"],
            color=color,
            alpha=0.12,
            lw=0,
            zorder=0,
        )
        ax.axvline(
            gap["midpoint_deg"],
            color=color,
            ls="--",
            lw=1.5,
            zorder=4,
        )

    ymax = max(float(np.max(counts)), float(np.max(density)))
    ax.annotate(
        "bunch density",
        xy=(176.0, 0.70 * ymax),
        xytext=(152.0, 0.90 * ymax),
        color=BLUE_1,
        fontsize=16,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=BLUE_1, lw=1.3),
    )
    ax.annotate(
        "tails / halo",
        xy=(145.0, max(0.9 * threshold, 0.5)),
        xytext=(112.0, 0.31 * ymax),
        color=GRAY,
        fontsize=15,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2),
    )
    ax.text(
        state.common_cut,
        0.49 * ymax,
        "common cut",
        color=RED,
        fontsize=16,
        ha="center",
    )
    if len(state.partition["selected_gaps"]) > 1:
        boundary_x = (
            state.common_cut + state.partition["internal_cuts_u_deg"][0]
        ) % PERIOD
        ax.text(
            boundary_x,
            0.49 * ymax,
            "bunch boundary",
            color=ORANGE,
            fontsize=16,
            ha="center",
        )

    ax.set_ylabel("particles / bin", fontsize=18)
    ax.set_title(
        "(b) Periodic density and bunch boundaries",
        fontsize=16,
    )
    style_phase_axis(
        ax,
        label_fontsize=18,
        tick_fontsize=18,
    )


def draw_step_5_core_envelopes(ax, state: DemoState):
    """Unwrap the bunches and show their main slicing ranges."""
    colors = (BLUE_1, BLUE_2)
    histogram_peak = _draw_unfolded_bunch_histogram(
        ax,
        state,
        title_fontsize=14,
        label_fontsize=18,
        tick_fontsize=18,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    for bucket, color in zip(state.buckets, colors):
        centers = 0.5 * (bucket.edges[:-1] + bucket.edges[1:])
        left = bucket.envelope["left"]
        right = bucket.envelope["right"]
        y_line = 0.90 * histogram_peak
        ax.axvspan(left, right, color=GREEN, alpha=0.08, lw=0)
        ax.plot([left, right], [y_line, y_line], color=GREEN, lw=4)
        ax.text(
            x=left + 0.50 * (right - left),
            y=0.94 * histogram_peak,
            s=f"main range {bucket.index + 1}",
            color=GREEN,
            fontsize=16,
            ha="center",
        )

        outside = (bucket.counts > 0) & ((centers < left) | (centers > right))
        if np.any(outside):
            outside_centers = centers[outside]
            outside_counts = bucket.counts[outside]
            target_index = int(np.argmax(outside_counts))
            target_x = float(outside_centers[target_index])
            target_y = float(max(outside_counts[target_index], 1))
            ax.annotate(
                "tails",
                xy=(target_x, target_y),
                xytext=(target_x, 0.38 * histogram_peak),
                color=GRAY,
                fontsize=14,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2),
            )

    ax.set_xlim(-8.0, PERIOD + 8.0)
    ax.set_ylim(0.0, 1.12 * histogram_peak)
    ax.set_title(
        "(c) Unwrapped bunch intervals",
        fontsize=16,
    )


def draw_step_6_slices(ax, state: DemoState):
    """Show regular slices plus two edge slices for every bunch."""
    # Typography for panel 6 -- edit here.
    title_fontsize = 16
    text_fontsize = 13

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    left, core_left, core_right, right = 0.08, 0.18, 0.82, 0.92
    height = 0.16
    rows = SLICE_ROW_Y
    for index, y in enumerate(rows, start=1):
        ax.add_patch(
            Rectangle(
                (left, y - height / 2),
                core_left - left,
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
                facecolor=BLUE_1,
                edgecolor="none",
                alpha=0.20,
            )
        )
        ax.add_patch(
            Rectangle(
                (core_right, y - height / 2),
                right - core_right,
                height,
                facecolor=GRAY,
                edgecolor="none",
                alpha=0.78,
            )
        )
        # Representative boundaries; the central label gives the actual count.
        for x in np.linspace(core_left, core_right, 25):
            ax.plot([x, x], [y - height / 2, y + height / 2], color=BLUE_1, lw=0.55)
        ax.text(
            x=0.055,
            y=y,
            s=f"bunch {index}",
            ha="right",
            va="center",
            fontsize=text_fontsize,
        )

    ax.text(
        x=0.13,
        y=0.84,
        s="head edge",
        color="0.20",
        ha="center",
        fontsize=text_fontsize,
    )
    ax.text(
        x=0.50,
        y=0.84,
        s="regular slices",
        color=BLUE_1,
        ha="center",
        fontsize=text_fontsize,
    )
    ax.text(
        x=0.87,
        y=0.84,
        s="tail edge",
        color="0.20",
        ha="center",
        fontsize=text_fontsize,
    )
    ax.set_title(
        "(d) Regular and edge slices",
        fontsize=title_fontsize,
    )


STEP_DRAWERS = (
    draw_step_1,
    draw_steps_2_to_4_combined,
    draw_step_5_core_envelopes,
    draw_step_6_slices,
)


def save_individual_steps(state: DemoState):
    paths = []
    for index, drawer in enumerate(STEP_DRAWERS, start=1):
        fig, ax = plt.subplots(figsize=(11.5, 5.3), constrained_layout=True)
        drawer(ax, state)
        path = OUTPUT_DIR / f"step_{index:02d}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
        plt.close(fig)
    return paths


def save_overview(state: DemoState):
    fig, axes = plt.subplots(2, 2, figsize=(17.0, 8.5))
    for ax, drawer in zip(axes.flat, STEP_DRAWERS):
        drawer(ax, state)
    fig.tight_layout()
    path = OUTPUT_DIR / "core_bin_continuity_four_steps.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(
        OUTPUT_DIR / "core_bin_continuity_four_steps.pdf",
        bbox_inches="tight",
    )
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)
    return path


def print_summary(state: DemoState, overview_path: Path, step_paths):
    print("Core-bin longitudinal continuity demo")
    print("=====================================")
    print(f"particles        = {len(state.phi_mod)}")
    print(f"harmonic h       = {HARMONIC_NUMBER}")
    print(f"density bins     = {len(state.unfolded_counts)} (built once and reused)")
    print(f"SC slices/bunch  = {TOTAL_SC_SLICES} = 1 + {CORE_SC_SLICES} + 1")
    print(f"common cut       = {state.common_cut:.6f} deg")
    print(
        "selected gaps    = "
        + str(
            [
                round(gap["length_deg"], 6)
                for gap in state.partition["selected_gaps"]
            ]
        )
    )
    for bucket in state.buckets:
        envelope = bucket.envelope
        print(
            f"bunch {bucket.index + 1}: "
            f"interval=[{bucket.left_u:.3f}, {bucket.right_u:.3f}] deg, "
            f"L_core={envelope['length']:.3f} deg, "
            f"selected charge={100*envelope['selected_charge_fraction']:.3f}%"
        )
    print(f"overview         = {overview_path}")
    if step_paths:
        print(f"individual steps = {OUTPUT_DIR / 'step_01.png'} ... step_04.png")


def main():
    state = build_state()
    if SAVE_FIGS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        step_paths = (
            save_individual_steps(state) if SAVE_INDIVIDUAL_STEPS else []
        )
        overview_path = save_overview(state)
    else:
        step_paths = []
        overview_path = Path("not saved")
        if SHOW_FIGS:
            for drawer in STEP_DRAWERS:
                fig, ax = plt.subplots(figsize=(10.0, 4.7), constrained_layout=True)
                drawer(ax, state)
                plt.show()
                plt.close(fig)
    print_summary(state, overview_path, step_paths)


if __name__ == "__main__":
    main()
