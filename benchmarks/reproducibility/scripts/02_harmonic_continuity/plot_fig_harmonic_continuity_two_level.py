"""Two-level paper schematic for harmonic longitudinal continuity.

Figure 1 gives the global route from a multi-turn h=2 distribution, through
projection onto the ring and selection of one common cut, to continuous bunch
coordinates.  Figure 2 expands the core-bin selection, continuous envelope,
and treatment of particles outside that envelope.

The numerical distribution is synthetic.  It deliberately contains a bunch
crossing the original periodic boundary, a second bunch with an internal
density valley, and a small ring-wide halo.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

import plot_fig_longitudinal_interval_unified as base


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

OUT_OVERVIEW_PNG = HERE / "fig_harmonic_continuity_overview.png"
OUT_OVERVIEW_PDF = HERE / "fig_harmonic_continuity_overview.pdf"
OUT_DETAILS_PNG = HERE / "fig_core_bin_continuity_details.png"
OUT_DETAILS_PDF = HERE / "fig_core_bin_continuity_details.pdf"

SAVE_FIGS = True
SHOW_FIGS = False

BLUE = base.BLUE
BLUE_LIGHT = base.BLUE_LIGHT
TEAL = base.TEAL
GREEN = base.GREEN
RED = base.RED
ORANGE = base.ORANGE
GRAY = base.GRAY
LIGHT_GRAY = base.LIGHT_GRAY
VERY_LIGHT_GRAY = base.VERY_LIGHT_GRAY


def style_linear_axis(ax, xlabel, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="in")
    ax.grid(axis="y", alpha=0.15)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def add_panel_label(ax, label):
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
    )


def sampled_particles():
    bunch_1, bunch_2, halo = base.generate_distribution()
    rng = np.random.default_rng(base.SEED + 17)

    def sample(values, n):
        if len(values) <= n:
            return values
        return rng.choice(values, n, replace=False)

    return sample(bunch_1, 1700), sample(bunch_2, 1300), sample(halo, 100), rng


# ---------------------------------------------------------------------------
# Figure 1: global route
# ---------------------------------------------------------------------------


def overview_multiturn(ax):
    bunch_1, bunch_2, halo, rng = sampled_particles()
    # Replication is only a schematic device to display the periodic sub-bunch
    # pattern over three consecutive turns.
    for turn in (-1, 0, 1):
        for values, color, size, alpha in (
            (bunch_1, BLUE, 4.0, 0.16),
            (bunch_2, TEAL, 4.0, 0.18),
            (halo, GRAY, 10.0, 0.65),
        ):
            jitter = rng.normal(0.0, 0.105, len(values))
            ax.scatter(
                values + turn * base.PERIOD,
                turn + jitter,
                s=size,
                color=color,
                alpha=alpha,
                linewidths=0,
                rasterized=True,
            )
    for edge in (-base.PERIOD, 0.0, base.PERIOD, 2.0 * base.PERIOD):
        ax.axvline(edge, color="0.55", lw=0.8, ls="--")
    ax.set_xlim(-base.PERIOD, 2.0 * base.PERIOD)
    ax.set_ylim(-1.38, 1.38)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([r"turn $n-1$", r"turn $n$", r"turn $n+1$"])
    ax.set_xticks([-360, 0, 360, 720])
    style_linear_axis(ax, r"multi-turn longitudinal coordinate $s$ [deg]")
    ax.set_title(r"Multi-turn sub-bunches ($h=2$)", fontsize=10)
    add_panel_label(ax, "(a)")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
                   markeredgecolor="none", label="sub-bunch 1"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL,
                   markeredgecolor="none", label="sub-bunch 2"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY,
                   markeredgecolor="none", label="low-density halo"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper center",
        ncol=3,
    )


def configure_ring(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.16)
    ax.set_yticks([])
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels([r"$0^\circ$", r"$90^\circ$", r"$180^\circ$", r"$270^\circ$"])
    ax.grid(color="0.85", lw=0.7)
    ax.spines["polar"].set_color("0.55")
    ax.spines["polar"].set_linewidth(0.8)


def overview_project_ring(ax, state):
    configure_ring(ax)
    theta = np.deg2rad(state["centers"])
    bin_width = np.deg2rad(state["width"] * 0.92)
    total = np.maximum(state["total_counts"], 1)
    norm = total / total.max()
    dominant_1 = state["counts_1"] >= state["counts_2"]
    colors = np.where(dominant_1, BLUE, TEAL)
    ax.bar(
        theta,
        0.16 + 0.28 * norm,
        width=bin_width,
        bottom=0.70,
        color=colors,
        edgecolor="none",
        alpha=0.88,
    )
    halo_bins = state["counts_halo"] > 0
    ax.scatter(
        theta[halo_bins],
        np.full(np.count_nonzero(halo_bins), 0.61),
        s=7,
        marker="s",
        color=GRAY,
        alpha=0.70,
        linewidths=0,
    )
    ax.plot([0.0, 0.0], [0.43, 1.10], color=RED, ls="--", lw=1.4)
    ax.text(np.deg2rad(8), 0.48, "original cut", color=RED, fontsize=8)
    ax.text(np.deg2rad(352), 0.90, "1", color=BLUE, fontsize=10, ha="right")
    ax.text(np.deg2rad(181), 0.92, "2", color=TEAL, fontsize=10, ha="center")
    ax.set_title(r"Project onto one periodic ring: $s_{\rm mod}=s\ \mathrm{mod}\ C$", fontsize=10)
    add_panel_label(ax, "(b)")


def overview_common_cut_ring(ax, state):
    configure_ring(ax)
    theta = np.deg2rad(state["centers"])
    density = state["density"] / state["density"].max()
    radius = 0.70 + 0.30 * density
    closed_theta = np.append(theta, theta[0] + 2.0 * np.pi)
    closed_radius = np.append(radius, radius[0])
    ax.plot(closed_theta, closed_radius, color="0.35", lw=1.4)
    ax.fill_between(closed_theta, 0.70, closed_radius, color=LIGHT_GRAY, alpha=0.55)

    for index, gap in enumerate(state["gaps"]):
        midpoint = np.deg2rad(gap["midpoint"])
        width = np.deg2rad(gap["length"])
        color = RED if index == 0 else ORANGE
        ax.bar(
            midpoint,
            0.48,
            width=width,
            bottom=0.22,
            color=color,
            alpha=0.16,
            edgecolor="none",
            align="center",
        )
        cut_deg = state["common_cut"] if index == 0 else (
            state["common_cut"] + state["boundary_u"]
        ) % base.PERIOD
        cut = np.deg2rad(cut_deg)
        ax.plot([cut, cut], [0.18, 1.08], color=color, ls="--", lw=1.5)

    ax.text(np.deg2rad(state["common_cut"] + 7), 0.31, "common cut", color=RED, fontsize=8)
    other = (state["common_cut"] + state["boundary_u"]) % base.PERIOD
    ax.text(np.deg2rad(other + 7), 0.31, "separator", color=ORANGE, fontsize=8)
    ax.text(np.deg2rad(3), 0.86, "bunch 1", color=BLUE, fontsize=8, ha="center")
    ax.text(np.deg2rad(181), 0.87, "bunch 2", color=TEAL, fontsize=8, ha="center")
    ax.set_title(r"Choose the $h$ gaps and one common cut", fontsize=10)
    add_panel_label(ax, "(c)")


def overview_final(ax, state):
    centers = state["u_centers"]
    width = 0.90 * state["width"]
    rows = (
        (state["region_1"], state["envelope_1"], BLUE, 1.25, "bunch 1"),
        (state["region_2"], state["envelope_2"], TEAL, 0.00, "bunch 2"),
    )
    for region, envelope, color, baseline, label in rows:
        counts = np.where(region, state["u_counts"], 0)
        heights = counts / max(1, counts.max())
        ax.bar(
            centers[region],
            heights[region],
            width=width,
            bottom=baseline,
            color=color,
            edgecolor="none",
            alpha=0.88,
        )
        ax.plot(
            [envelope[0], envelope[1]],
            [baseline + 1.07, baseline + 1.07],
            color=GREEN,
            lw=3.5,
        )
        ax.text(
            0.5 * sum(envelope),
            baseline + 1.13,
            "continuous core interval",
            color=GREEN,
            fontsize=8,
            ha="center",
        )
        ax.axhline(baseline, color="0.6", lw=0.6)
    ax.axvline(0, color=RED, ls="--", lw=1.2)
    ax.axvline(base.PERIOD, color=RED, ls="--", lw=1.2)
    ax.axvline(state["boundary_u"], color=ORANGE, ls="--", lw=1.2)
    ax.set_xlim(0, base.PERIOD)
    ax.set_ylim(-0.05, 2.48)
    ax.set_yticks([0.45, 1.70])
    ax.set_yticklabels(["bunch 2", "bunch 1"])
    ax.set_xticks([0, 90, 180, 270, 360])
    style_linear_axis(ax, r"common unfolded coordinate $u=(s_{\rm mod}-s_{\rm cut})\ \mathrm{mod}\ C$ [deg]")
    ax.set_title("Unfold all sub-bunches in one coordinate", fontsize=10)
    add_panel_label(ax, "(d)")


def make_overview(state):
    fig = plt.figure(figsize=(11.4, 8.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.18, 1.0), hspace=0.34, wspace=0.24)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], projection="polar")
    ax_c = fig.add_subplot(grid[1, 0], projection="polar")
    ax_d = fig.add_subplot(grid[1, 1])
    overview_multiturn(ax_a)
    overview_project_ring(ax_b, state)
    overview_common_cut_ring(ax_c, state)
    overview_final(ax_d, state)
    fig.suptitle("From multi-turn sub-bunches to continuous longitudinal coordinates", fontsize=13, y=0.975)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.07, top=0.90)
    if SAVE_FIGS:
        fig.savefig(OUT_OVERVIEW_PNG, dpi=300, bbox_inches="tight")
        fig.savefig(OUT_OVERVIEW_PDF, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: core-bin and halo details
# ---------------------------------------------------------------------------


def details_reuse_bins(ax, state):
    centers = state["u_centers"]
    width = 0.90 * state["width"]
    region_1 = state["region_1"]
    region_2 = state["region_2"]
    ax.bar(centers[region_1], state["u_counts"][region_1], width=width,
           color=BLUE, edgecolor="none", alpha=0.88)
    ax.bar(centers[region_2], state["u_counts"][region_2], width=width,
           color=TEAL, edgecolor="none", alpha=0.88)
    ax.axvline(state["boundary_u"], color=ORANGE, ls="--", lw=1.3)
    ax.text(
        0.5,
        0.91,
        "roll/reindex the existing full-ring bins — no second histogram",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
    )
    ax.text(state["boundary_u"] - 4, 0.76 * ax.get_ylim()[1], "bunch boundary",
            color=ORANGE, fontsize=8, ha="right")
    ax.set_xlim(0, base.PERIOD)
    ax.set_xticks([0, 90, 180, 270, 360])
    style_linear_axis(ax, r"common unfolded coordinate $u$ [deg]", "particles / bin")
    ax.set_title("Reuse the full-ring bins (no rebinning)", fontsize=10)
    add_panel_label(ax, "(a)")


def details_rank_selection(ax, state):
    # Use bunch 2 because its internal valley makes density ranking visibly
    # distinct from a simple contiguous-edge threshold.
    region = state["region_2"]
    candidates = np.flatnonzero(region & (state["u_counts"] > 0))
    order = candidates[
        np.lexsort((-state["u_counts"][candidates], -state["u_density"][candidates]))
    ]
    charge = state["u_counts"][order]
    cumulative = np.cumsum(charge) / state["u_counts"][region].sum()
    stop = int(np.searchsorted(cumulative, base.CORE_CHARGE_FRACTION, side="left")) + 1
    ranks = np.arange(1, len(order) + 1)
    colors = np.where(ranks <= stop, TEAL, GRAY)
    ax.bar(ranks, charge, width=0.88, color=colors, edgecolor="none")
    ax.axvline(stop + 0.5, color=GREEN, ls="--", lw=1.2)
    ax.set_xlim(0.5, len(order) + 0.5)
    style_linear_axis(ax, "bins ranked by decreasing smoothed density", "charge in bin")
    ax.set_title(r"Select bins until cumulative charge reaches $qQ_k$", fontsize=10)
    add_panel_label(ax, "(b)")

    twin = ax.twinx()
    twin.plot(ranks, cumulative, color=GREEN, lw=1.6)
    twin.axhline(base.CORE_CHARGE_FRACTION, color=GREEN, ls=":", lw=1.1)
    twin.set_ylim(0, 1.04)
    twin.set_ylabel("cumulative charge fraction", color=GREEN)
    twin.tick_params(axis="y", colors=GREEN, direction="in")
    twin.spines["top"].set_visible(False)
    twin.text(
        max(1, stop - 2),
        base.CORE_CHARGE_FRACTION - 0.08,
        rf"$q={base.CORE_CHARGE_FRACTION:.3f}$",
        color=GREEN,
        fontsize=8,
        ha="right",
    )


def details_map_and_enclose(ax, state):
    centers = state["u_centers"]
    width = 0.90 * state["width"]
    rows = (
        (state["region_1"], state["selected_1"], state["envelope_1"], BLUE, 1.25, "bunch 1"),
        (state["region_2"], state["selected_2"], state["envelope_2"], TEAL, 0.00, "bunch 2"),
    )
    for region, selected, envelope, color, baseline, label in rows:
        counts = np.where(region, state["u_counts"], 0)
        normalized = counts / max(1, counts.max())
        selected_mask = np.zeros_like(region, dtype=bool)
        selected_mask[selected] = True
        rejected = region & (counts > 0) & ~selected_mask
        ax.bar(centers[rejected], normalized[rejected], width=width, bottom=baseline,
               color=GRAY, alpha=0.72, edgecolor="none")
        ax.bar(centers[selected_mask], normalized[selected_mask], width=width,
               bottom=baseline, color=color, edgecolor="none")
        ax.plot([envelope[0], envelope[1]], [baseline + 1.08, baseline + 1.08],
                color=GREEN, lw=3.5)
        ax.text(0.5 * sum(envelope), baseline + 1.14,
                rf"$L_{{\rm core}}={envelope[1]-envelope[0]:.0f}^\circ$",
                color=GREEN, fontsize=8, ha="center")
        ax.axhline(baseline, color="0.58", lw=0.6)
    ax.set_xlim(0, base.PERIOD)
    ax.set_ylim(-0.05, 2.45)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0.45, 1.70])
    ax.set_yticklabels(["bunch 2", "bunch 1"])
    style_linear_axis(ax, r"same full-ring bins in $u$ [deg]")
    ax.set_title("Map selected bins back and form a continuous envelope", fontsize=10)
    add_panel_label(ax, "(c)")
    ax.legend(
        handles=[
            Patch(color=BLUE, label="selected core-density bins"),
            Patch(color=GRAY, label="low-density/halo bins"),
            Line2D([0], [0], color=GREEN, lw=3.5, label="continuous core envelope"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )


def details_slice_treatment(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Retain halo charge in two overflow slices", fontsize=10)
    add_panel_label(ax, "(d)")

    y = 0.68
    x0, x1, x2, x3 = 0.07, 0.17, 0.83, 0.93
    height = 0.17
    ax.add_patch(Rectangle((x0, y - height / 2), x1 - x0, height,
                           facecolor=GRAY, edgecolor="none", alpha=0.82))
    ax.add_patch(Rectangle((x1, y - height / 2), x2 - x1, height,
                           facecolor=BLUE_LIGHT, edgecolor="none", alpha=0.38))
    ax.add_patch(Rectangle((x2, y - height / 2), x3 - x2, height,
                           facecolor=GRAY, edgecolor="none", alpha=0.82))
    for x in np.linspace(x1, x2, 27):
        ax.plot([x, x], [y - height / 2, y + height / 2], color=BLUE, lw=0.55)

    ax.text(0.5 * (x0 + x1), 0.82, "left halo", color="0.30", fontsize=8, ha="center")
    ax.text(0.5 * (x1 + x2), 0.82, "core interval", color=BLUE, fontsize=8, ha="center")
    ax.text(0.5 * (x2 + x3), 0.82, "right halo", color="0.30", fontsize=8, ha="center")
    ax.text(0.5 * (x0 + x1), y, "1", color="white", fontsize=9, ha="center", va="center")
    ax.text(0.5 * (x1 + x2), y, "126 uniform slices", color=BLUE, fontsize=8,
            ha="center", va="center")
    ax.text(0.5 * (x2 + x3), y, "1", color="white", fontsize=9, ha="center", va="center")
    ax.text(0.50, 0.53, "128 longitudinal slices for each bunch", fontsize=9, ha="center")

    ax.annotate("", xy=(0.50, 0.40), xytext=(0.50, 0.50),
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))
    ax.text(0.50, 0.34, "all slices retain deposited charge and receive interpolated fields",
            fontsize=8.5, ha="center", color="0.25")
    ax.text(0.50, 0.22, "transverse SC: one 2D solve in each slice",
            fontsize=8.5, ha="center", color=BLUE)
    ax.text(0.50, 0.12, "longitudinal SC: solve each bunch independently over its slices",
            fontsize=8.5, ha="center", color=GREEN)


def make_details(state):
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0))
    details_reuse_bins(axes[0, 0], state)
    details_rank_selection(axes[0, 1], state)
    details_map_and_enclose(axes[1, 0], state)
    details_slice_treatment(axes[1, 1])
    fig.suptitle("Core-density-bin envelope and halo treatment", fontsize=13, y=0.975)
    fig.tight_layout(w_pad=2.0, h_pad=2.5, rect=(0, 0, 1, 0.93))
    if SAVE_FIGS:
        fig.savefig(OUT_DETAILS_PNG, dpi=300, bbox_inches="tight")
        fig.savefig(OUT_DETAILS_PDF, bbox_inches="tight")
    return fig


def main():
    state = base.build_state()
    fig_overview = make_overview(state)
    fig_details = make_details(state)
    print(f"saved -> {OUT_OVERVIEW_PNG}")
    print(f"saved -> {OUT_OVERVIEW_PDF}")
    print(f"saved -> {OUT_DETAILS_PNG}")
    print(f"saved -> {OUT_DETAILS_PDF}")
    print(f"common cut = {state['common_cut']:.3f} deg")
    print(f"bunch boundary u = {state['boundary_u']:.3f} deg")
    if SHOW_FIGS:
        plt.show()
    plt.close(fig_overview)
    plt.close(fig_details)


if __name__ == "__main__":
    main()
