"""Restrained paper figures for harmonic longitudinal continuity.

Figure 1: multi-turn sub-bunches -> one periodic ring -> common unfolding.
Figure 2: final occupied intervals and the per-bunch longitudinal slice grids.

The synthetic h=2 distribution is shared with
``plot_fig_longitudinal_interval_unified.py``.
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


HERE = Path(__file__).resolve().parent
FIG1_PNG = HERE / "fig_harmonic_continuity_paper.png"
FIG1_PDF = HERE / "fig_harmonic_continuity_paper.pdf"
FIG2_PNG = HERE / "fig_occupied_interval_and_slicing_paper.png"
FIG2_PDF = HERE / "fig_occupied_interval_and_slicing_paper.pdf"

SAVE_FIGS = True
SHOW_FIGS = False

BLUE = "#3565f3"
GRAY = "#969696"
LIGHT_GRAY = "#e6e6e6"
DARK = "#4d4d4d"
RED = "#c73832"
GREEN = "#07833d"


def panel_label(ax, label):
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )


def style_cartesian(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="in")
    ax.grid(axis="y", color="0.90", lw=0.6)


def sample_particles():
    bunch_1, bunch_2, halo = base.generate_distribution()
    rng = np.random.default_rng(base.SEED + 31)

    def choose(values, n):
        return values if len(values) <= n else rng.choice(values, n, replace=False)

    return choose(bunch_1, 1000), choose(bunch_2, 750), choose(halo, 80), rng


def draw_multiturn(ax):
    bunch_1, bunch_2, halo, rng = sample_particles()
    for turn in (-1, 0, 1):
        for values, color, size, alpha in (
            (bunch_1, BLUE, 3.5, 0.15),
            (bunch_2, BLUE, 3.5, 0.15),
            (halo, GRAY, 8.0, 0.58),
        ):
            ax.scatter(
                values + turn * base.PERIOD,
                turn + rng.normal(0.0, 0.09, len(values)),
                s=size,
                color=color,
                alpha=alpha,
                linewidths=0,
                rasterized=True,
            )
    for edge in (-360, 0, 360, 720):
        ax.axvline(edge, color="0.55", ls="--", lw=0.8)
    ax.set_xlim(-360, 720)
    ax.set_ylim(-1.32, 1.32)
    ax.set_xticks([-360, 0, 360, 720])
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([r"$n-1$", r"$n$", r"$n+1$"])
    ax.set_xlabel(r"multi-turn coordinate $s$ [deg]")
    ax.set_ylabel("turn")
    ax.set_title(r"Multi-turn sub-bunches ($h=2$)", fontsize=10)
    style_cartesian(ax)
    panel_label(ax, "(a)")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
                   markeredgecolor="none", label="beam particles"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY,
                   markeredgecolor="none", label="low-density halo"),
        ],
        frameon=False,
        fontsize=7.5,
        loc="upper center",
        ncol=2,
    )


def configure_ring(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([])
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels([r"$0^\circ$", r"$90^\circ$", r"$180^\circ$", r"$270^\circ$"])
    ax.grid(color="0.86", lw=0.6)
    ax.spines["polar"].set_color("0.55")
    ax.spines["polar"].set_linewidth(0.8)


def draw_periodic_ring(ax, state):
    configure_ring(ax)
    theta = np.deg2rad(state["centers"])
    density = state["density"] / state["density"].max()
    radius = 0.60 + 0.32 * density
    closed_theta = np.r_[theta, theta[0] + 2.0 * np.pi]
    closed_radius = np.r_[radius, radius[0]]
    ax.fill_between(closed_theta, 0.60, closed_radius, color=BLUE, alpha=0.12)
    ax.plot(closed_theta, closed_radius, color=BLUE, lw=1.5)

    # Mark halo-bin locations without exaggerating their density.
    halo_bins = state["counts_halo"] > 0
    ax.scatter(
        theta[halo_bins],
        np.full(np.count_nonzero(halo_bins), 0.52),
        s=5,
        marker="s",
        color=GRAY,
        alpha=0.65,
        linewidths=0,
    )

    common = np.deg2rad(state["common_cut"])
    separator_deg = (state["common_cut"] + state["boundary_u"]) % base.PERIOD
    separator = np.deg2rad(separator_deg)
    ax.plot([common, common], [0.22, 1.00], color=RED, ls="--", lw=1.5)
    ax.plot([separator, separator], [0.22, 1.00], color=DARK, ls="--", lw=1.1)
    ax.text(common + np.deg2rad(5), 0.25, "common cut", color=RED, fontsize=7.5)
    ax.text(separator + np.deg2rad(4), 0.25, "boundary", color=DARK, fontsize=7.5)
    ax.text(np.deg2rad(0), 0.96, "1", color=BLUE, fontsize=8, ha="center")
    ax.text(np.deg2rad(180), 0.97, "2", color=BLUE, fontsize=8, ha="center")
    ax.set_title(r"Projection onto one period, $s_{\rm mod}=s\ \mathrm{mod}\ C$", fontsize=10)
    panel_label(ax, "(b)")


def draw_unfolded_result(ax, state):
    centers = state["u_centers"]
    width = 0.90 * state["width"]
    rows = (
        (state["region_1"], state["envelope_1"], 1.20, "bunch 1"),
        (state["region_2"], state["envelope_2"], 0.00, "bunch 2"),
    )
    for region, envelope, baseline, label in rows:
        counts = np.where(region, state["u_counts"], 0)
        heights = counts / max(1, counts.max())
        ax.bar(
            centers[region],
            heights[region],
            width=width,
            bottom=baseline,
            color=BLUE,
            edgecolor="none",
            alpha=0.88,
        )
        ax.plot(
            [envelope[0], envelope[1]],
            [baseline + 1.05, baseline + 1.05],
            color=GREEN,
            lw=3.0,
        )
        ax.axhline(baseline, color="0.62", lw=0.6)
    ax.axvline(0, color=RED, ls="--", lw=1.2)
    ax.axvline(360, color=RED, ls="--", lw=1.2)
    ax.axvline(state["boundary_u"], color=DARK, ls="--", lw=1.0)
    ax.set_xlim(0, 360)
    ax.set_ylim(-0.04, 2.34)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0.42, 1.62])
    ax.set_yticklabels(["bunch 2", "bunch 1"])
    ax.set_xlabel(r"common unfolded coordinate $u$ [deg]")
    ax.set_title("Continuous bunch coordinates", fontsize=10)
    style_cartesian(ax)
    panel_label(ax, "(c)")
    ax.legend(
        handles=[Line2D([0], [0], color=GREEN, lw=3, label="occupied core interval")],
        frameon=False,
        fontsize=7.5,
        loc="upper right",
    )


def make_figure_1(state):
    fig = plt.figure(figsize=(11.2, 3.65))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.16, 0.86, 1.10), wspace=0.34)
    draw_multiturn(fig.add_subplot(grid[0, 0]))
    draw_periodic_ring(fig.add_subplot(grid[0, 1], projection="polar"), state)
    draw_unfolded_result(fig.add_subplot(grid[0, 2]), state)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.18, top=0.88)
    if SAVE_FIGS:
        fig.savefig(FIG1_PNG, dpi=300, bbox_inches="tight")
        fig.savefig(FIG1_PDF, bbox_inches="tight")
    return fig


def draw_final_intervals(ax, state):
    centers = state["u_centers"]
    width = 0.90 * state["width"]
    rows = (
        (state["region_1"], state["selected_1"], state["envelope_1"], 1.20),
        (state["region_2"], state["selected_2"], state["envelope_2"], 0.00),
    )
    for region, selected, envelope, baseline in rows:
        counts = np.where(region, state["u_counts"], 0)
        values = counts / max(1, counts.max())
        selected_mask = np.zeros_like(region, dtype=bool)
        selected_mask[selected] = True
        outside = region & (counts > 0) & ~selected_mask
        ax.bar(centers[outside], values[outside], width=width, bottom=baseline,
               color=GRAY, edgecolor="none", alpha=0.72)
        ax.bar(centers[selected_mask], values[selected_mask], width=width,
               bottom=baseline, color=BLUE, edgecolor="none")
        ax.plot([envelope[0], envelope[1]], [baseline + 1.05, baseline + 1.05],
                color=GREEN, lw=3.2)
        ax.text(0.5 * sum(envelope), baseline + 1.10,
                rf"$L_{{\rm occ}}={envelope[1]-envelope[0]:.0f}^\circ$",
                color=GREEN, fontsize=8, ha="center")
        ax.axhline(baseline, color="0.62", lw=0.6)
    ax.set_xlim(0, 360)
    ax.set_ylim(-0.04, 2.34)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0.42, 1.62])
    ax.set_yticklabels(["bunch 2", "bunch 1"])
    ax.set_xlabel(r"common unfolded coordinate $u$ [deg]")
    ax.set_title(r"Occupied intervals containing $qQ_k$", fontsize=10)
    style_cartesian(ax)
    panel_label(ax, "(a)")
    ax.legend(
        handles=[
            Patch(color=BLUE, label="core-density bins"),
            Patch(color=GRAY, label="low-density bins"),
            Line2D([0], [0], color=GREEN, lw=3, label="occupied interval"),
        ],
        frameon=False,
        fontsize=7.5,
        loc="upper right",
    )


def slice_band(ax, y, label):
    x0, x1, x2, x3 = 0.08, 0.18, 0.82, 0.92
    height = 0.16
    ax.add_patch(Rectangle((x0, y - height / 2), x1 - x0, height,
                           facecolor=GRAY, edgecolor="none", alpha=0.82))
    ax.add_patch(Rectangle((x1, y - height / 2), x2 - x1, height,
                           facecolor=BLUE, edgecolor="none", alpha=0.18))
    ax.add_patch(Rectangle((x2, y - height / 2), x3 - x2, height,
                           facecolor=GRAY, edgecolor="none", alpha=0.82))
    for x in np.linspace(x1, x2, 25):
        ax.plot([x, x], [y - height / 2, y + height / 2], color=BLUE, lw=0.55)
    ax.text(0.045, y, label, ha="right", va="center", fontsize=8)
    ax.text(0.5 * (x0 + x1), y, "1", ha="center", va="center",
            fontsize=8, color="white")
    ax.text(0.5 * (x1 + x2), y, "126", ha="center", va="center",
            fontsize=8, color=BLUE)
    ax.text(0.5 * (x2 + x3), y, "1", ha="center", va="center",
            fontsize=8, color="white")


def draw_slice_grids(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Per-bunch longitudinal grids", fontsize=10)
    panel_label(ax, "(b)")
    ax.text(0.13, 0.86, "left\noverflow", ha="center", va="center",
            fontsize=7.5, color=DARK)
    ax.text(0.50, 0.86, "uniform core slices", ha="center", va="center",
            fontsize=7.5, color=BLUE)
    ax.text(0.87, 0.86, "right\noverflow", ha="center", va="center",
            fontsize=7.5, color=DARK)
    slice_band(ax, 0.66, "bunch 1")
    slice_band(ax, 0.42, "bunch 2")
    ax.text(0.50, 0.27, "all particles retain charge and receive fields",
            ha="center", fontsize=8, color=DARK)
    ax.text(0.50, 0.15, "transverse: one 2D solve per slice",
            ha="center", fontsize=8, color=BLUE)
    ax.text(0.50, 0.07, "longitudinal: one independent solve per bunch",
            ha="center", fontsize=8, color=GREEN)


def make_figure_2(state):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.75), gridspec_kw={"wspace": 0.30})
    draw_final_intervals(axes[0], state)
    draw_slice_grids(axes[1])
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.88)
    if SAVE_FIGS:
        fig.savefig(FIG2_PNG, dpi=300, bbox_inches="tight")
        fig.savefig(FIG2_PDF, bbox_inches="tight")
    return fig


def main():
    state = base.build_state()
    fig1 = make_figure_1(state)
    fig2 = make_figure_2(state)
    print(f"saved -> {FIG1_PNG}")
    print(f"saved -> {FIG1_PDF}")
    print(f"saved -> {FIG2_PNG}")
    print(f"saved -> {FIG2_PDF}")
    if SHOW_FIGS:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    main()
