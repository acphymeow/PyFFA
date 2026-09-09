"""Create the single-column manuscript composite with resonance lines.

The left column contains the LSC-off/on 3-D PIC histograms.  The right
column contains the corresponding fixed-reference particles colored by
their total transverse action.  Existing scripts and figures are not
overwritten.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

import plot_longitudinal_sc_benchmark_tunes_with_resonances as base


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR.parent / "04_lsc_tune_comparison" / "results"
OFF_PATH = (
    RESULTS
    / "new_lsc_off_nominal60"
    / "new_3d_vs_nominal60_toy_lsc_off_plasma_action.npz"
)
ON_PATH = (
    RESULTS
    / "new_lsc_on_nominal60"
    / "new_3d_vs_nominal60_theory_lsc_on_plasma_action.npz"
)
OUTPUT = (
    SCRIPT_DIR
    / "outputs"
    / "longitudinal_sc_tune_benchmark_with_resonances_plasma_action"
)


def add_references(
    axis: plt.Axes,
    star: tuple[float, float],
    *,
    show_legend: bool,
) -> None:
    """Draw the established references at a compact manuscript scale."""
    axis.axvline(
        base.QX_BARE,
        color=base.BARE_GUIDE_COLOR,
        linestyle=base.BARE_GUIDE_LINESTYLE,
        linewidth=1.25,
        zorder=4,
    )
    axis.axhline(
        base.QZ_BARE,
        color=base.BARE_GUIDE_COLOR,
        linestyle=base.BARE_GUIDE_LINESTYLE,
        linewidth=1.25,
        zorder=4,
    )
    axis.axvline(
        star[0],
        color=base.STAR_GUIDE_COLOR,
        linestyle=base.STAR_GUIDE_LINESTYLE,
        linewidth=1.45,
        zorder=4,
    )
    axis.axhline(
        star[1],
        color=base.STAR_GUIDE_COLOR,
        linestyle=base.STAR_GUIDE_LINESTYLE,
        linewidth=1.45,
        zorder=4,
    )
    axis.scatter(
        base.QX_BARE,
        base.QZ_BARE,
        marker="o",
        s=48,
        facecolor="white",
        edgecolor="blue",
        linewidth=1.4,
        label="Bare tune" if show_legend else "_nolegend_",
        zorder=12,
    )
    axis.scatter(
        *star,
        marker="*",
        s=105,
        color="red",
        edgecolor="black",
        linewidth=1.0,
        label="Max SC tune shift" if show_legend else "_nolegend_",
        zorder=13,
    )
    axis.set_xlim(base.QX_LIMITS)
    axis.set_ylim(base.QZ_LIMITS)
    if show_legend:
        legend = axis.legend(
            loc="upper left",
            fontsize=7.0,
            frameon=True,
            borderpad=0.28,
            handletextpad=0.45,
            labelspacing=0.22,
        )
        legend.set_zorder(30)


def main() -> None:
    off = np.load(OFF_PATH)
    on = np.load(ON_PATH)
    datasets = (
        (off["qr_3d"], off["qz_3d"]),
        (off["qr_toy"], off["qz_toy"]),
        (on["new_qr_fft"], on["new_qz_fft"]),
        (on["theory_qr_fft"], on["theory_qz_fft"]),
    )
    actions = (
        np.asarray(off["toy_jtot_mm_mrad"], dtype=np.float64),
        np.asarray(on["theory_jtot_mm_mrad"], dtype=np.float64),
    )
    action_norm = Normalize(
        vmin=float(min(action.min() for action in actions)),
        vmax=float(max(action.max() for action in actions)),
    )

    histograms = [
        np.histogram2d(
            *datasets[index],
            bins=256,
            range=(base.QX_LIMITS, base.QZ_LIMITS),
        )[0]
        for index in (0, 2)
    ]
    count_norm = LogNorm(
        vmin=1,
        vmax=max(float(histogram.max()) for histogram in histograms),
    )
    stars = (
        base.OFF_RED_STAR_OVERRIDE,
        base.OFF_RED_STAR_OVERRIDE,
        base.ON_RED_STAR_OVERRIDE,
        base.ON_RED_STAR_OVERRIDE,
    )
    if any(star is None for star in stars):
        off_star = (
            tuple(off["common_star"])
            if base.OFF_RED_STAR_OVERRIDE is None
            else base.OFF_RED_STAR_OVERRIDE
        )
        on_star = (
            tuple(on["common_star"])
            if base.ON_RED_STAR_OVERRIDE is None
            else base.ON_RED_STAR_OVERRIDE
        )
        stars = (off_star, off_star, on_star, on_star)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    x_edges = np.linspace(*base.QX_LIMITS, 257)
    z_edges = np.linspace(*base.QZ_LIMITS, 257)
    count_mesh = None
    for row, histogram in enumerate(histograms):
        count_mesh = axes[row, 0].pcolormesh(
            x_edges,
            z_edges,
            histogram.T,
            norm=count_norm,
            cmap="viridis",
            shading="auto",
            rasterized=True,
            zorder=1,
        )

    action_scatter = axes[0, 1].scatter(
        *datasets[1],
        c=actions[0],
        cmap="plasma",
        norm=action_norm,
        s=8,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    axes[1, 1].scatter(
        *datasets[3],
        c=actions[1],
        cmap="plasma",
        norm=action_norm,
        s=8,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for index, axis in enumerate(axes.flat):
        add_references(axis, stars[index], show_legend=index == 0)
        base.draw_resonances(
            axis,
            datasets[index][0],
            datasets[index][1],
            max_order=base.RESONANCE_MAX_ORDER,
            label_mode=base.RESONANCE_LABEL_MODE,
            line_color=base.RESONANCE_LINE_COLOR,
            text_color=base.RESONANCE_TEXT_COLOR,
            line_style=base.RESONANCE_LINESTYLE,
            line_width=base.RESONANCE_LINEWIDTH,
            line_alpha=base.RESONANCE_LINE_ALPHA,
            label_fontsize=8.5,
        )
        axis.text(
            0.035,
            0.68 if index == 0 else 0.88,
            panel_labels[index],
            transform=axis.transAxes,
            fontsize=14,
            fontweight="bold",
            zorder=35,
        )
        axis.tick_params(axis="both", labelsize=10)

    for axis in axes[:, 0]:
        axis.set_ylabel(r"$Q_z$", fontsize=15)
    for axis in axes[1, :]:
        axis.set_xlabel(r"$Q_x$", fontsize=15)

    if count_mesh is None:
        raise RuntimeError("3-D PIC count map was not created")
    count_bar = fig.colorbar(
        count_mesh,
        ax=axes[:, 0],
        pad=0.02,
        fraction=0.05,
    )
    count_bar.set_label("Counts", fontsize=12)
    count_bar.ax.tick_params(labelsize=9)
    action_bar = fig.colorbar(
        action_scatter,
        ax=axes[:, 1],
        pad=0.02,
        fraction=0.05,
    )
    action_bar.set_label(r"$J_r+J_z$ [$\pi$ mm mrad]", fontsize=12)
    action_bar.ax.tick_params(labelsize=9)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {OUTPUT.with_suffix('.png')}")
    print(f"Saved {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
