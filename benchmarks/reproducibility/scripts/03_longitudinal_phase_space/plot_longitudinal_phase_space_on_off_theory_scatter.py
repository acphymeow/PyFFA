"""Plot 3-D PIC density maps against 1-D theory particle scatter plots.

This is an alternative presentation of
``plot_longitudinal_phase_space_on_off.py``.  The original script is left
unchanged.  Only the 3-D PIC row is binned; the 1-D theory row shows the
individual theory macroparticles directly.
"""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import plot_longitudinal_phase_space_on_off as base


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "longitudinal_phase_space_on_off_theory_scatter"
THEORY_COLOR = "#D55E00"


def metrics_text(s, de):
    sigma_s, sigma_e, corr = base.metrics(s, de)
    return (
        rf"$\sigma_s$={sigma_s:.2f} m" + "\n"
        + rf"$\sigma_E$={sigma_e:.2f} MeV" + "\n"
        + rf"$\rho$={corr:+.2f}"
    )


def draw(case, states, norm):
    fig = plt.figure(figsize=(17.2, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        len(base.TURNS) + 1,
        width_ratios=[1.0] * len(base.TURNS) + [0.055],
        wspace=0.04,
        hspace=0.08,
    )
    axes = np.empty((2, len(base.TURNS)), dtype=object)
    for row in range(2):
        for column in range(len(base.TURNS)):
            if row == 0 and column == 0:
                axes[row, column] = fig.add_subplot(grid[row, column])
            else:
                axes[row, column] = fig.add_subplot(
                    grid[row, column],
                    sharex=axes[0, 0],
                    sharey=axes[0, 0],
                )
            axes[row, column].label_outer()
    colorbar_axis = fig.add_subplot(grid[0, -1])
    unused_axis = fig.add_subplot(grid[1, -1])
    unused_axis.set_axis_off()

    image = None
    for column, turn in enumerate(base.TURNS):
        axes[0, column].set_title(f"Turn {turn}")

        s_3d, de_3d = states[turn]["3d"]
        image = axes[0, column].pcolormesh(
            base.S_EDGES,
            base.DE_EDGES,
            base.probability_hist(s_3d, de_3d),
            shading="auto",
            cmap="viridis",
            norm=norm,
            rasterized=True,
        )
        axes[0, column].text(
            0.04,
            0.95,
            metrics_text(s_3d, de_3d),
            transform=axes[0, column].transAxes,
            va="top",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.42, edgecolor="none", pad=2.0),
        )

        s_theory, de_theory = states[turn]["theory"]
        axes[1, column].scatter(
            s_theory,
            de_theory,
            s=2.0,
            color=THEORY_COLOR,
            alpha=0.18,
            edgecolors="none",
            rasterized=True,
        )
        axes[1, column].text(
            0.04,
            0.95,
            metrics_text(s_theory, de_theory),
            transform=axes[1, column].transAxes,
            va="top",
            fontsize=8,
            color="black",
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2.0),
        )

        for row in range(2):
            axes[row, column].set_xlim(base.S_EDGES[0], base.S_EDGES[-1])
            axes[row, column].set_ylim(base.DE_EDGES[0], base.DE_EDGES[-1])
            axes[row, column].grid(alpha=0.12)

    axes[0, 0].set_ylabel("3-D PIC density\n" + r"$\Delta E_k$ [MeV]")
    axes[1, 0].set_ylabel("1-D theory particles\n" + r"$\Delta E_k$ [MeV]")
    for axis in axes[1]:
        axis.set_xlabel(r"$s$ [m]")

    label = "LSC on" if case == "on" else "LSC off"
    fig.suptitle(
        f"Longitudinal phase space, {label}: 3-D density and 1-D particles",
        fontsize=16,
    )
    fig.colorbar(
        image,
        cax=colorbar_axis,
        label="3-D PIC probability per bin",
    )

    stem = OUT / f"lsc_{case}_3d_hist_theory_scatter_longitudinal_phase_space"
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_states = {}
    rows = []
    histograms_3d = []

    for case, folder in base.CASES.items():
        theory = base.load_theory(case)
        states = {}
        for turn in base.TURNS:
            states[turn] = {
                "3d": base.load_3d(folder, turn),
                "theory": theory[turn],
            }
            histograms_3d.append(base.probability_hist(*states[turn]["3d"]))
            for source in ("3d", "theory"):
                s, de = states[turn][source]
                sigma_s, sigma_e, corr = base.metrics(s, de)
                rows.append((case, turn, source, len(s), sigma_s, sigma_e, corr))
            print(f"loaded {case}, turn {turn}", flush=True)
        all_states[case] = states

    # The color normalization is derived only from the 3-D density maps.
    # Theory is rendered as a single-color scatter plot and has no color scale.
    finite = np.concatenate(
        [hist[np.isfinite(hist)] for hist in histograms_3d]
    )
    norm = LogNorm(
        vmin=max(5.0e-6, float(np.percentile(finite, 1))),
        vmax=float(np.percentile(finite, 99.9)),
    )
    for case in base.CASES:
        draw(case, all_states[case], norm)

    with (OUT / "longitudinal_phase_space_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("lsc", "turn", "source", "n", "sigma_s_m", "sigma_de_MeV", "corr_s_de")
        )
        writer.writerows(rows)
    print(f"saved outputs in {OUT}")


if __name__ == "__main__":
    main()
