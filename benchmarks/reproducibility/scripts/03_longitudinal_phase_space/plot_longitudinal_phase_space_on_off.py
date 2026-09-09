"""Plot longitudinal phase space: 3-D LSC on/off versus paired 1-D theory."""
import argparse
from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter


# -----------------------------------------------------------------------------
# Plot appearance.  Keep these settings together so paper styling is easy to
# adjust without searching through the drawing code.
# -----------------------------------------------------------------------------
FONT_FAMILY = "sans-serif"
FONT_SIZE = 10
AXES_LABEL_SIZE = 11
TICK_LABEL_SIZE = 10
TURN_TITLE_SIZE = 11
ANNOTATION_SIZE = 8
COLORBAR_LABEL_SIZE = 10
FIGURE_SIZE = (17.2, 6.6)
SAVE_DPI = 260


def apply_plot_settings():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "axes.labelsize": AXES_LABEL_SIZE,
        "axes.titlesize": TURN_TITLE_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
    })


apply_plot_settings()


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT as REPO, SHARED_LONGITUDINAL
sys.path[:0] = [str(HERE), str(SHARED_LONGITUDINAL)]

from level5_bucket_sampled_tracking import BETA_C, load_seo_freq_table, rk4_step_numba
from level47_3d_recipe_in_1d import disk_sum_1d
from run_gauss_waterbag_coupling_prediction import (
    EK_MEV, N_PROTONS, NS, QE, SEO, longitudinal_pair,
    smooth_slice_counts,
)


CASES = {
    "on": REPO / "HPC_dir" / "data_coupling_withLsc500",
    "off": REPO / "HPC_dir" / "Newdata_coupling_withoutLsc100",
}
TURNS = (1, 51, 101, 151, 201, 251)
OUT = HERE / "results" / "longitudinal_phase_space_on_off"
T_REF = 365.64994e-9
C_REF = BETA_C * T_REF
V_TURN = 200.0e3
SUBSTEPS = 64
S_EDGES = np.linspace(-22.0, 22.0, 111)
DE_EDGES = np.linspace(-8.0, 8.0, 91)


def load_3d(folder, turn):
    files = sorted((folder / f"TBT_Dump_{turn}").glob("*.npz"))
    particles = np.concatenate([np.load(path)["particles"] for path in files])
    order = np.lexsort((particles[:, 16], particles[:, 15]))
    particles = particles[order]
    _, first = np.unique(particles[:, 15].astype(np.int64), return_index=True)
    particles = particles[first]
    phase = particles[:, 9]
    phase_center = np.angle(np.mean(np.exp(1j * phase)))
    s = -C_REF * np.angle(np.exp(1j * (phase - phase_center))) / (2.0*np.pi)
    de = particles[:, 5] - particles[:, 5].mean()
    return s, de


def make_stepper(use_lsc, density_smooth_sigma_bins=0.0):
    energy, frequency = load_seo_freq_table(str(SEO / "SEO_ini.txt"))
    period = 1.0 / frequency
    central_period = float(np.interp(EK_MEV, energy, period))
    energy_grid = np.linspace(EK_MEV - 30.0, EK_MEV + 30.0, 20_001)
    period_grid = np.interp(energy_grid, energy, period).astype(np.float64)
    energy_min = float(energy_grid[0])
    energy_step = float(energy_grid[1] - energy_grid[0])
    dt = central_period / SUBSTEPS
    q_macro = N_PROTONS * QE / 20_000

    def advance(s, de):
        time_offset = -s.astype(np.float64) / BETA_C
        de = de.astype(np.float64).copy()
        electric_field = np.zeros_like(de)
        for _ in range(SUBSTEPS):
            if use_lsc:
                position = -BETA_C * time_offset
                grid = np.linspace(float(position.min()), float(position.max()), NS)
                ds = float(grid[1] - grid[0])
                bins = np.clip(((position - grid[0]) / ds).astype(np.int64), 0, NS - 1)
                counts = np.bincount(bins, minlength=NS)
                counts_for_field = smooth_slice_counts(
                    counts, density_smooth_sigma_bins
                )
                electric_field = disk_sum_1d(
                    grid, counts_for_field*q_macro
                )[bins]
            time_offset, de = rk4_step_numba(
                time_offset, de, dt, central_period,
                2.0*np.pi/central_period, 0.0, V_TURN,
                energy_min, energy_step, period_grid,
                float(use_lsc), BETA_C, electric_field,
            )
        position = -BETA_C * time_offset
        return position - position.mean(), de - de.mean()

    return advance


def load_theory(case, density_smooth_sigma_bins):
    cache_path = longitudinal_pair(
        max(TURNS),
        substeps=SUBSTEPS,
        density_smooth_sigma_bins=density_smooth_sigma_bins,
    )
    cache = np.load(cache_path)
    snap_turns = cache["snap_turns"].astype(int)
    step = make_stepper(case == "on", density_smooth_sigma_bins)
    key = "yes" if case == "on" else "no"
    initial_index = np.flatnonzero(snap_turns == 0)
    if len(initial_index) != 1:
        raise ValueError("theory cache has no unique turn-0 state")
    k0 = int(initial_index[0])
    s0 = cache[f"s_{key}"][k0].astype(np.float64)
    de0 = cache[f"de_{key}"][k0].astype(np.float64)
    s_semi_axis = BETA_C * 36.0e-9
    initial_action = (s0/s_semi_axis)**2 + (de0/4.0)**2
    initial_action = np.clip(initial_action, 0.0, 1.0)
    result = {}
    for turn in TURNS:
        index = np.flatnonzero(snap_turns == turn - 1)
        if len(index) != 1:
            raise ValueError(f"theory cache has no unique turn-{turn - 1} state")
        k = int(index[0])
        result[turn] = step(cache[f"s_{key}"][k], cache[f"de_{key}"][k])
    return result, initial_action


def probability_hist(s, de, smooth_sigma=0.0):
    hist, _, _ = np.histogram2d(s, de, bins=(S_EDGES, DE_EDGES))
    hist = hist.T / len(s)
    if smooth_sigma > 0.0:
        hist = gaussian_filter(
            hist,
            sigma=float(smooth_sigma),
            mode="constant",
            cval=0.0,
        )
    hist[hist == 0.0] = np.nan
    return hist


def metrics(s, de):
    return float(s.std()), float(de.std()), float(np.corrcoef(s, de)[0, 1])


def draw(
    case,
    states,
    norm,
    theory_smooth_sigma,
    output_suffix,
    plot_style,
    show_title=True,
    show_correlation=True,
    annotation_color="white",
    annotation_box=True,
    show_row_descriptions=True,
    annotation_positions=None,
):
    fig, axes = plt.subplots(2, len(TURNS), figsize=FIGURE_SIZE,
                             sharex=True, sharey=True, constrained_layout=True)
    image = None
    theory_scatter = None
    for column, turn in enumerate(TURNS):
        axes[0, column].set_title(f"Turn {turn}", fontsize=TURN_TITLE_SIZE)
        for row, source in enumerate(("3d", "theory")):
            axis = axes[row, column]
            s, de = states[turn][source]
            use_histogram = (
                plot_style == "hist2d"
                or (plot_style == "hybrid" and source == "3d")
            )
            if use_histogram:
                smooth_sigma = theory_smooth_sigma if source == "theory" else 0.0
                image = axis.pcolormesh(
                    S_EDGES,
                    DE_EDGES,
                    probability_hist(s, de, smooth_sigma=smooth_sigma),
                    shading="auto",
                    cmap="viridis",
                    norm=norm,
                    rasterized=True,
                )
            else:
                if plot_style == "hybrid":
                    theory_scatter = axis.scatter(
                        s,
                        de,
                        c=states[turn]["theory_color"],
                        cmap="plasma",
                        vmin=0.0,
                        vmax=1.0,
                        s=0.55,
                        alpha=0.75,
                        linewidths=0.0,
                        rasterized=True,
                    )
                else:
                    marker_size = 0.12 if source == "3d" else 0.45
                    marker_alpha = 0.22 if source == "3d" else 0.38
                    axis.scatter(
                        s,
                        de,
                        s=marker_size,
                        c="#1f77b4",
                        alpha=marker_alpha,
                        linewidths=0.0,
                        rasterized=True,
                    )
            sigma_s, sigma_e, corr = metrics(s, de)
            annotation_lines = [
                rf"$\sigma_s$={sigma_s:.2f} m",
                rf"$\sigma_E$={sigma_e:.2f} MeV",
            ]
            if show_correlation:
                annotation_lines.append(rf"$\rho$={corr:+.2f}")
            annotation_x, annotation_y = 0.04, 0.95
            annotation_va, annotation_ha = "top", "left"
            if annotation_positions is not None:
                annotation_position = annotation_positions.get((case, turn))
                if annotation_position is not None:
                    annotation_x, annotation_y, annotation_va, annotation_ha = (
                        annotation_position
                    )
            annotation_kwargs = {
                "transform": axis.transAxes,
                "va": annotation_va,
                "ha": annotation_ha,
                "fontsize": ANNOTATION_SIZE,
                "color": annotation_color,
            }
            if annotation_box:
                annotation_kwargs["bbox"] = dict(
                    facecolor="black",
                    alpha=0.42,
                    edgecolor="none",
                    pad=2.0,
                )
            axis.text(
                annotation_x,
                annotation_y,
                "\n".join(annotation_lines),
                **annotation_kwargs,
            )
            axis.grid(alpha=0.12)
    if show_row_descriptions:
        axes[0, 0].set_ylabel(
            "Complete PyFFA tracking\n" + r"$\Delta E_k$ [MeV]"
        )
        axes[1, 0].set_ylabel(
            "Reduced 1D model\n" + r"$\Delta E_k$ [MeV]"
        )
    else:
        axes[0, 0].set_ylabel(r"$\Delta E_k$ [MeV]")
        axes[1, 0].set_ylabel(r"$\Delta E_k$ [MeV]")
    for axis in axes[1]:
        axis.set_xlabel(r"$s$ [m]")
    if show_title:
        fig.suptitle(
            "Longitudinal phase space "
            f"{'with' if case == 'on' else 'without'} LSC: "
            "complete PyFFA tracking vs reduced 1D model"
        )
    if plot_style == "hybrid":
        density_colorbar = fig.colorbar(
            image,
            ax=axes[0, :],
            label="probability per bin",
            shrink=0.86,
            pad=0.015,
        )
        action_colorbar = fig.colorbar(
            theory_scatter,
            ax=axes[1, :],
            label="Normalized initial longitudinal amplitude",
            shrink=0.86,
            pad=0.015,
        )
        density_colorbar.set_label(
            "probability per bin", fontsize=COLORBAR_LABEL_SIZE
        )
        action_colorbar.set_label(
            "Normalized initial\nlongitudinal amplitude",
            fontsize=COLORBAR_LABEL_SIZE,
        )
    elif image is not None:
        fig.colorbar(
            image, ax=axes, label="probability per bin", shrink=0.86, pad=0.015
        )
    stem = OUT / (
        f"lsc_{case}_3d_vs_theory_longitudinal_phase_space{output_suffix}"
    )
    fig.savefig(stem.with_suffix(".png"), dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theory-smooth-sigma",
        type=float,
        default=0.0,
        help="Display-only Gaussian smoothing width in histogram bins.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix appended to output figure stems.",
    )
    parser.add_argument(
        "--density-smooth-sigma-bins",
        type=float,
        default=0.0,
        help=("Gaussian sigma applied to the 1-D line density during "
              "tracking, in current slice widths."),
    )
    parser.add_argument(
        "--plot-style",
        choices=("hist2d", "scatter", "hybrid"),
        default="hist2d",
        help=("Render both rows as histograms, both as scatters, or use a "
              "3-D histogram with a theory scatter."),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figures without opening the two interactive windows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.theory_smooth_sigma < 0.0:
        raise ValueError("--theory-smooth-sigma must be non-negative")
    if args.density_smooth_sigma_bins < 0.0:
        raise ValueError("--density-smooth-sigma-bins must be non-negative")
    output_suffix = args.output_suffix
    if args.density_smooth_sigma_bins > 0.0 and not output_suffix:
        tag = f"{args.density_smooth_sigma_bins:g}".replace(".", "p")
        output_suffix = f"_density_smooth{tag}bin"
    if args.plot_style != "hist2d":
        output_suffix += f"_{args.plot_style}"
    OUT.mkdir(parents=True, exist_ok=True)
    all_states = {}
    rows = []
    histograms = []
    for case, folder in CASES.items():
        theory, theory_color = load_theory(
            case, args.density_smooth_sigma_bins
        )
        states = {}
        for turn in TURNS:
            states[turn] = {
                "3d": load_3d(folder, turn),
                "theory": theory[turn],
                "theory_color": theory_color,
            }
            for source in ("3d", "theory"):
                s, de = states[turn][source]
                sigma_s, sigma_e, corr = metrics(s, de)
                rows.append((case, turn, source, len(s), sigma_s, sigma_e, corr))
                if (
                    args.plot_style == "hist2d"
                    or (args.plot_style == "hybrid" and source == "3d")
                ):
                    smooth_sigma = (
                        args.theory_smooth_sigma if source == "theory" else 0.0
                    )
                    histograms.append(
                        probability_hist(s, de, smooth_sigma=smooth_sigma)
                    )
            print(f"loaded {case}, turn {turn}", flush=True)
        all_states[case] = states

    norm = None
    if histograms:
        finite = np.concatenate([hist[np.isfinite(hist)] for hist in histograms])
        norm = LogNorm(vmin=max(5.0e-6, float(np.percentile(finite, 1))),
                       vmax=float(np.percentile(finite, 99.9)))
    figures = []
    for case in CASES:
        figures.append(draw(
            case,
            all_states[case],
            norm,
            theory_smooth_sigma=args.theory_smooth_sigma,
            output_suffix=output_suffix,
            plot_style=args.plot_style,
        ))

    with (OUT / f"longitudinal_phase_space_metrics{output_suffix}.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("lsc", "turn", "source", "n", "sigma_s_m",
                         "sigma_de_MeV", "corr_s_de"))
        writer.writerows(rows)
    print(f"saved outputs in {OUT}")
    print(
        "1-D density smoothing sigma = "
        f"{args.density_smooth_sigma_bins:.3g} slice widths"
    )
    print(f"phase-space plot style = {args.plot_style}")
    print(
        "theory display smoothing sigma = "
        f"{args.theory_smooth_sigma:.3g} histogram bins"
    )
    if not args.no_show:
        print("opening the LSC-on and LSC-off figure windows together")
        plt.show()
    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
