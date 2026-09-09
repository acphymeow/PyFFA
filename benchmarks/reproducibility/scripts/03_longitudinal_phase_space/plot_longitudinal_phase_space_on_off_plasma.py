"""Plot the LSC-on/off longitudinal phase-space benchmarks in paper style.

For each case, the upper row is the 3-D PIC density map and the lower row is
the paired 1-D theory scatter colored by initial normalized longitudinal
action with the ``plasma`` colormap.  Both figures are saved before they are
opened together.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import plot_longitudinal_phase_space_on_off as base


# =============================================================================
# User-adjustable plot settings
# =============================================================================
FONT_FAMILY = "sans-serif"
FONT_SIZE = 16
AXES_LABEL_SIZE = 18
TICK_LABEL_SIZE = 15
TURN_TITLE_SIZE = 16
ANNOTATION_SIZE = 13
COLORBAR_LABEL_SIZE = 14
FIGURE_SIZE = (17.2, 6.6)
SAVE_DPI = 260
ANNOTATION_COLOR = "black"
ANNOTATION_BOX = False
SHOW_CORRELATION = False
# (case, turn): (x, y, vertical alignment, horizontal alignment), using
# axes-relative coordinates.  Turn 101 with LSC on reaches the upper-left
# corner, so place its text at the upper right.
ANNOTATION_POSITIONS = {("on", 101): (0.96, 0.95, "top", "right")}

# Data-display settings
DENSITY_SMOOTH_SIGMA_BINS = 1.75
HIST_VMIN = 5.0e-6
HIST_PERCENTILE_MIN = 1.0
HIST_PERCENTILE_MAX = 99.9


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "longitudinal_phase_space_lsc_on_off_plasma"
CASES_TO_PLOT = ("on", "off")


def apply_plot_settings():
    """Apply the settings above to the shared drawing routine."""
    base.FONT_FAMILY = FONT_FAMILY
    base.FONT_SIZE = FONT_SIZE
    base.AXES_LABEL_SIZE = AXES_LABEL_SIZE
    base.TICK_LABEL_SIZE = TICK_LABEL_SIZE
    base.TURN_TITLE_SIZE = TURN_TITLE_SIZE
    base.ANNOTATION_SIZE = ANNOTATION_SIZE
    base.COLORBAR_LABEL_SIZE = COLORBAR_LABEL_SIZE
    base.FIGURE_SIZE = FIGURE_SIZE
    base.SAVE_DPI = SAVE_DPI
    base.apply_plot_settings()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--density-smooth-sigma-bins",
        type=float,
        default=DENSITY_SMOOTH_SIGMA_BINS,
        help=(
            "Gaussian sigma applied to the 1-D line density during theory "
            f"tracking, in slice widths (default: {DENSITY_SMOOTH_SIGMA_BINS})."
        ),
    )
    parser.add_argument(
        "--hist-vmin",
        type=float,
        default=HIST_VMIN,
        help="Lower limit of the logarithmic 3-D density color scale.",
    )
    parser.add_argument(
        "--hist-vmax",
        type=float,
        default=None,
        help=(
            "Upper limit of the logarithmic 3-D density color scale. "
            "By default, use the configured percentile of all on/off maps."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save both figures without opening interactive windows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.density_smooth_sigma_bins < 0.0:
        raise ValueError("--density-smooth-sigma-bins must be non-negative")
    if args.hist_vmin <= 0.0:
        raise ValueError("--hist-vmin must be positive")
    if args.hist_vmax is not None and args.hist_vmax <= args.hist_vmin:
        raise ValueError("--hist-vmax must be greater than --hist-vmin")

    apply_plot_settings()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_states = {}
    histograms = []
    rows = []
    for case in CASES_TO_PLOT:
        theory, theory_color = base.load_theory(
            case,
            args.density_smooth_sigma_bins,
        )
        states = {}
        for turn in base.TURNS:
            states[turn] = {
                "3d": base.load_3d(base.CASES[case], turn),
                "theory": theory[turn],
                "theory_color": theory_color,
            }
            histograms.append(base.probability_hist(*states[turn]["3d"]))
            for source in ("3d", "theory"):
                s, de = states[turn][source]
                sigma_s, sigma_e, corr = base.metrics(s, de)
                rows.append(
                    (case, turn, source, len(s), sigma_s, sigma_e, corr)
                )
            print(f"loaded LSC {case}, turn {turn}", flush=True)
        all_states[case] = states

    finite = np.concatenate(
        [histogram[np.isfinite(histogram)] for histogram in histograms]
    )
    vmax = (
        float(args.hist_vmax)
        if args.hist_vmax is not None
        else float(np.percentile(finite, HIST_PERCENTILE_MAX))
    )
    norm = LogNorm(
        vmin=max(
            args.hist_vmin,
            float(np.percentile(finite, HIST_PERCENTILE_MIN)),
        ),
        vmax=vmax,
    )

    smooth_tag = f"{args.density_smooth_sigma_bins:g}".replace(".", "p")
    output_suffix = f"_density_smooth{smooth_tag}bin_plasma_action"
    figures = []
    original_out = base.OUT
    try:
        base.OUT = args.output_dir
        for case in CASES_TO_PLOT:
            figures.append(
                base.draw(
                    case,
                    all_states[case],
                    norm,
                    theory_smooth_sigma=0.0,
                    output_suffix=output_suffix,
                    plot_style="hybrid",
                    show_title=False,
                    show_correlation=SHOW_CORRELATION,
                    annotation_color=ANNOTATION_COLOR,
                    annotation_box=ANNOTATION_BOX,
                    show_row_descriptions=False,
                    annotation_positions=ANNOTATION_POSITIONS,
                )
            )
    finally:
        base.OUT = original_out

    metrics_path = args.output_dir / "longitudinal_phase_space_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "lsc",
                "turn",
                "source",
                "n",
                "sigma_s_m",
                "sigma_de_MeV",
                "corr_s_de",
            )
        )
        writer.writerows(rows)

    print(f"saved outputs in {args.output_dir}")
    print(
        "1-D theory color: plasma, initial normalized longitudinal action "
        "in [0, 1]"
    )
    if not args.no_show:
        print("opening the LSC-on and LSC-off plasma figures together")
        plt.show()
    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
