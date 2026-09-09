"""Plot the LSC-off longitudinal phase-space benchmark in paper style.

The upper row shows the 3-D PIC distribution as a two-dimensional density
map.  The lower row shows the paired 1-D theory particles, colored with the
``plasma`` colormap by their initial normalized longitudinal action.

Only the LSC-off case is loaded, so this entry point is faster and more
explicit than rerunning the general on/off plotting script.
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


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "longitudinal_phase_space_lsc_off_plasma"


def apply_plot_settings():
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
        default=1.75,
        help=(
            "Gaussian sigma applied to the 1-D line density during theory "
            "tracking, in slice widths (default: 1.75)."
        ),
    )
    parser.add_argument(
        "--hist-vmin",
        type=float,
        default=5.0e-6,
        help="Lower limit of the logarithmic 3-D density color scale.",
    )
    parser.add_argument(
        "--hist-vmax",
        type=float,
        default=None,
        help=(
            "Upper limit of the logarithmic 3-D density color scale. "
            "By default, use the 99.9th percentile of the LSC-off maps."
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
        help="Save the figure without opening an interactive window.",
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

    case = "off"
    theory, theory_color = base.load_theory(
        case,
        args.density_smooth_sigma_bins,
    )
    states = {}
    histograms = []
    rows = []

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
        print(f"loaded LSC off, turn {turn}", flush=True)

    finite = np.concatenate(
        [histogram[np.isfinite(histogram)] for histogram in histograms]
    )
    vmax = (
        float(args.hist_vmax)
        if args.hist_vmax is not None
        else float(np.percentile(finite, 99.9))
    )
    norm = LogNorm(
        vmin=max(args.hist_vmin, float(np.percentile(finite, 1.0))),
        vmax=vmax,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_out = base.OUT
    try:
        base.OUT = args.output_dir
        smooth_tag = f"{args.density_smooth_sigma_bins:g}".replace(".", "p")
        figure = base.draw(
            case,
            states,
            norm,
            theory_smooth_sigma=0.0,
            output_suffix=(
                f"_density_smooth{smooth_tag}bin_plasma_action"
            ),
            plot_style="hybrid",
            show_title=False,
            show_correlation=SHOW_CORRELATION,
            annotation_color=ANNOTATION_COLOR,
            annotation_box=ANNOTATION_BOX,
            show_row_descriptions=False,
        )
    finally:
        base.OUT = original_out

    metrics_path = args.output_dir / "lsc_off_longitudinal_phase_space_metrics.csv"
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
        print("opening the LSC-off plasma figure")
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
