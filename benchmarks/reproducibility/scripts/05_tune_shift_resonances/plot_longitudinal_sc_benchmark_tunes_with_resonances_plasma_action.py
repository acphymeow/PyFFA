"""Plot action-colored LSC-off/on 1-D theory tune footprints with resonances.

This is an independent companion to
``plot_longitudinal_sc_benchmark_tunes_with_resonances.py``.  It reads tune
files that also contain the particle-by-particle initial transverse action,
uses the established plasma action coloring, and writes new suffixed figures.
The original resonance script and figures are not modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

import plot_longitudinal_sc_benchmark_tunes_with_resonances as base


SCRIPT_DIR = Path(__file__).resolve().parent
LSC_RESULTS = SCRIPT_DIR.parent / "04_lsc_tune_comparison" / "results"
DEFAULT_OFF = (
    LSC_RESULTS
    / "new_lsc_off_nominal60"
    / "new_3d_vs_nominal60_toy_lsc_off_plasma_action.npz"
)
DEFAULT_ON = (
    LSC_RESULTS
    / "new_lsc_on_nominal60"
    / "new_3d_vs_nominal60_theory_lsc_on_plasma_action.npz"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs"


def load_theory_data(
    off_path: Path,
    on_path: Path,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Load tune/action arrays and verify their particle-wise alignment."""
    off = np.load(off_path)
    on = np.load(on_path)
    datasets = (
        (
            np.asarray(off["qr_toy"], dtype=np.float64),
            np.asarray(off["qz_toy"], dtype=np.float64),
            np.asarray(off["toy_jtot_mm_mrad"], dtype=np.float64),
        ),
        (
            np.asarray(on["theory_qr_fft"], dtype=np.float64),
            np.asarray(on["theory_qz_fft"], dtype=np.float64),
            np.asarray(on["theory_jtot_mm_mrad"], dtype=np.float64),
        ),
    )
    for qr, qz, action in datasets:
        if qr.shape != qz.shape or qr.shape != action.shape:
            raise ValueError("Qr, Qz, and J_r+J_z arrays must have equal shapes")
        if not np.all(np.isfinite(action)):
            raise ValueError("action array contains non-finite values")
    return datasets


def make_plots(
    off_path: Path,
    on_path: Path,
    output_dir: Path,
    *,
    max_order: int,
    label_mode: str,
    line_color: str,
    text_color: str,
    line_style: str,
    line_width: float,
    line_alpha: float,
    label_fontsize: float,
) -> tuple[Path, Path]:
    datasets = load_theory_data(off_path, on_path)
    all_actions = np.concatenate([dataset[2] for dataset in datasets])
    action_norm = Normalize(
        vmin=float(all_actions.min()),
        vmax=float(all_actions.max()),
    )
    stars = (
        base.OFF_RED_STAR_OVERRIDE,
        base.ON_RED_STAR_OVERRIDE,
    )
    if any(star is None for star in stars):
        off = np.load(off_path)
        on = np.load(on_path)
        stars = (
            tuple(off["common_star"])
            if base.OFF_RED_STAR_OVERRIDE is None
            else base.OFF_RED_STAR_OVERRIDE,
            tuple(on["common_star"])
            if base.ON_RED_STAR_OVERRIDE is None
            else base.ON_RED_STAR_OVERRIDE,
        )
    output_names = (
        "longitudinal_sc_tune_fixed_reference_lsc_off_"
        "with_resonances_plasma_action",
        "longitudinal_sc_tune_fixed_reference_lsc_on_"
        "with_resonances_plasma_action",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for (qr, qz, action), star, output_name in zip(
        datasets, stars, output_names
    ):
        fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
        grid = fig.add_gridspec(
            1, 2, width_ratios=(1.0, 0.055), wspace=0.08
        )
        axis = fig.add_subplot(grid[0, 0])
        colorbar_axis = fig.add_subplot(grid[0, 1])

        scatter = axis.scatter(
            qr,
            qz,
            c=action,
            cmap="plasma",
            norm=action_norm,
            s=18,
            alpha=0.75,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        colorbar = fig.colorbar(scatter, cax=colorbar_axis)
        colorbar.set_label(r"$J_r+J_z$ [$\pi$ mm mrad]", fontsize=19)
        colorbar.ax.tick_params(labelsize=12)

        base.add_references(axis, star)
        base.draw_resonances(
            axis,
            qr,
            qz,
            max_order=max_order,
            label_mode=label_mode,
            line_color=line_color,
            text_color=text_color,
            line_style=line_style,
            line_width=line_width,
            line_alpha=line_alpha,
            label_fontsize=label_fontsize,
        )
        axis.set_ylabel(r"$Q_z$", fontsize=24)
        axis.set_xlabel(r"$Q_x$", fontsize=24)
        axis.tick_params(axis="both", labelsize=20)

        output = output_dir / output_name
        fig.savefig(output.with_suffix(".png"), dpi=300)
        fig.savefig(output.with_suffix(".pdf"))
        plt.close(fig)
        outputs.append(output)
    return outputs[0], outputs[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-npz", type=Path, default=DEFAULT_OFF)
    parser.add_argument("--on-npz", type=Path, default=DEFAULT_ON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-order", type=int, default=base.RESONANCE_MAX_ORDER
    )
    parser.add_argument(
        "--label-mode",
        choices=("all", "low-order", "highlighted", "none"),
        default=base.RESONANCE_LABEL_MODE,
    )
    parser.add_argument("--line-color", default=base.RESONANCE_LINE_COLOR)
    parser.add_argument("--text-color", default=base.RESONANCE_TEXT_COLOR)
    parser.add_argument(
        "--line-style",
        choices=("-", "--", "-.", ":"),
        default=base.RESONANCE_LINESTYLE,
    )
    parser.add_argument(
        "--line-width", type=float, default=base.RESONANCE_LINEWIDTH
    )
    parser.add_argument(
        "--line-alpha", type=float, default=base.RESONANCE_LINE_ALPHA
    )
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=base.RESONANCE_LABEL_FONTSIZE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_order < 1:
        raise ValueError("--max-order must be at least 1")
    if args.line_width < 0.0:
        raise ValueError("--line-width must be non-negative")
    if not 0.0 <= args.line_alpha <= 1.0:
        raise ValueError("--line-alpha must be between 0 and 1")
    outputs = make_plots(
        args.off_npz,
        args.on_npz,
        args.output_dir,
        max_order=args.max_order,
        label_mode=args.label_mode,
        line_color=args.line_color,
        text_color=args.text_color,
        line_style=args.line_style,
        line_width=args.line_width,
        line_alpha=args.line_alpha,
        label_fontsize=args.label_fontsize,
    )
    for output in outputs:
        print(f"Saved {output.with_suffix('.png')}")
        print(f"Saved {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
