#!/usr/bin/env python3
"""Generate the rightmost Fig. 8 benchmark panel as a paper figure.

The plotted numerical result is the authoritative NS=256 Gaussian
source-slice-averaged calculation after 8 cm Gaussian smoothing.  It is the
same result shown in the rightmost panel of
``fig8_gaussian_kernel_benchmark.png``.  No title or panel letter is added so
the output can be used directly as a standalone manuscript figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CANONICAL_DATA = (
    HERE.parent / "01_fig8_es" / "fig8_gaussian_only_data.csv"
)
LOCAL_DATA = HERE / "data" / "fig8_gaussian_right_panel_data.csv"
OUTPUT_DIR = HERE / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional three-column CSV overriding the bundled data snapshot.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def prepare_local_data(source: Path) -> Path:
    """Create a local, provenance-preserving snapshot when needed."""
    if not source.exists():
        raise FileNotFoundError(f"Fig. 8 source data not found: {source}")
    data = np.genfromtxt(source, delimiter=",", names=True)
    required = (
        "s_minus_sc_m",
        "Es_3D_Gaussian_theory_V_per_m",
        "Es_Gaussian_V_per_m",
    )
    missing = [name for name in required if name not in (data.dtype.names or ())]
    if missing:
        raise ValueError(f"Missing columns {missing} in {source}")

    LOCAL_DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        LOCAL_DATA,
        np.column_stack([data[name] for name in required]),
        delimiter=",",
        header=",".join(required),
        comments="",
    )
    return LOCAL_DATA


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return (
        np.asarray(data["s_minus_sc_m"], dtype=float),
        np.asarray(data["Es_3D_Gaussian_theory_V_per_m"], dtype=float),
        np.asarray(data["Es_Gaussian_V_per_m"], dtype=float),
    )


def make_figure(
    s_relative: np.ndarray,
    field_theory: np.ndarray,
    field_gaussian: np.ndarray,
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 18.0,
            "axes.labelsize": 22.0,
            "xtick.labelsize": 18.0,
            "ytick.labelsize": 18.0,
            "legend.fontsize": 14.0,
            "axes.linewidth": 0.9,
        }
    )

    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    pic_points, = axis.plot(
        s_relative,
        field_gaussian / 1.0e3,
        linestyle="none",
        marker=".",
        markersize=3.2,
        color="#d62728",
        alpha=0.9,
        label=r"2.5D PIC $E_s$",
        zorder=3,
    )
    theory_line, = axis.plot(
        s_relative,
        field_theory / 1.0e3,
        color="#2453ff",
        linewidth=2.5,
        label="3D Gaussian theory",
        zorder=1,
    )
    axis.set_xlabel(r"$s-s_c$ [m]")
    axis.set_ylabel(r"$E_s$ [kV/m]")
    axis.legend(
        handles=(pic_points, theory_line),
        loc="upper left",
        frameon=False,
    )
    fig.subplots_adjust(left=0.135, right=0.985, top=0.98, bottom=0.15)
    return fig


def main() -> None:
    args = parse_args()
    if args.data is None:
        data_path = LOCAL_DATA if LOCAL_DATA.exists() else prepare_local_data(CANONICAL_DATA)
    else:
        data_path = args.data

    s_relative, field_theory, field_gaussian = load_data(data_path)
    fig = make_figure(s_relative, field_theory, field_gaussian)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "fig8_gaussian_kernel_right_panel.png"
    pdf_path = args.output_dir / "fig8_gaussian_kernel_right_panel.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Data: {data_path}")
    print(f"PNG:  {png_path}")
    print(f"PDF:  {pdf_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
