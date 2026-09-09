"""Shared paper-style plotting for the paired tune comparisons."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def _add_references(
    axis: plt.Axes,
    endpoint: tuple[float, float],
    qr_bare: float,
    qz_bare: float,
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    endpoint_label: str,
) -> None:
    axis.axvline(qr_bare, ls="--", color="tab:blue", lw=1.6)
    axis.axhline(qz_bare, ls="--", color="tab:blue", lw=1.6)
    axis.axvline(endpoint[0], ls=":", color="tab:red", lw=1.7)
    axis.axhline(endpoint[1], ls=":", color="tab:red", lw=1.7)
    axis.scatter(
        qr_bare, qz_bare, marker="o", s=110, facecolor="white",
        edgecolor="tab:blue", linewidth=2.0, label="Bare tune", zorder=10,
    )
    axis.scatter(
        *endpoint, marker="*", s=230, color="red", edgecolor="black",
        linewidth=1.2, label=endpoint_label, zorder=11,
    )
    axis.set(
        xlim=xrange, ylim=yrange, xlabel=r"$Q_r$", ylabel=r"$Q_z$",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="lower right", frameon=True)


def plot_tune_comparison(
    qr_3d: np.ndarray,
    qz_3d: np.ndarray,
    qr_toy: np.ndarray,
    qz_toy: np.ndarray,
    *,
    endpoint_3d: tuple[float, float],
    endpoint_toy: tuple[float, float],
    qr_bare: float,
    qz_bare: float,
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    title_3d: str,
    title_toy: str,
    output_stem: Path,
    endpoint_label_3d: str = "1% endpoint",
    endpoint_label_toy: str = "1% endpoint",
) -> plt.Figure:
    """Draw and save a matched-axis 3-D PIC versus toy comparison."""
    fig, axes = plt.subplots(
        1, 2, figsize=(13.8, 5.8), constrained_layout=True
    )
    hist = axes[0].hist2d(
        qr_3d, qz_3d, bins=256, range=(xrange, yrange),
        norm=LogNorm(), cmap="viridis",
    )
    fig.colorbar(hist[3], ax=axes[0], label="Counts")
    axes[1].scatter(
        qr_toy, qz_toy, s=15, alpha=0.72, color="tab:purple",
        linewidths=0, rasterized=True,
    )
    axes[0].set_title(title_3d)
    axes[1].set_title(title_toy)
    for index, (axis, endpoint, endpoint_label) in enumerate(
        zip(
            axes,
            (endpoint_3d, endpoint_toy),
            (endpoint_label_3d, endpoint_label_toy),
        )
    ):
        _add_references(
            axis, endpoint, qr_bare, qz_bare, xrange, yrange, endpoint_label
        )
        axis.text(
            0.045, 0.91, f"({chr(97 + index)})", transform=axis.transAxes,
            fontsize=25, fontweight="bold",
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    return fig
