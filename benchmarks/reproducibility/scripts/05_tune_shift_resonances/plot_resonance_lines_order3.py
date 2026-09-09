"""Plot tune resonances up to a selected order.

The resonance condition is

    m * Qx + n * Qz = p,

where m, n, and p are integers and the resonance order is |m| + |n|.
Geometrically duplicate lines are removed.  By default the script overlays
the available 80 pi mm mrad Fig. 10 benchmark data, making it easy to judge
whether a full third-order resonance diagram is too dense for the paper.

Examples
--------
python plot_resonance_lines_order3.py
python plot_resonance_lines_order3.py --label-mode highlighted
python plot_resonance_lines_order3.py --no-data --qx-limits 2.3 3.3 --qz-limits 1.9 2.9
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bundle_paths import REPO_ROOT

DEFAULT_DATA_DIR = SCRIPT_DIR / "outputs" / "Benchmark_SC_tuneshift_gaussL"
DEFAULT_PIC_CSV = DEFAULT_DATA_DIR / "pyffa_tracking_tune_20000_hist2d.csv"
DEFAULT_THEORY_CSV = DEFAULT_DATA_DIR / "theory_tracking_tune_2000_scatter.csv"

ORDER_STYLE = {
    1: dict(color="#bdbdbd", linestyle="--", linewidth=0.85, alpha=0.78),
    2: dict(color="#bdd7e7", linestyle="-.", linewidth=0.85, alpha=0.78),
    3: dict(color="#fdd0a2", linestyle=":", linewidth=0.90, alpha=0.82),
}
ORDER_TEXT_COLOR = {
    1: "#737373",
    2: "#4f91bd",
    3: "#d97913",
}


@dataclass(frozen=True)
class ResonanceLine:
    m: int
    n: int
    p: int
    order: int
    label: str


def _canonical_coefficients(m: int, n: int, p: int) -> tuple[int, int, int]:
    """Return a unique integer representation of one geometric line."""
    divisor = math.gcd(math.gcd(abs(m), abs(n)), abs(p))
    if divisor > 1:
        m, n, p = m // divisor, n // divisor, p // divisor

    # Lines remain unchanged when all coefficients are multiplied by -1.
    if m < 0 or (m == 0 and n < 0):
        m, n, p = -m, -n, -p
    return m, n, p


def _term(coefficient: int, symbol: str, first: bool) -> str:
    if coefficient == 0:
        return ""
    sign = "-" if coefficient < 0 else ("" if first else "+")
    magnitude = "" if abs(coefficient) == 1 else str(abs(coefficient))
    return f"{sign}{magnitude}{symbol}"


def resonance_label(m: int, n: int, p: int) -> str:
    first = True
    pieces: list[str] = []
    if m:
        pieces.append(_term(m, r"Q_x", first))
        first = False
    if n:
        pieces.append(_term(n, r"Q_z", first))
    return "$" + "".join(pieces) + f"={p}$"


def generate_resonances(
    qx_limits: tuple[float, float],
    qz_limits: tuple[float, float],
    max_order: int = 3,
) -> list[ResonanceLine]:
    """Generate unique resonance lines intersecting the requested tune window."""
    xmin, xmax = qx_limits
    ymin, ymax = qz_limits
    corners = ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))
    unique: dict[tuple[int, int, int], ResonanceLine] = {}

    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            requested_order = abs(m) + abs(n)
            if requested_order == 0 or requested_order > max_order:
                continue

            values = [m * x + n * y for x, y in corners]
            p_min = math.ceil(min(values) - 1e-12)
            p_max = math.floor(max(values) + 1e-12)
            for p in range(p_min, p_max + 1):
                cm, cn, cp = _canonical_coefficients(m, n, p)
                actual_order = abs(cm) + abs(cn)
                if actual_order > max_order:
                    continue
                key = (cm, cn, cp)
                unique[key] = ResonanceLine(
                    cm, cn, cp, actual_order, resonance_label(cm, cn, cp)
                )

    return sorted(
        unique.values(),
        key=lambda line: (line.order, line.m, line.n, line.p),
    )


def line_segment_in_window(
    line: ResonanceLine,
    qx_limits: tuple[float, float],
    qz_limits: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the endpoints of a resonance clipped to the tune window."""
    xmin, xmax = qx_limits
    ymin, ymax = qz_limits
    m, n, p = line.m, line.n, line.p
    candidates: list[tuple[float, float]] = []

    if n != 0:
        for x in (xmin, xmax):
            y = (p - m * x) / n
            if ymin - 1e-10 <= y <= ymax + 1e-10:
                candidates.append((x, y))
    if m != 0:
        for y in (ymin, ymax):
            x = (p - n * y) / m
            if xmin - 1e-10 <= x <= xmax + 1e-10:
                candidates.append((x, y))

    distinct: list[tuple[float, float]] = []
    for point in candidates:
        if not any(np.hypot(point[0] - q[0], point[1] - q[1]) < 1e-9 for q in distinct):
            distinct.append(point)
    if len(distinct) < 2:
        return None

    best_pair = (distinct[0], distinct[1])
    best_distance = -1.0
    for i, point_a in enumerate(distinct):
        for point_b in distinct[i + 1 :]:
            distance = (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2
            if distance > best_distance:
                best_distance = distance
                best_pair = (point_a, point_b)
    return best_pair


def is_reviewer_highlight(line: ResonanceLine) -> bool:
    return (line.m, line.n, line.p) in {(1, 0, 3), (0, 3, 8)}


def _label_rotation(ax: plt.Axes, segment: tuple[tuple[float, float], tuple[float, float]]) -> float:
    p0, p1 = segment
    xy = ax.transData.transform(np.asarray([p0, p1]))
    delta = xy[1] - xy[0]
    angle = math.degrees(math.atan2(delta[1], delta[0]))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return angle


def draw_resonance_lines(
    ax: plt.Axes,
    qx_limits: tuple[float, float],
    qz_limits: tuple[float, float],
    max_order: int = 3,
    label_mode: str = "all",
) -> list[ResonanceLine]:
    """Draw resonance lines and return the list that intersects the axes."""
    lines = generate_resonances(qx_limits, qz_limits, max_order=max_order)
    ax.set_xlim(qx_limits)
    ax.set_ylim(qz_limits)
    ax.figure.canvas.draw()

    for line in lines:
        segment = line_segment_in_window(line, qx_limits, qz_limits)
        if segment is None:
            continue
        highlighted = is_reviewer_highlight(line)
        style = ORDER_STYLE[line.order]
        (x0, y0), (x1, y1) = segment
        ax.plot([x0, x1], [y0, y1], zorder=6, **style)

        should_label = label_mode == "all" or (
            label_mode == "highlighted" and highlighted
        )
        if should_label:
            fraction = 0.46 if line.order == 1 else (0.56 if line.order == 2 else 0.66)
            x_text = x0 + fraction * (x1 - x0)
            y_text = y0 + fraction * (y1 - y0)
            ax.text(
                x_text,
                y_text,
                line.label,
                color=ORDER_TEXT_COLOR[line.order],
                fontsize=7.2 if label_mode == "all" else 10.0,
                fontweight="bold",
                rotation=_label_rotation(ax, segment),
                rotation_mode="anchor",
                ha="center",
                va="center",
                zorder=8,
                bbox=dict(
                    boxstyle="round,pad=0.10",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.68,
                ),
            )
    return lines


def add_tune_markers(
    ax: plt.Axes,
    bare_tune: tuple[float, float],
    max_shift_tune: tuple[float, float],
) -> None:
    """Add the bare tune, maximum-SC-shift tune, and projection guides."""
    bare_qx, bare_qz = bare_tune
    shift_qx, shift_qz = max_shift_tune
    ax.axvline(bare_qx, color="tab:blue", linestyle="--", linewidth=1.15, alpha=0.9, zorder=5)
    ax.axhline(bare_qz, color="tab:blue", linestyle="--", linewidth=1.15, alpha=0.9, zorder=5)
    ax.axvline(shift_qx, color="tab:blue", linestyle=":", linewidth=1.15, alpha=0.9, zorder=5)
    ax.axhline(shift_qz, color="tab:blue", linestyle=":", linewidth=1.15, alpha=0.9, zorder=5)
    ax.scatter(
        [bare_qx], [bare_qz], marker="o", s=78, facecolor="white",
        edgecolor="blue", linewidth=1.8, label="Bare tune", zorder=20,
    )
    ax.scatter(
        [shift_qx], [shift_qz], marker="*", s=170, facecolor="red",
        edgecolor="black", linewidth=1.25, label="Max SC tune shift", zorder=21,
    )


def add_resonance_legend(ax: plt.Axes, max_order: int, include_tune_markers: bool) -> None:
    handles = []
    for order in range(1, max_order + 1):
        style = ORDER_STYLE[order]
        handles.append(
            Line2D(
                [0], [0], label=f"Order {order}",
                color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"] + 0.4,
            )
        )
    if include_tune_markers:
        handles.extend(
            [
                Line2D(
                    [0], [0], marker="o", linestyle="none", markersize=7.2,
                    markerfacecolor="white", markeredgecolor="blue",
                    markeredgewidth=1.5, label="Bare tune",
                ),
                Line2D(
                    [0], [0], marker="*", linestyle="none", markersize=10.5,
                    markerfacecolor="red", markeredgecolor="black",
                    markeredgewidth=1.0, label="Max SC tune shift",
                ),
            ]
        )
    ax.legend(handles=handles, loc="upper left", fontsize=8.0, framealpha=0.92)


def load_csv_columns(path: Path, columns: tuple[str, ...]) -> tuple[np.ndarray, ...]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    missing = [name for name in columns if name not in (data.dtype.names or ())]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return tuple(np.asarray(data[name], dtype=float) for name in columns)


def make_preview(
    args: argparse.Namespace,
) -> tuple[list[tuple[str, plt.Figure]], list[ResonanceLine]]:
    qx_limits = tuple(args.qx_limits)
    qz_limits = tuple(args.qz_limits)
    use_data = not args.no_data

    if use_data:
        if not args.pic_csv.exists() or not args.theory_csv.exists():
            raise FileNotFoundError(
                "The default Fig. 10 CSV files were not found. "
                "Use --no-data for a resonance-only diagram or provide --pic-csv/--theory-csv."
            )
        qx_pic, qz_pic = load_csv_columns(args.pic_csv, ("QR_py", "QZ_py"))
        qx_th, qz_th, action = load_csv_columns(
            args.theory_csv, ("QR_th_plot", "QZ_th_plot", "J_total_mm_mrad")
        )
        fig_pic, axis_pic = plt.subplots(
            figsize=(7.2, 6.2), constrained_layout=True
        )
        hist = axis_pic.hist2d(
            qx_pic,
            qz_pic,
            bins=(220, 220),
            range=(qx_limits, qz_limits),
            norm=LogNorm(),
            cmap="viridis",
            zorder=1,
        )
        fig_pic.colorbar(hist[3], ax=axis_pic, label="Counts")
        axis_pic.set_title(r"2.5D PIC tracking, $\epsilon=80\pi$ mm mrad")

        fig_theory, axis_theory = plt.subplots(
            figsize=(7.2, 6.2), constrained_layout=True
        )
        scatter = axis_theory.scatter(
            qx_th,
            qz_th,
            c=action,
            cmap="plasma",
            s=9,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        fig_theory.colorbar(
            scatter,
            ax=axis_theory,
            label=r"$J_x+J_z$ [$\pi$ mm mrad]",
        )
        axis_theory.set_title(
            r"Analytical tracking, $\epsilon=80\pi$ mm mrad"
        )
        figures_and_axes = [
            ("pic", fig_pic, axis_pic),
            ("theory", fig_theory, axis_theory),
        ]
    else:
        fig, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
        axis.set_title(f"Tune resonances up to order {args.max_order}")
        figures_and_axes = [("", fig, axis)]

    all_lines: list[ResonanceLine] = []
    figures = []
    for suffix, fig, ax in figures_and_axes:
        ax.set_xlabel(r"$Q_x$")
        ax.set_ylabel(r"$Q_z$")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="0.9", linewidth=0.6, zorder=0)
        lines = draw_resonance_lines(
            ax,
            qx_limits,
            qz_limits,
            max_order=args.max_order,
            label_mode=args.label_mode,
        )
        include_tune_markers = use_data and not args.no_tune_markers
        if include_tune_markers:
            add_tune_markers(
                ax,
                tuple(args.bare_tune),
                tuple(args.max_shift_tune),
            )
        if not all_lines:
            all_lines = lines
        add_resonance_legend(ax, args.max_order, include_tune_markers)
        figures.append((suffix, fig))

    return figures, all_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--qx-limits", nargs=2, type=float, default=(2.8, 3.3))
    parser.add_argument("--qz-limits", nargs=2, type=float, default=(2.4, 2.9))
    parser.add_argument(
        "--label-mode",
        choices=("all", "highlighted", "none"),
        default="all",
        help="Which resonance lines receive direct text labels.",
    )
    parser.add_argument("--no-data", action="store_true")
    parser.add_argument("--pic-csv", type=Path, default=DEFAULT_PIC_CSV)
    parser.add_argument("--theory-csv", type=Path, default=DEFAULT_THEORY_CSV)
    parser.add_argument(
        "--bare-tune", nargs=2, type=float, default=(3.23, 2.81),
        metavar=("QX", "QZ"),
    )
    parser.add_argument(
        "--max-shift-tune", nargs=2, type=float, default=(3.077, 2.643),
        metavar=("QX", "QZ"),
    )
    parser.add_argument("--no-tune-markers", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--output-stem", default="fig10_order3_resonance_preview")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_order < 1 or args.max_order > 3:
        raise ValueError("This preview supports resonance orders 1 through 3.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures, lines = make_preview(args)
    output_paths = []
    for suffix, fig in figures:
        stem = args.output_stem if not suffix else f"{args.output_stem}_{suffix}"
        png_path = args.output_dir / f"{stem}.png"
        pdf_path = args.output_dir / f"{stem}.pdf"
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        output_paths.append((png_path, pdf_path))

    print(f"Generated {len(lines)} unique resonance lines up to order {args.max_order}.")
    for order in range(1, args.max_order + 1):
        labels = [line.label.strip("$") for line in lines if line.order == order]
        print(f"Order {order}: {len(labels)} lines: " + ", ".join(labels))
    for png_path, pdf_path in output_paths:
        print(f"PNG: {png_path}")
        print(f"PDF: {pdf_path}")
    plt.show()


if __name__ == "__main__":
    main()
