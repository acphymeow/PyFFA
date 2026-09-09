"""Plot four independent LSC-off/on tune figures with resonances."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm


SCRIPT_DIR = Path(__file__).resolve().parent
LSC_RESULTS = (
    SCRIPT_DIR.parent / "04_lsc_tune_comparison" / "results"
)
DEFAULT_OFF = (
    LSC_RESULTS
    / "new_lsc_off_nominal60"
    / "new_3d_vs_nominal60_toy_lsc_off.npz"
)
DEFAULT_ON = (
    LSC_RESULTS
    / "new_lsc_on_nominal60"
    / "new_3d_vs_nominal60_theory_lsc_on.npz"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs"

QX_LIMITS = (2.80, 3.30)
QZ_LIMITS = (2.40, 2.90)
QX_BARE = 3.23
QZ_BARE = 2.81
BARE_GUIDE_COLOR = "tab:blue"
BARE_GUIDE_LINESTYLE = "--"
BARE_GUIDE_LINEWIDTH = 2.0
STAR_GUIDE_COLOR = "tab:blue"
STAR_GUIDE_LINESTYLE = ":"
STAR_GUIDE_LINEWIDTH = 2.4
# OFF_RED_STAR_OVERRIDE: tuple[float, float] | None = None
# ON_RED_STAR_OVERRIDE: tuple[float, float] | None = None
OFF_RED_STAR_OVERRIDE = (3.014, 2.560)
ON_RED_STAR_OVERRIDE = (3.115, 2.685)

# Resonance controls.  These defaults can also be overridden from the command
# line; for example: --max-order 4 --line-width 0.45 --label-mode low-order.
RESONANCE_MAX_ORDER = 3
RESONANCE_LABEL_MODE = "all"  # all, low-order, highlighted, or none
RESONANCE_LINE_COLOR = "#c7c7c7"
RESONANCE_TEXT_COLOR = "#686868"
RESONANCE_LINESTYLE = "-"
RESONANCE_LINEWIDTH = 0.65
RESONANCE_LINE_ALPHA = 1.0
RESONANCE_LABEL_FONTSIZE = 7.5
RESONANCE_LABEL_BOX_ALPHA = 0.68
RESONANCE_LABEL_ZORDER = -1
RESONANCE_BOLD_LABELS = True


@dataclass(frozen=True)
class ResonanceLine:
    m: int
    n: int
    p: int
    order: int
    label: str


def canonical_coefficients(m: int, n: int, p: int) -> tuple[int, int, int]:
    """Return one unique integer representation of a resonance line."""
    divisor = math.gcd(math.gcd(abs(m), abs(n)), abs(p))
    if divisor > 1:
        m, n, p = m // divisor, n // divisor, p // divisor
    if m < 0 or (m == 0 and n < 0):
        m, n, p = -m, -n, -p
    return m, n, p


def resonance_term(coefficient: int, symbol: str, first: bool) -> str:
    if coefficient == 0:
        return ""
    sign = "-" if coefficient < 0 else ("" if first else "+")
    magnitude = "" if abs(coefficient) == 1 else str(abs(coefficient))
    return f"{sign}{magnitude}{symbol}"


def resonance_label(m: int, n: int, p: int) -> str:
    pieces: list[str] = []
    if m:
        pieces.append(resonance_term(m, r"Q_x", True))
    if n:
        pieces.append(resonance_term(n, r"Q_z", not pieces))
    return "$" + "".join(pieces) + f"={p}$"


def generate_resonances(
    qx_limits: tuple[float, float],
    qz_limits: tuple[float, float],
    max_order: int,
) -> list[ResonanceLine]:
    """Generate unique m Qx + n Qz = p lines crossing the tune window."""
    xmin, xmax = qx_limits
    ymin, ymax = qz_limits
    corners = ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))
    unique: dict[tuple[int, int, int], ResonanceLine] = {}

    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            if not 1 <= abs(m) + abs(n) <= max_order:
                continue
            values = [m * x + n * y for x, y in corners]
            p_min = math.ceil(min(values) - 1e-12)
            p_max = math.floor(max(values) + 1e-12)
            for p in range(p_min, p_max + 1):
                cm, cn, cp = canonical_coefficients(m, n, p)
                order = abs(cm) + abs(cn)
                if order <= max_order:
                    unique[(cm, cn, cp)] = ResonanceLine(
                        cm, cn, cp, order, resonance_label(cm, cn, cp)
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
    """Clip one resonance line to the rectangular tune window."""
    xmin, xmax = qx_limits
    ymin, ymax = qz_limits
    candidates: list[tuple[float, float]] = []

    if line.n:
        for x in (xmin, xmax):
            y = (line.p - line.m * x) / line.n
            if ymin - 1e-10 <= y <= ymax + 1e-10:
                candidates.append((x, y))
    if line.m:
        for y in (ymin, ymax):
            x = (line.p - line.n * y) / line.m
            if xmin - 1e-10 <= x <= xmax + 1e-10:
                candidates.append((x, y))

    distinct: list[tuple[float, float]] = []
    for point in candidates:
        if not any(np.hypot(point[0] - x, point[1] - y) < 1e-9 for x, y in distinct):
            distinct.append(point)
    if len(distinct) < 2:
        return None
    return max(
        (
            (point_a, point_b)
            for index, point_a in enumerate(distinct)
            for point_b in distinct[index + 1 :]
        ),
        key=lambda pair: (pair[0][0] - pair[1][0]) ** 2
        + (pair[0][1] - pair[1][1]) ** 2,
    )


def is_highlighted(line: ResonanceLine) -> bool:
    return (line.m, line.n, line.p) in {(1, 0, 3), (0, 3, 8)}


def point_in_box(x: float, y: float, box: tuple[float, ...]) -> bool:
    xmin, xmax, ymin, ymax = box
    return xmin <= x <= xmax and ymin <= y <= ymax


def label_position(
    segment: tuple[tuple[float, float], tuple[float, float]],
    data_center: tuple[float, float],
    avoid_boxes: list[tuple[float, ...]],
    occupied: list[tuple[float, float]],
) -> tuple[float, float]:
    """Place a label near a line end and away from data and other labels."""
    (x0, y0), (x1, y1) = segment
    xspan = QX_LIMITS[1] - QX_LIMITS[0]
    yspan = QZ_LIMITS[1] - QZ_LIMITS[0]
    best = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    best_score = -np.inf

    for fraction in (0.15, 0.85, 0.24, 0.76, 0.34, 0.66):
        x = x0 + fraction * (x1 - x0)
        y = y0 + fraction * (y1 - y0)
        score = np.hypot(
            (x - data_center[0]) / xspan,
            (y - data_center[1]) / yspan,
        )
        if any(point_in_box(x, y, box) for box in avoid_boxes):
            score -= 8.0
        for old_x, old_y in occupied:
            distance = np.hypot((x - old_x) / xspan, (y - old_y) / yspan)
            if distance < 0.13:
                score -= 5.0 * (0.13 - distance) / 0.13
        edge_distance = min(
            (x - QX_LIMITS[0]) / xspan,
            (QX_LIMITS[1] - x) / xspan,
            (y - QZ_LIMITS[0]) / yspan,
            (QZ_LIMITS[1] - y) / yspan,
        )
        if edge_distance < 0.065:
            score -= 2.5 * (0.065 - edge_distance) / 0.065
        if score > best_score:
            best_score = score
            best = (x, y)
    return best


def label_rotation(
    axis: plt.Axes,
    segment: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    display = axis.transData.transform(np.asarray(segment))
    delta = display[1] - display[0]
    angle = math.degrees(math.atan2(delta[1], delta[0]))
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def draw_resonances(
    axis: plt.Axes,
    qx_data: np.ndarray,
    qz_data: np.ndarray,
    *,
    max_order: int,
    label_mode: str,
    line_color: str,
    text_color: str,
    line_style: str,
    line_width: float,
    line_alpha: float,
    label_fontsize: float,
) -> list[ResonanceLine]:
    """Draw independently generated resonance lines and labels."""
    if label_mode not in {"all", "low-order", "highlighted", "none"}:
        raise ValueError(f"unsupported label mode: {label_mode}")

    qx_data = np.asarray(qx_data, dtype=float)
    qz_data = np.asarray(qz_data, dtype=float)
    good = (
        np.isfinite(qx_data)
        & np.isfinite(qz_data)
        & (qx_data >= QX_LIMITS[0])
        & (qx_data <= QX_LIMITS[1])
        & (qz_data >= QZ_LIMITS[0])
        & (qz_data <= QZ_LIMITS[1])
    )
    qx_good = qx_data[good]
    qz_good = qz_data[good]
    if qx_good.size:
        data_center = (float(np.median(qx_good)), float(np.median(qz_good)))
        qx_low, qx_high = np.percentile(qx_good, [3.0, 97.0])
        qz_low, qz_high = np.percentile(qz_good, [3.0, 97.0])
    else:
        data_center = (0.5 * sum(QX_LIMITS), 0.5 * sum(QZ_LIMITS))
        qx_low = qx_high = data_center[0]
        qz_low = qz_high = data_center[1]

    xspan = QX_LIMITS[1] - QX_LIMITS[0]
    yspan = QZ_LIMITS[1] - QZ_LIMITS[0]
    axis.set_xlim(QX_LIMITS)
    axis.set_ylim(QZ_LIMITS)
    axis.figure.canvas.draw()
    avoid_boxes = [
        (
            qx_low - 0.025 * xspan,
            qx_high + 0.025 * xspan,
            qz_low - 0.025 * yspan,
            qz_high + 0.025 * yspan,
        )
    ]
    legend = axis.get_legend()
    if legend is not None:
        bbox = legend.get_window_extent(renderer=axis.figure.canvas.get_renderer())
        data_bbox = axis.transData.inverted().transform(
            np.asarray([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
        )
        avoid_boxes.append(
            (
                data_bbox[0, 0] - 0.015 * xspan,
                data_bbox[1, 0] + 0.015 * xspan,
                data_bbox[0, 1] - 0.015 * yspan,
                data_bbox[1, 1] + 0.015 * yspan,
            )
        )

    occupied: list[tuple[float, float]] = []
    lines = generate_resonances(QX_LIMITS, QZ_LIMITS, max_order)
    for line in lines:
        segment = line_segment_in_window(line, QX_LIMITS, QZ_LIMITS)
        if segment is None:
            continue
        (x0, y0), (x1, y1) = segment
        axis.plot(
            [x0, x1],
            [y0, y1],
            color=line_color,
            linestyle=line_style,
            linewidth=line_width,
            alpha=line_alpha,
            zorder=0,
        )
        should_label = (
            label_mode == "all"
            or (label_mode == "low-order" and (line.order <= 2 or is_highlighted(line)))
            or (label_mode == "highlighted" and is_highlighted(line))
        )
        if not should_label:
            continue
        position = label_position(segment, data_center, avoid_boxes, occupied)
        occupied.append(position)
        label = line.label
        if RESONANCE_BOLD_LABELS:
            label = r"$\mathbf{" + line.label.strip("$") + "}$"
        axis.text(
            *position,
            label,
            color=text_color,
            fontsize=label_fontsize,
            fontweight="bold",
            rotation=label_rotation(axis, segment),
            rotation_mode="anchor",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": RESONANCE_LABEL_BOX_ALPHA,
            },
            clip_on=True,
            zorder=RESONANCE_LABEL_ZORDER,
        )
    return lines


def add_references(axis: plt.Axes, star: tuple[float, float]) -> None:
    axis.axvline(
        QX_BARE,
        color=BARE_GUIDE_COLOR,
        linestyle=BARE_GUIDE_LINESTYLE,
        linewidth=BARE_GUIDE_LINEWIDTH,
        zorder=4,
    )
    axis.axhline(
        QZ_BARE,
        color=BARE_GUIDE_COLOR,
        linestyle=BARE_GUIDE_LINESTYLE,
        linewidth=BARE_GUIDE_LINEWIDTH,
        zorder=4,
    )
    axis.axvline(
        star[0],
        color=STAR_GUIDE_COLOR,
        linestyle=STAR_GUIDE_LINESTYLE,
        linewidth=STAR_GUIDE_LINEWIDTH,
        zorder=4,
    )
    axis.axhline(
        star[1],
        color=STAR_GUIDE_COLOR,
        linestyle=STAR_GUIDE_LINESTYLE,
        linewidth=STAR_GUIDE_LINEWIDTH,
        zorder=4,
    )
    axis.scatter(
        QX_BARE,
        QZ_BARE,
        marker="o",
        s=160,
        facecolor="white",
        edgecolor="blue",
        linewidth=2.8,
        label="Bare tune",
        zorder=12,
    )
    axis.scatter(
        *star,
        marker="*",
        s=300,
        color="red",
        edgecolor="black",
        linewidth=2.4,
        label="Max SC tune shift",
        zorder=13,
    )
    axis.set_xlim(QX_LIMITS)
    axis.set_ylim(QZ_LIMITS)
    legend = axis.legend(loc="upper left", fontsize=15, frameon=True)
    legend.set_zorder(30)


def make_plot(
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
) -> tuple[Path, ...]:
    off = np.load(off_path)
    on = np.load(on_path)
    datasets = (
        (off["qr_3d"], off["qz_3d"]),
        (off["qr_toy"], off["qz_toy"]),
        (on["new_qr_fft"], on["new_qz_fft"]),
        (on["theory_qr_fft"], on["theory_qz_fft"]),
    )
    histograms = [
        np.histogram2d(*datasets[index], bins=256, range=(QX_LIMITS, QZ_LIMITS))[0]
        for index in (0, 2)
    ]
    norm = LogNorm(vmin=1, vmax=max(histogram.max() for histogram in histograms))

    off_star = (
        tuple(off["common_star"])
        if OFF_RED_STAR_OVERRIDE is None
        else OFF_RED_STAR_OVERRIDE
    )
    on_star = (
        tuple(on["common_star"])
        if ON_RED_STAR_OVERRIDE is None
        else ON_RED_STAR_OVERRIDE
    )
    stars = (off_star, off_star, on_star, on_star)
    output_stems = (
        "longitudinal_sc_tune_3d_lsc_off_with_resonances",
        "longitudinal_sc_tune_fixed_reference_lsc_off_with_resonances",
        "longitudinal_sc_tune_3d_lsc_on_with_resonances",
        "longitudinal_sc_tune_fixed_reference_lsc_on_with_resonances",
    )
    histogram_by_index = {0: histograms[0], 2: histograms[1]}

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, dataset in enumerate(datasets):
        fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=(1.0, 0.055),
            wspace=0.08,
        )
        axis = fig.add_subplot(grid[0, 0])
        colorbar_axis = fig.add_subplot(grid[0, 1])
        if index in histogram_by_index:
            mesh = axis.pcolormesh(
                np.linspace(*QX_LIMITS, 257),
                np.linspace(*QZ_LIMITS, 257),
                histogram_by_index[index].T,
                norm=norm,
                cmap="viridis",
                shading="auto",
                rasterized=True,
                zorder=1,
            )
            colorbar = fig.colorbar(mesh, cax=colorbar_axis)
            colorbar.set_label("Counts", fontsize=19)
            colorbar.ax.tick_params(labelsize=12)
        else:
            axis.scatter(
                dataset[0],
                dataset[1],
                s=15,
                alpha=0.72,
                color="tab:purple",
                linewidths=0,
                rasterized=True,
                zorder=1,
            )
            # Reserve exactly the same colorbar space as the PIC panels while
            # keeping the theory panels free of a physically meaningless bar.
            placeholder = fig.colorbar(
                ScalarMappable(norm=norm, cmap="viridis"),
                cax=colorbar_axis,
            )
            placeholder.set_label("Counts", fontsize=19)
            placeholder.solids.set_alpha(0.0)
            placeholder.outline.set_edgecolor("white")
            placeholder.ax.yaxis.label.set_color("white")
            placeholder.ax.tick_params(
                which="both",
                labelsize=12,
                colors="white",
            )

        add_references(axis, stars[index])
        draw_resonances(
            axis,
            dataset[0],
            dataset[1],
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

        output = output_dir / output_stems[index]
        fig.savefig(output.with_suffix(".png"), dpi=300)
        fig.savefig(output.with_suffix(".pdf"))
        outputs.append(output)

    return tuple(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-npz", type=Path, default=DEFAULT_OFF)
    parser.add_argument("--on-npz", type=Path, default=DEFAULT_ON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-order",
        type=int,
        default=RESONANCE_MAX_ORDER,
        help="Maximum resonance order |m|+|n| (default: %(default)s).",
    )
    parser.add_argument(
        "--label-mode",
        choices=("all", "low-order", "highlighted", "none"),
        default=RESONANCE_LABEL_MODE,
        help="Which resonance lines receive text labels.",
    )
    parser.add_argument("--line-color", default=RESONANCE_LINE_COLOR)
    parser.add_argument("--text-color", default=RESONANCE_TEXT_COLOR)
    parser.add_argument(
        "--line-style",
        choices=("-", "--", "-.", ":"),
        default=RESONANCE_LINESTYLE,
    )
    parser.add_argument("--line-width", type=float, default=RESONANCE_LINEWIDTH)
    parser.add_argument("--line-alpha", type=float, default=RESONANCE_LINE_ALPHA)
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=RESONANCE_LABEL_FONTSIZE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_order < 1:
        raise ValueError("--max-order must be at least 1")
    if args.line_width < 0:
        raise ValueError("--line-width must be non-negative")
    if not 0.0 <= args.line_alpha <= 1.0:
        raise ValueError("--line-alpha must be between 0 and 1")
    if args.label_fontsize <= 0:
        raise ValueError("--label-fontsize must be positive")
    outputs = make_plot(
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
    print(f"Axes: Qx={QX_LIMITS}, Qz={QZ_LIMITS}")
    line_count = len(generate_resonances(QX_LIMITS, QZ_LIMITS, args.max_order))
    print(
        f"Resonances: {line_count} lines through order {args.max_order}; "
        f"labels={args.label_mode}"
    )
    plt.show()


if __name__ == "__main__":
    main()
