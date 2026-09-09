"""Paper-style tune-resonance overlay shared by the Comment 5 figures."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from plot_resonance_lines_order3 import (
    generate_resonances,
    is_reviewer_highlight,
    line_segment_in_window,
)


LINE_COLOR = "#c7c7c7"
TEXT_COLOR = "#686868"
LINEWIDTH = 0.65
LINE_ALPHA = 0.42
LABEL_FONTSIZE = 9.5


def _point_in_box(x: float, y: float, box: tuple[float, ...]) -> bool:
    xmin, xmax, ymin, ymax = box
    return xmin <= x <= xmax and ymin <= y <= ymax


def _label_position(
    segment: tuple[tuple[float, float], tuple[float, float]],
    data_center: tuple[float, float],
    avoid_boxes: list[tuple[float, ...]],
    occupied: list[tuple[float, float]],
    qx_limits: tuple[float, float],
    qz_limits: tuple[float, float],
) -> tuple[float, float]:
    """Put labels near line ends, away from data, legends, and old labels."""
    (x0, y0), (x1, y1) = segment
    xspan = qx_limits[1] - qx_limits[0]
    yspan = qz_limits[1] - qz_limits[0]
    best = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    best_score = -np.inf

    for fraction in (0.15, 0.85, 0.24, 0.76, 0.34, 0.66):
        x = x0 + fraction * (x1 - x0)
        y = y0 + fraction * (y1 - y0)
        score = np.hypot(
            (x - data_center[0]) / xspan,
            (y - data_center[1]) / yspan,
        )
        if any(_point_in_box(x, y, box) for box in avoid_boxes):
            score -= 8.0
        for old_x, old_y in occupied:
            distance = np.hypot((x - old_x) / xspan, (y - old_y) / yspan)
            if distance < 0.13:
                score -= 5.0 * (0.13 - distance) / 0.13
        edge_distance = min(
            (x - qx_limits[0]) / xspan,
            (qx_limits[1] - x) / xspan,
            (y - qz_limits[0]) / yspan,
            (qz_limits[1] - y) / yspan,
        )
        if edge_distance < 0.065:
            score -= 2.5 * (0.065 - edge_distance) / 0.065
        if score > best_score:
            best_score = score
            best = (x, y)
    return best


def _rotation(
    axis: plt.Axes,
    segment: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    display = axis.transData.transform(np.asarray(segment))
    delta = display[1] - display[0]
    angle = np.degrees(np.arctan2(delta[1], delta[0]))
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return float(angle)


def draw_paper_resonances(
    axis: plt.Axes,
    qx_data: np.ndarray,
    qz_data: np.ndarray,
    qx_limits: tuple[float, float],
    qz_limits: tuple[float, float],
    *,
    max_order: int = 3,
    label_mode: str = "all",
    line_color: str = LINE_COLOR,
    text_color: str = TEXT_COLOR,
    linewidth: float = LINEWIDTH,
    line_alpha: float = LINE_ALPHA,
    label_fontsize: float = LABEL_FONTSIZE,
    label_zorder: float = 10,
    bold_math_labels: bool = False,
    text_alpha: float = 1.0,
    label_fontweight: str = "bold",
    label_box_alpha: float = 0.68,
    extra_label_keys: set[tuple[int, int, int]] | None = None,
) -> None:
    """Draw light resonance lines without obscuring the tune footprint.

    ``label_mode`` may be ``all``, ``low-order`` (first and second order plus
    the reviewer-highlighted third-order line), ``highlighted`` (only Qx=3
    and 3Qz=8), or ``none``.  All resonance lines are drawn in every mode.
    """
    if label_mode not in {"all", "low-order", "highlighted", "none"}:
        raise ValueError(f"unsupported label mode: {label_mode}")

    qx_data = np.asarray(qx_data, dtype=float)
    qz_data = np.asarray(qz_data, dtype=float)
    good = (
        np.isfinite(qx_data)
        & np.isfinite(qz_data)
        & (qx_data >= qx_limits[0])
        & (qx_data <= qx_limits[1])
        & (qz_data >= qz_limits[0])
        & (qz_data <= qz_limits[1])
    )
    qx_good = qx_data[good]
    qz_good = qz_data[good]
    if qx_good.size:
        data_center = (float(np.median(qx_good)), float(np.median(qz_good)))
        qx_low, qx_high = np.percentile(qx_good, [3.0, 97.0])
        qz_low, qz_high = np.percentile(qz_good, [3.0, 97.0])
    else:
        data_center = (
            0.5 * sum(qx_limits),
            0.5 * sum(qz_limits),
        )
        qx_low = qx_high = data_center[0]
        qz_low = qz_high = data_center[1]

    xspan = qx_limits[1] - qx_limits[0]
    yspan = qz_limits[1] - qz_limits[0]
    axis.set_xlim(qx_limits)
    axis.set_ylim(qz_limits)
    axis.figure.canvas.draw()

    avoid_boxes: list[tuple[float, ...]] = [
        (
            qx_low - 0.025 * xspan,
            qx_high + 0.025 * xspan,
            qz_low - 0.025 * yspan,
            qz_high + 0.025 * yspan,
        )
    ]
    legend = axis.get_legend()
    if legend is not None:
        renderer = axis.figure.canvas.get_renderer()
        bbox = legend.get_window_extent(renderer=renderer)
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
    for line in generate_resonances(qx_limits, qz_limits, max_order=max_order):
        segment = line_segment_in_window(line, qx_limits, qz_limits)
        if segment is None:
            continue
        (x0, y0), (x1, y1) = segment
        axis.plot(
            [x0, x1],
            [y0, y1],
            color=line_color,
            linewidth=linewidth,
            alpha=line_alpha,
            zorder=3,
        )

        should_label = (
            label_mode == "all"
            or (label_mode == "low-order" and (
                line.order <= 2 or is_reviewer_highlight(line)
            ))
            or (label_mode == "highlighted" and is_reviewer_highlight(line))
            or (
                extra_label_keys is not None
                and (line.m, line.n, line.p) in extra_label_keys
            )
        )
        if not should_label:
            continue
        position = _label_position(
            segment,
            data_center,
            avoid_boxes,
            occupied,
            qx_limits,
            qz_limits,
        )
        occupied.append(position)
        label = line.label
        if bold_math_labels:
            label = r"$\mathbf{" + line.label.strip("$") + "}$"
        axis.text(
            position[0],
            position[1],
            label,
            color=text_color,
            alpha=text_alpha,
            fontsize=label_fontsize,
            fontweight=label_fontweight,
            rotation=_rotation(axis, segment),
            rotation_mode="anchor",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": label_box_alpha,
            },
            clip_on=True,
            zorder=label_zorder,
        )
