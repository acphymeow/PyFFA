"""Offline validation of gap-based h-bunch partition and core-bin lengths.

The tested construction is deliberately close to the proposed algorithm:

1. build one periodic histogram from the current particle coordinates;
2. find the h widest low-density gaps (all h RF buckets are assumed filled);
3. use the widest gap as the common branch cut and the other h-1 gaps as
   internal separators, producing h independent bunch intervals;
4. histogram each interval, rank non-empty bins by their smoothed density,
   and retain the densest bins
   carrying a prescribed fraction q of that bucket's charge;
5. take the left/right bin edges enclosing all retained bins, including any
   empty or rejected bins between them;

The script includes a stress case where the requested retained charge exceeds
the true core fraction.  That case is expected to show the principal limit of
the method: the enclosing arc can expand strongly when it is forced to retain
part of a diffuse, ring-wide halo.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SHOW_FIGURES = "--no-show" not in sys.argv
if not SHOW_FIGURES:
    os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np


SEED = 20260811
PERIOD = 360.0
BINS_PER_BUCKET = 256
SMOOTH_SIGMA_BINS = 1.5
PERIODIC_BINS = 2048
GAP_DENSITY_FRACTION = 0.03
MIN_GAP_BINS = 4
HERE = Path(__file__).resolve().parent
OUT_FIG = HERE / "core_bin_enclosing_arc_validation.png"
OUT_FLOW_FIG = HERE / "core_bin_enclosing_arc_workflow.png"
OUT_JSON = HERE / "core_bin_enclosing_arc_metrics.json"
OUT_SUMMARY = HERE / "core_bin_enclosing_arc_summary.txt"
OUT_BENCH = HERE / "core_bin_enclosing_arc_complexity.csv"
OUT_Q_FIG = HERE / "core_bin_enclosing_arc_q_sensitivity.png"
OUT_Q_CSV = HERE / "core_bin_enclosing_arc_q_sensitivity.csv"


@dataclass
class Case:
    name: str
    h: int
    angles: np.ndarray
    is_core: np.ndarray
    true_bucket: np.ndarray
    q: float
    description: str


def wrap_delta(values, center, period=PERIOD):
    return (np.asarray(values) - center + 0.5 * period) % period - 0.5 * period


def gaussian_smooth(values, sigma_bins=SMOOTH_SIGMA_BINS):
    half = max(2, int(np.ceil(4.0 * sigma_bins)))
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values.astype(float), kernel, mode="same")


def gaussian_smooth_periodic(values, sigma_bins=SMOOTH_SIGMA_BINS):
    """Gaussian smoothing with circular rather than zero-padded boundaries."""
    half = max(2, int(np.ceil(4.0 * sigma_bins)))
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    out = np.zeros_like(values, dtype=float)
    for shift, weight in zip(range(-half, half + 1), kernel):
        out += weight * np.roll(values, shift)
    return out


def circular_true_runs(mask):
    """Return inclusive true-bin runs, merging a run across 0/PERIOD."""
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    if np.all(mask):
        return [(0, n - 1)]
    if not np.any(mask):
        return []

    first_false = int(np.flatnonzero(~mask)[0])
    order = (first_false + 1 + np.arange(n)) % n
    runs = []
    start = previous = None
    for index in order:
        if mask[index]:
            if start is None:
                start = int(index)
            previous = int(index)
        elif start is not None:
            runs.append((start, previous))
            start = previous = None
    if start is not None:
        runs.append((start, previous))
    return runs


def find_h_gap_partition(angles, h):
    """Split a periodic distribution into h intervals using h density gaps."""
    if h < 1:
        raise ValueError("h must be at least one")
    counts, edges = np.histogram(
        np.mod(angles, PERIOD), bins=PERIODIC_BINS, range=(0.0, PERIOD)
    )
    density = gaussian_smooth_periodic(counts)
    threshold = GAP_DENSITY_FRACTION * float(np.max(density))
    low_density = density <= threshold
    bin_width = PERIOD / PERIODIC_BINS

    candidates = []
    for start_bin, end_bin in circular_true_runs(low_density):
        length_bins = (end_bin - start_bin) % PERIODIC_BINS + 1
        if length_bins < MIN_GAP_BINS:
            continue
        start = float(edges[start_bin])
        length = float(length_bins * bin_width)
        candidates.append({
            "start_deg": start,
            "length_deg": length,
            "midpoint_deg": float((start + 0.5 * length) % PERIOD),
            "start_bin": int(start_bin),
            "end_bin": int(end_bin),
        })

    if len(candidates) < h:
        raise ValueError(
            f"Only {len(candidates)} low-density gaps were found, but h={h}. "
            "The bunches are not sufficiently separated for this construction."
        )

    selected = sorted(
        candidates, key=lambda gap: gap["length_deg"], reverse=True
    )[:h]
    common_gap = selected[0]
    common_cut = common_gap["midpoint_deg"]
    internal_cuts = sorted(
        float((gap["midpoint_deg"] - common_cut) % PERIOD)
        for gap in selected[1:]
    )
    boundaries = [0.0, *internal_cuts, PERIOD]
    return {
        "counts": counts,
        "density": density,
        "edges": edges,
        "threshold": threshold,
        "candidate_gaps": candidates,
        "selected_gaps": selected,
        "common_cut_deg": float(common_cut),
        "internal_cuts_u_deg": internal_cuts,
        "interval_boundaries_u_deg": boundaries,
    }


def select_core_bin_envelope(counts, edges, q):
    """Return density-ranked bins and their continuous enclosing interval."""
    smooth = gaussian_smooth(counts)
    candidates = np.flatnonzero(counts > 0)
    if candidates.size == 0:
        return None

    # Primary key: decreasing smoothed density; secondary key: raw charge.
    order = candidates[np.lexsort((-counts[candidates], -smooth[candidates]))]
    target = q * counts.sum()
    stop = int(np.searchsorted(np.cumsum(counts[order]), target, side="left")) + 1
    selected = np.sort(order[:stop])
    left = int(selected[0])
    right = int(selected[-1])
    return {
        "left_bin": left,
        "right_bin": right,
        "left": float(edges[left]),
        "right": float(edges[right + 1]),
        "length": float(edges[right + 1] - edges[left]),
        "selected_bins": selected,
        "selected_charge_fraction": float(counts[selected].sum() / counts.sum()),
        "holes_inside_envelope": int(
            (right - left + 1) - np.count_nonzero(np.isin(np.arange(left, right + 1), selected))
        ),
        "smooth": smooth,
    }


def analyse_case(case):
    partition = find_h_gap_partition(case.angles, case.h)
    common_cut = partition["common_cut_deg"]
    boundaries = partition["interval_boundaries_u_deg"]
    u = (case.angles - common_cut) % PERIOD
    buckets = []
    used_true_labels = set()

    for interval_index, (left_u, right_u) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        if interval_index == len(boundaries) - 2:
            member = (u >= left_u) & (u <= right_u)
        else:
            member = (u >= left_u) & (u < right_u)
        core_member = member & case.is_core
        if not np.any(core_member):
            raise AssertionError(
                f"{case.name}: interval {interval_index} contains no core bunch"
            )

        labels, label_counts = np.unique(
            case.true_bucket[core_member], return_counts=True
        )
        labels = labels[labels >= 0]
        if labels.size == 0:
            raise AssertionError(
                f"{case.name}: interval {interval_index} contains only halo"
            )
        label_counts = np.array([
            np.count_nonzero(case.true_bucket[core_member] == label)
            for label in labels
        ])
        true_label = int(labels[np.argmax(label_counts)])
        if true_label in used_true_labels:
            raise AssertionError(
                f"{case.name}: true bunch {true_label} was split across intervals"
            )
        used_true_labels.add(true_label)

        edges = np.linspace(left_u, right_u, BINS_PER_BUCKET + 1)
        counts, _ = np.histogram(u[member], bins=edges)
        envelope = select_core_bin_envelope(counts, edges, case.q)
        if envelope is None:
            raise AssertionError(
                f"{case.name}: interval {interval_index} has no populated bins"
            )

        inside = (u >= envelope["left"]) & (u <= envelope["right"])
        true_core_k = case.is_core & (case.true_bucket == true_label)
        halo_k = (~case.is_core) & member
        core_coverage = float(np.mean(inside[true_core_k])) if np.any(true_core_k) else None
        halo_inclusion = float(np.mean(inside[halo_k])) if np.any(halo_k) else None

        oracle_values = np.sort(u[true_core_k])
        oracle_n = max(2, int(np.ceil(0.995 * len(oracle_values))))
        spans = oracle_values[oracle_n - 1:] - oracle_values[:len(oracle_values) - oracle_n + 1]
        oracle_length = float(np.min(spans)) if len(spans) else float("nan")

        buckets.append({
            "bucket": true_label,
            "interval_index": interval_index,
            "interval_left_u_deg": float(left_u),
            "interval_right_u_deg": float(right_u),
            "charge_fraction": float(member.sum() / len(case.angles)),
            "left_u_deg": envelope["left"],
            "right_u_deg": envelope["right"],
            "length_deg": envelope["length"],
            "selected_charge_fraction": envelope["selected_charge_fraction"],
            "selected_bins": int(len(envelope["selected_bins"])),
            "holes_inside_envelope": envelope["holes_inside_envelope"],
            "core_coverage": core_coverage,
            "halo_inclusion": halo_inclusion,
            "oracle_core_99p5_length_deg": oracle_length,
            "length_over_oracle": float(envelope["length"] / oracle_length),
            "selected_bin_indices": envelope["selected_bins"].tolist(),
        })

    arcs = []
    for b in buckets:
        start = (common_cut + b["left_u_deg"]) % PERIOD
        arcs.append((start, b["length_deg"]))
    if len(buckets) != case.h:
        raise AssertionError(
            f"{case.name}: expected h={case.h} bunch intervals, found {len(buckets)}"
        )
    return {
        "name": case.name,
        "description": case.description,
        "h": case.h,
        "q": case.q,
        "n_particles": int(len(case.angles)),
        "halo_fraction": float(np.mean(~case.is_core)),
        "occupied_buckets": len(buckets),
        "common_cut_deg": common_cut,
        "selected_gaps": partition["selected_gaps"],
        "internal_cuts_u_deg": partition["internal_cuts_u_deg"],
        "interval_boundaries_u_deg": boundaries,
        "gap_density_threshold": partition["threshold"],
        "buckets": buckets,
        "arcs": arcs,
    }


def add_arc_span(ax, start, length, color="#07833d"):
    end = start + length
    if end <= PERIOD:
        ax.axvspan(start, end, color=color, alpha=0.16, lw=0)
        ax.plot([start, end], [1.03, 1.03], transform=ax.get_xaxis_transform(),
                color=color, lw=3.0, clip_on=False)
    else:
        add_arc_span(ax, start, PERIOD - start, color)
        add_arc_span(ax, 0.0, end - PERIOD, color)


def make_cases(rng):
    cases = []

    def build(name, h, specs, halo_fraction, q, description):
        angles = []
        core_flags = []
        bucket_labels = []
        for bucket, offsets, sigmas, sizes in specs:
            for offset, sigma, size in zip(offsets, sigmas, sizes):
                center = bucket * PERIOD / h
                values = center + rng.normal(offset, sigma, size)
                angles.append(values % PERIOD)
                core_flags.append(np.ones(size, dtype=bool))
                bucket_labels.append(np.full(size, bucket, dtype=int))
        n_core = sum(len(a) for a in angles)
        n_halo = int(round(n_core * halo_fraction / (1.0 - halo_fraction)))
        if n_halo:
            angles.append(rng.uniform(0.0, PERIOD, n_halo))
            core_flags.append(np.zeros(n_halo, dtype=bool))
            bucket_labels.append(np.full(n_halo, -1, dtype=int))
        cases.append(Case(name, h, np.concatenate(angles),
                          np.concatenate(core_flags), np.concatenate(bucket_labels),
                          q, description))

    build("h1_wrap", 1, [(0, [0], [18], [16000])], 0.005, 0.995,
          "One bunch crossing 0/2pi; 0.5% uniform halo.")
    build("h2_unequal", 2,
          [(0, [8], [15], [11000]), (1, [-9], [20], [7500])],
          0.005, 0.995, "Two unequal occupied RF buckets; 0.5% halo.")
    build("h3_internal_valley", 3,
          [(0, [5], [14], [10000]),
           (1, [-14, 14], [6, 6], [5000, 4500]),
           (2, [-7], [17], [8000])],
          0.005, 0.995,
          "All three buckets occupied; one bunch has an internal density valley.")
    build("h2_halo_stress", 2,
          [(0, [5], [15], [12000]), (1, [-5], [17], [9000])],
          0.020, 0.995,
          "Stress test: 2% ring-wide halo but q=99.5% forces halo retention.")
    return cases


def plot_cases(cases, results):
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.8), sharex=True)
    for ax, case, result in zip(axes.flat, cases, results):
        core = case.angles[case.is_core]
        halo = case.angles[~case.is_core]
        bins = np.linspace(0.0, PERIOD, 361)
        ax.hist(core, bins=bins, color="#3565f3", alpha=0.82,
                histtype="stepfilled", label="true core")
        if len(halo):
            ax.hist(halo, bins=bins, color="0.55", alpha=0.75,
                    histtype="step", lw=1.0, label="halo")
        for start, length in result["arcs"]:
            add_arc_span(ax, start, length)
        ax.axvline(result["common_cut_deg"], color="#c73832", ls="--", lw=1.2)
        for gap in result["selected_gaps"][1:]:
            ax.axvline(gap["midpoint_deg"], color="#d18b28", ls=":", lw=1.0)
        ratios = [b["length_over_oracle"] for b in result["buckets"]]
        cover = [100.0 * b["core_coverage"] for b in result["buckets"]]
        ax.text(0.015, 0.95,
                f"q={case.q:.3f}; occupied={result['occupied_buckets']}\n"
                f"core coverage min={min(cover):.2f}%\n"
                f"length/oracle max={max(ratios):.2f}x",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.82, edgecolor="none"))
        ax.set_title(case.name.replace("_", " "), fontsize=10)
        ax.set_xlim(0, PERIOD)
        ax.set_ylabel("particles / 1 deg")
        ax.grid(alpha=0.15)
    for ax in axes[-1, :]:
        ax.set_xlabel("periodic longitudinal coordinate (deg)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Dynamic h-gap partition and halo-robust bunch lengths",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_FIG, dpi=210)
    if not SHOW_FIGURES:
        plt.close(fig)


def workflow_interval_details(case, partition):
    """Expose the exact per-interval bin masks used by analyse_case()."""
    common_cut = partition["common_cut_deg"]
    boundaries = partition["interval_boundaries_u_deg"]
    u = (case.angles - common_cut) % PERIOD
    details = []
    for interval_index, (left_u, right_u) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        if interval_index == len(boundaries) - 2:
            member = (u >= left_u) & (u <= right_u)
        else:
            member = (u >= left_u) & (u < right_u)
        edges = np.linspace(left_u, right_u, BINS_PER_BUCKET + 1)
        counts, _ = np.histogram(u[member], bins=edges)
        envelope = select_core_bin_envelope(counts, edges, case.q)
        if envelope is None:
            continue
        selected_mask = np.zeros_like(counts, dtype=bool)
        selected_mask[envelope["selected_bins"]] = True
        details.append({
            "index": interval_index,
            "left": left_u,
            "right": right_u,
            "edges": edges,
            "centers": 0.5 * (edges[:-1] + edges[1:]),
            "counts": counts,
            "selected": selected_mask,
            "halo": (counts > 0) & ~selected_mask,
            "empty": counts == 0,
            "envelope": envelope,
        })
    return u, details


def plot_workflow(case):
    """Four-step view of the actual h-gap and core-bin calculation."""
    partition = find_h_gap_partition(case.angles, case.h)
    u, details = workflow_interval_details(case, partition)
    common_cut = partition["common_cut_deg"]
    boundaries = partition["interval_boundaries_u_deg"]

    blue = "#3565f3"
    blue_2 = "#6d8fdc"
    green = "#07833d"
    gray = "#8f8f8f"
    cut = "#c73832"
    boundary_color = "#d18b28"
    interval_colors = [blue, blue_2]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))

    # (a) The periodic histogram and the h selected low-density gaps.
    ax = axes[0, 0]
    edges = partition["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = partition["density"]
    ax.plot(centers, density, color=blue, lw=1.25, label="smoothed density")
    ax.axhline(partition["threshold"], color=gray, ls="--", lw=1.0,
               label="3% of peak")
    for gap_index, gap in enumerate(partition["selected_gaps"]):
        start = gap["start_deg"]
        end = start + gap["length_deg"]
        spans = [(start, min(end, PERIOD))]
        if end > PERIOD:
            spans.append((0.0, end - PERIOD))
        for span_left, span_right in spans:
            ax.axvspan(span_left, span_right, color=gray, alpha=0.18, lw=0)
        label = "widest gap\n(common cut)" if gap_index == 0 else "second gap\n(bunch boundary)"
        ax.annotate(label, xy=(gap["midpoint_deg"], partition["threshold"]),
                    xytext=(gap["midpoint_deg"], density.max() * (0.52 if gap_index == 0 else 0.32)),
                    ha="center", va="center", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=gray, lw=0.8))
    ax.set_title("(a) Find the two widest low-density gaps")
    ax.set_ylabel("smoothed particles / bin")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # (b) Shift the common cut to u=0 and obtain two coarse intervals.
    ax = axes[0, 1]
    u_counts, u_edges = np.histogram(u, bins=256, range=(0.0, PERIOD))
    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    ax.fill_between(u_centers, u_counts, step="mid", color=blue, alpha=0.76)
    for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ax.axvspan(left, right, color=interval_colors[i % len(interval_colors)],
                   alpha=0.07, lw=0)
        ax.text(0.5 * (left + right), u_counts.max() * 0.88,
                f"coarse bunch interval {i + 1}", ha="center", fontsize=8,
                color=interval_colors[i % len(interval_colors)])
    ax.axvline(0.0, color=cut, ls="--", lw=1.2)
    ax.axvline(PERIOD, color=cut, ls="--", lw=1.2)
    for internal in boundaries[1:-1]:
        ax.axvline(internal, color=boundary_color, ls="--", lw=1.2)
    ax.text(7.0, u_counts.max() * 0.97, "common cut", color=cut, fontsize=8)
    ax.set_title("(b) Move the common cut to 0 and split the bunches")
    ax.set_ylabel("particles / bin")

    # (c) Apply the qQ_k selection independently inside each coarse interval.
    ax = axes[1, 0]
    ymax = max(float(detail["counts"].max()) for detail in details)
    for detail in details:
        widths = np.diff(detail["edges"])
        ax.bar(detail["centers"][detail["selected"]],
               detail["counts"][detail["selected"]],
               width=0.94 * widths[detail["selected"]], color=blue,
               edgecolor="none")
        ax.bar(detail["centers"][detail["halo"]],
               detail["counts"][detail["halo"]],
               width=0.94 * widths[detail["halo"]], color=gray,
               alpha=0.75, edgecolor="none")
    for internal in boundaries[1:-1]:
        ax.axvline(internal, color=boundary_color, ls="--", lw=1.0)
    ax.text(0.02, 0.95, "blue: bunch bins\ngray: halo bins\nblank: empty bins",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none"))
    ax.set_ylim(0.0, ymax * 1.12)
    ax.set_title(r"(c) In each interval, retain bins carrying $qQ_k$")
    ax.set_ylabel("particles / bin")

    # (d) The requested output: h occupied bunch intervals.
    ax = axes[1, 1]
    for i, detail in enumerate(details):
        envelope = detail["envelope"]
        local_width = detail["edges"][1] - detail["edges"][0]
        ax.bar(detail["centers"], detail["counts"],
               width=0.94 * local_width, color="#c8c8c8", edgecolor="none",
               alpha=0.65)
        selected_centers = detail["centers"][detail["selected"]]
        selected_counts = detail["counts"][detail["selected"]]
        ax.bar(selected_centers, selected_counts,
               width=0.94 * local_width,
               color=interval_colors[i % len(interval_colors)], edgecolor="none",
               alpha=0.86)
        y = u_counts.max() * (0.78 - 0.12 * i)
        ax.plot([envelope["left"], envelope["right"]], [y, y],
                color=green, lw=3.0)
        ax.plot([envelope["left"], envelope["left"]], [y * 0.95, y * 1.05],
                color=green, lw=1.2)
        ax.plot([envelope["right"], envelope["right"]], [y * 0.95, y * 1.05],
                color=green, lw=1.2)
        ax.text(0.5 * (envelope["left"] + envelope["right"]), y * 1.07,
                rf"occupied interval {i + 1}: $L_{i + 1}$",
                color=green, fontsize=8, ha="center")
    for internal in boundaries[1:-1]:
        ax.axvline(internal, color=boundary_color, ls="--", lw=1.0)
    ax.set_title(r"(d) Output: $h=2$ occupied bunch intervals")
    ax.set_ylabel("particles / bin")

    for ax in axes.flat:
        ax.set_xlim(0.0, PERIOD)
        ax.set_xticks([0.0, 90.0, 180.0, 270.0, PERIOD])
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        ax.set_xlabel("periodic longitudinal coordinate")
        ax.grid(axis="y", alpha=0.14)
    fig.suptitle("Step-by-step view of test_core_bin_enclosing_arc.py", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_FLOW_FIG, dpi=220)
    if not SHOW_FIGURES:
        plt.close(fig)


def benchmark(rng):
    rows = []
    for n_bins in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        x = np.linspace(-4.0, 4.0, n_bins)
        mean = 2.0 + 120.0 * np.exp(-0.5 * x * x)
        counts = rng.poisson(mean).astype(int)
        edges = np.linspace(-90.0, 90.0, n_bins + 1)
        repeats = max(12, int(150000 / n_bins))
        t0 = time.perf_counter()
        for _ in range(repeats):
            select_core_bin_envelope(counts, edges, 0.995)
        elapsed = (time.perf_counter() - t0) / repeats
        rows.append({"n_bins": n_bins, "seconds_per_call": elapsed,
                     "repeats": repeats})
    with OUT_BENCH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def q_sensitivity(case):
    rows = []
    for q in [0.970, 0.980, 0.985, 0.990, 0.995, 0.999]:
        trial = Case(case.name, case.h, case.angles, case.is_core,
                     case.true_bucket, q, case.description)
        result = analyse_case(trial)
        rows.append({
            "q": q,
            "max_length_over_oracle": max(
                b["length_over_oracle"] for b in result["buckets"]
            ),
            "min_core_coverage": min(
                b["core_coverage"] for b in result["buckets"]
            ),
            "max_halo_inclusion": max(
                b["halo_inclusion"] for b in result["buckets"]
                if b["halo_inclusion"] is not None
            ),
        })
    with OUT_Q_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    q = np.array([row["q"] for row in rows])
    ratio = np.array([row["max_length_over_oracle"] for row in rows])
    coverage = 100.0 * np.array([row["min_core_coverage"] for row in rows])
    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    ax1.plot(100.0 * q, ratio, "o-", color="#07833d",
             label="max envelope/oracle")
    ax1.axvline(98.0, color="0.45", ls="--", lw=1.0,
                label="true core fraction")
    ax1.set_xlabel("retained total charge q (%)")
    ax1.set_ylabel("max envelope length / oracle", color="#07833d")
    ax1.tick_params(axis="y", labelcolor="#07833d")
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(100.0 * q, coverage, "s--", color="#3565f3",
             label="min true-core coverage")
    ax2.set_ylabel("minimum true-core coverage (%)", color="#3565f3")
    ax2.tick_params(axis="y", labelcolor="#3565f3")
    handles = ax1.get_lines() + ax2.get_lines()
    ax1.legend(handles, [line.get_label() for line in handles],
               loc="upper left", fontsize=8)
    ax1.set_title("2% diffuse-halo stress case")
    fig.tight_layout()
    fig.savefig(OUT_Q_FIG, dpi=210)
    if not SHOW_FIGURES:
        plt.close(fig)
    return rows


def main():
    rng = np.random.default_rng(SEED)
    cases = make_cases(rng)
    results = [analyse_case(case) for case in cases]
    timings = benchmark(rng)
    q_scan = q_sensitivity(cases[-1])
    plot_cases(cases, results)
    plot_workflow(cases[1])
    if SHOW_FIGURES:
        print("Displaying figures; close the plot windows to finish.")
        plt.show()
        plt.close("all")

    payload = {
        "method": (
            "find h low-density gaps; use the widest as the common cut and "
            "the other h-1 as separators; then apply density-ranked core-bin envelopes"
        ),
        "scope": "all h RF buckets are occupied and separated by detectable density gaps",
        "periodic_bins": PERIODIC_BINS,
        "gap_density_fraction": GAP_DENSITY_FRACTION,
        "bins_per_bucket": BINS_PER_BUCKET,
        "smooth_sigma_bins": SMOOTH_SIGMA_BINS,
        "cases": results,
        "complexity_benchmark": timings,
        "stress_case_q_sensitivity": q_scan,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = []
    for result in results:
        lines.append(
            f"{result['name']}: h={result['h']}, q={result['q']:.4f}, "
            f"halo={100*result['halo_fraction']:.3f}%, "
            f"occupied={result['occupied_buckets']}, "
            f"common_cut={result['common_cut_deg']:.3f} deg, "
            f"gap_widths={[round(g['length_deg'], 3) for g in result['selected_gaps']]}"
        )
        for b in result["buckets"]:
            lines.append(
                f"  bucket {b['bucket']}: L={b['length_deg']:.3f} deg, "
                f"core coverage={100*b['core_coverage']:.3f}%, "
                f"halo inside={100*b['halo_inclusion']:.3f}%, "
                f"L/oracle={b['length_over_oracle']:.3f}, "
                f"internal rejected/empty bins={b['holes_inside_envelope']}"
            )
    lines.append("")
    lines.append("Complexity: periodic binning O(N), circular smoothing O(B*w), "
                 "gap scan O(B), density sorting O(B log B), envelope O(B), memory O(B).")
    lines.append("The stress case is intentionally adverse: q exceeds the true "
                 "core fraction, so the envelope may expand into diffuse halo.")
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"saved -> {OUT_FIG}")
    print(f"saved -> {OUT_FLOW_FIG}")
    print(f"saved -> {OUT_JSON}")
    print(f"saved -> {OUT_SUMMARY}")
    print(f"saved -> {OUT_BENCH}")
    print(f"saved -> {OUT_Q_FIG}")
    print(f"saved -> {OUT_Q_CSV}")


if __name__ == "__main__":
    main()
