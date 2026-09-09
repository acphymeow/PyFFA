"""Offline validation of density-support longitudinal unwrapping for h > 1.

The test is intentionally independent of the tracking and MPI code.  It
generates synthetic multi-bunch distributions on a ring, adds a diffuse
longitudinal halo, identifies occupied arcs from a periodically smoothed line
density, and unwraps every bunch into its own continuous coordinate.

Outputs are written next to this script:
  density_support_harmonics.png
  density_support_metrics.json
  density_support_summary.txt
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# CONFIG -- edit here
# ================================================================

CIRCUMFERENCE_M = 71.53
NBINS = 1024
SMOOTH_SIGMA_BINS = 2.0
DENSITY_THRESHOLD = 1.0e-2
NS_PER_ARC = 128
N_CORE = 240_000
SEED = 20260811
SAVE_FIGS = True
SHOW_FIGS = False

# Acceptance criteria for this deterministic offline test.
MIN_CORE_COVERAGE = 0.990
MAX_COORD_RMSE_M = 1.0e-10

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Case:
    name: str
    harmonic: int
    filled_buckets: tuple[int, ...]
    offsets_m: tuple[float, ...]
    sigmas_m: tuple[float, ...]
    fractions: tuple[float, ...]
    halo_fraction: float


CASES = (
    Case(
        name="h2_equal_wrap_halo0p5",
        harmonic=2,
        filled_buckets=(0, 1),
        offsets_m=(-0.45, 0.70),
        sigmas_m=(1.25, 1.25),
        fractions=(0.5, 0.5),
        halo_fraction=0.005,
    ),
    Case(
        name="h3_unequal_halo1",
        harmonic=3,
        filled_buckets=(0, 1, 2),
        offsets_m=(-0.60, 1.10, -0.35),
        sigmas_m=(1.00, 1.55, 0.80),
        fractions=(0.50, 0.30, 0.20),
        halo_fraction=0.010,
    ),
    Case(
        name="h4_partial_two_filled_halo2",
        harmonic=4,
        filled_buckets=(0, 3),
        offsets_m=(-0.40, 0.85),
        sigmas_m=(1.15, 1.45),
        fractions=(0.65, 0.35),
        halo_fraction=0.020,
    ),
)


def circular_delta(a, b, period):
    """Signed shortest displacement a-b on a circle."""
    return np.mod(a - b + 0.5 * period, period) - 0.5 * period


def smooth_periodic(values, sigma_bins):
    half = max(2, int(np.ceil(4.0 * sigma_bins)))
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    out = np.zeros_like(values, dtype=np.float64)
    for shift, weight in zip(range(-half, half + 1), kernel):
        out += weight * np.roll(values, shift)
    return out


def circular_true_runs(mask):
    """Return inclusive (start_bin, end_bin) runs, merging the wrap run."""
    n = len(mask)
    if np.all(mask):
        return [(0, n - 1)]
    if not np.any(mask):
        return []

    # Start immediately after an invalid bin.  A wrap-around component is then
    # encountered as one linear run whose end index may be smaller than start.
    invalid = int(np.flatnonzero(~mask)[0])
    order = (invalid + 1 + np.arange(n)) % n
    runs = []
    start = None
    previous = None
    for index in order:
        if mask[index]:
            if start is None:
                start = int(index)
            previous = int(index)
        elif start is not None:
            runs.append((start, previous))
            start = None
            previous = None
    if start is not None:
        runs.append((start, previous))
    return runs


def identify_density_arcs(s_mod, period):
    counts, edges = np.histogram(s_mod, bins=NBINS, range=(0.0, period))
    density = smooth_periodic(counts.astype(np.float64), SMOOTH_SIGMA_BINS)
    threshold = DENSITY_THRESHOLD * np.max(density)
    valid = density > threshold
    bw = period / NBINS
    arcs = []
    for start_bin, end_bin in circular_true_runs(valid):
        length_bins = (end_bin - start_bin) % NBINS + 1
        start = edges[start_bin]
        length = length_bins * bw
        end = np.mod(start + length, period)
        center = np.mod(start + 0.5 * length, period)
        arcs.append(
            {
                "start_m": float(start),
                "end_m": float(end),
                "length_m": float(length),
                "center_m": float(center),
                "wraps": bool(end < start),
                "start_bin": int(start_bin),
                "end_bin": int(end_bin),
            }
        )
    arcs.sort(key=lambda item: item["start_m"])
    return arcs, counts, density, threshold


def four_shift_span(values, period):
    spans = []
    for shift in (0.0, 0.25 * period, 0.5 * period, 0.75 * period):
        shifted = np.mod(values + shift, period)
        spans.append(float(np.max(shifted) - np.min(shifted)))
    return min(spans)


def generate_case(case, rng):
    period_bucket = CIRCUMFERENCE_M / case.harmonic
    n_halo = int(round(N_CORE * case.halo_fraction))
    raw = np.asarray(case.fractions, dtype=np.float64)
    raw /= raw.sum()
    numbers = np.floor(raw * N_CORE).astype(int)
    numbers[-1] += N_CORE - int(numbers.sum())

    coordinates = []
    true_local = []
    labels = []
    centers = []
    for label, (bucket, offset, sigma, number) in enumerate(
        zip(case.filled_buckets, case.offsets_m, case.sigmas_m, numbers)
    ):
        center = np.mod(bucket * period_bucket + offset, CIRCUMFERENCE_M)
        local = rng.normal(0.0, sigma, int(number))
        coordinates.append(np.mod(center + local, CIRCUMFERENCE_M))
        true_local.append(local)
        labels.append(np.full(int(number), label, dtype=np.int32))
        centers.append(center)

    core_s = np.concatenate(coordinates)
    core_local = np.concatenate(true_local)
    core_label = np.concatenate(labels)
    halo_s = rng.uniform(0.0, CIRCUMFERENCE_M, n_halo)
    all_s = np.concatenate((core_s, halo_s))
    return {
        "all_s": all_s,
        "core_s": core_s,
        "core_local": core_local,
        "core_label": core_label,
        "centers": np.asarray(centers),
        "n_halo": n_halo,
    }


def match_arcs_to_bunches(arcs, centers, period):
    """One-to-one minimum-distance matching; case sizes are deliberately small."""
    unmatched = set(range(len(arcs)))
    mapping = []
    for center in centers:
        index = min(
            unmatched,
            key=lambda j: abs(circular_delta(arcs[j]["center_m"], center, period)),
        )
        mapping.append(index)
        unmatched.remove(index)
    return mapping


def analyze_case(case, rng):
    generated = generate_case(case, rng)
    all_s = generated["all_s"]
    core_s = generated["core_s"]
    true_local = generated["core_local"]
    labels = generated["core_label"]
    centers = generated["centers"]

    arcs, counts, density, threshold = identify_density_arcs(
        all_s, CIRCUMFERENCE_M
    )
    if len(arcs) != len(case.filled_buckets):
        raise AssertionError(
            f"{case.name}: expected {len(case.filled_buckets)} occupied arcs, "
            f"found {len(arcs)}"
        )
    mapping = match_arcs_to_bunches(arcs, centers, CIRCUMFERENCE_M)

    bunch_metrics = []
    recovered_by_label = []
    for label, arc_index in enumerate(mapping):
        arc = arcs[arc_index]
        mask = labels == label
        s = core_s[mask]
        truth = true_local[mask]

        # The detected empty region provides the branch cut.  Subtracting the
        # known RF-bucket center only fixes an arbitrary origin; continuity is
        # supplied by the detected arc start and modulo operation.
        unfolded = np.mod(s - arc["start_m"], CIRCUMFERENCE_M)
        center_unfolded = np.mod(centers[label] - arc["start_m"], CIRCUMFERENCE_M)
        recovered = unfolded - center_unfolded
        recovered = circular_delta(recovered, 0.0, CIRCUMFERENCE_M)

        inside = (
            np.mod(s - arc["start_m"], CIRCUMFERENCE_M)
            <= arc["length_m"]
        )
        error = recovered - truth
        rmse = float(np.sqrt(np.mean(error * error)))
        max_error = float(np.max(np.abs(error)))
        coverage = float(np.mean(inside))
        if rmse > MAX_COORD_RMSE_M:
            raise AssertionError(f"{case.name}/bunch {label}: RMSE={rmse}")
        if coverage < MIN_CORE_COVERAGE:
            raise AssertionError(
                f"{case.name}/bunch {label}: core coverage={coverage:.5f}"
            )
        recovered_by_label.append((truth, recovered))
        bunch_metrics.append(
            {
                "label": label,
                "rf_bucket": int(case.filled_buckets[label]),
                "true_center_m": float(centers[label]),
                "detected_arc_index": int(arc_index),
                "arc_start_m": arc["start_m"],
                "arc_end_m": arc["end_m"],
                "arc_length_m": arc["length_m"],
                "arc_wraps_boundary": arc["wraps"],
                "core_coverage": coverage,
                "coordinate_rmse_m": rmse,
                "coordinate_max_error_m": max_error,
                "true_sigma_m": float(np.std(truth)),
                "recovered_sigma_m": float(np.std(recovered)),
                "slice_step_per_arc_m": arc["length_m"] / (NS_PER_ARC - 1),
            }
        )

    single_span = four_shift_span(all_s, CIRCUMFERENCE_M)
    metrics = {
        "case": case.name,
        "harmonic": case.harmonic,
        "expected_occupied_arcs": len(case.filled_buckets),
        "detected_occupied_arcs": len(arcs),
        "halo_fraction": case.halo_fraction,
        "n_core": len(core_s),
        "n_halo": generated["n_halo"],
        "legacy_single_arc_span_m": single_span,
        "legacy_single_arc_slice_step_m": single_span / (NS_PER_ARC - 1),
        "mean_density_arc_slice_step_m": float(
            np.mean([item["slice_step_per_arc_m"] for item in bunch_metrics])
        ),
        "resolution_gain": float(
            (single_span / (NS_PER_ARC - 1))
            / np.mean([item["slice_step_per_arc_m"] for item in bunch_metrics])
        ),
        "bunches": bunch_metrics,
    }
    plotting = {
        "all_s": all_s,
        "core_s": core_s,
        "labels": labels,
        "arcs": arcs,
        "counts": counts,
        "density": density,
        "threshold": threshold,
        "recovered": recovered_by_label,
    }
    return metrics, plotting


def shade_arc(ax, arc, period, **kwargs):
    start = arc["start_m"]
    end_linear = start + arc["length_m"]
    if end_linear <= period:
        ax.axvspan(start, end_linear, **kwargs)
    else:
        ax.axvspan(start, period, **kwargs)
        ax.axvspan(0.0, end_linear - period, **kwargs)


def make_figure(results):
    fig, axes = plt.subplots(
        len(results), 2, figsize=(12.8, 3.7 * len(results)), squeeze=False
    )
    centers = (np.arange(NBINS) + 0.5) * CIRCUMFERENCE_M / NBINS
    colors = plt.cm.tab10.colors

    for row, (case, metrics, data) in enumerate(results):
        ax = axes[row, 0]
        ax.plot(centers, data["density"], color="k", lw=1.2,
                label="periodically smoothed density")
        ax.axhline(data["threshold"], color="C3", ls="--", lw=1.1,
                   label=r"$\epsilon\lambda_{max}$")
        for index, arc in enumerate(data["arcs"]):
            shade_arc(ax, arc, CIRCUMFERENCE_M, color=colors[index], alpha=0.18)
        ax.set_xlim(0.0, CIRCUMFERENCE_M)
        ax.set_xlabel(r"periodic coordinate $s_{mod}$ [m]")
        ax.set_ylabel("smoothed bin counts")
        ax.set_title(
            f"{case.name}: detected {metrics['detected_occupied_arcs']} arcs"
        )
        ax.grid(alpha=0.25)
        if row == 0:
            ax.legend(fontsize=8)

        ax = axes[row, 1]
        rng = np.random.default_rng(SEED + row)
        for label, (truth, recovered) in enumerate(data["recovered"]):
            take = rng.choice(len(truth), min(3500, len(truth)), replace=False)
            ax.scatter(truth[take], recovered[take], s=3, alpha=0.22,
                       color=colors[label], label=f"bunch {label}")
        lim = max(
            5.0,
            max(np.quantile(np.abs(item[0]), 0.999) for item in data["recovered"]),
        )
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="exact recovery")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("true local coordinate [m]")
        ax.set_ylabel("recovered continuous coordinate [m]")
        ax.set_title(
            "legacy/per-arc ds ratio = "
            f"{metrics['resolution_gain']:.1f}x"
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Offline validation of density-support longitudinal continuity for h > 1",
        y=0.998,
    )
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(HERE / "density_support_harmonics.png", dpi=200)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


def write_summary(metrics_list):
    lines = [
        "Density-support continuity offline test",
        "=======================================",
        f"C={CIRCUMFERENCE_M:.3f} m, bins={NBINS}, "
        f"smooth={SMOOTH_SIGMA_BINS:.1f} bins, epsilon={DENSITY_THRESHOLD:g}",
        "",
    ]
    for metrics in metrics_list:
        lines.append(
            f"{metrics['case']}: h={metrics['harmonic']}, "
            f"arcs={metrics['detected_occupied_arcs']}/"
            f"{metrics['expected_occupied_arcs']}, "
            f"halo={100*metrics['halo_fraction']:.2f}%, "
            f"resolution gain={metrics['resolution_gain']:.2f}x"
        )
        for bunch in metrics["bunches"]:
            lines.append(
                f"  bunch {bunch['label']} (bucket {bunch['rf_bucket']}): "
                f"arc={bunch['arc_length_m']:.3f} m, "
                f"wrap={bunch['arc_wraps_boundary']}, "
                f"coverage={100*bunch['core_coverage']:.3f}%, "
                f"RMSE={bunch['coordinate_rmse_m']:.3e} m"
            )
        lines.append("")
    lines.append("PASS: all arc-count, continuity, and core-coverage assertions passed.")
    (HERE / "density_support_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    rng = np.random.default_rng(SEED)
    results = []
    metrics_list = []
    for case in CASES:
        print(f"[{case.name}] generating and identifying arcs ...", flush=True)
        metrics, plotting = analyze_case(case, rng)
        results.append((case, metrics, plotting))
        metrics_list.append(metrics)
        print(
            f"  arcs {metrics['detected_occupied_arcs']}/"
            f"{metrics['expected_occupied_arcs']}, resolution gain "
            f"{metrics['resolution_gain']:.2f}x",
            flush=True,
        )
        for bunch in metrics["bunches"]:
            print(
                f"    bucket {bunch['rf_bucket']}: coverage="
                f"{100*bunch['core_coverage']:.3f}%, RMSE="
                f"{bunch['coordinate_rmse_m']:.3e} m, "
                f"wrap={bunch['arc_wraps_boundary']}",
                flush=True,
            )

    with (HERE / "density_support_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2)
    write_summary(metrics_list)
    make_figure(results)
    print("PASS: all offline continuity checks passed.")
    print(f"[saved] {HERE / 'density_support_harmonics.png'}")
    print(f"[saved] {HERE / 'density_support_metrics.json'}")


if __name__ == "__main__":
    main()
