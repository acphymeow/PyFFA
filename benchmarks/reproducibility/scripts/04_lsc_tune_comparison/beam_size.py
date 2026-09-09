"""Measure the turn-1 transverse slice size directly from a 3-D bunch dump."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SliceSigmaMeasurement:
    sigma_x_m: float
    slice_index: int
    slice_count: int
    slice_mean_ek_mev: float
    n_particles: int


def measure_first_tune_record_densest_phase_slice_sigma_x(
    dump_dir: Path,
    seo_dir: Path,
    *,
    n_slices: int = 128,
) -> SliceSigmaMeasurement:
    """Reproduce the original LSC-off first-FFT-record width calculation.

    One first record is read from every selected-particle tune file.  The
    periodic longitudinal phase in column 9 is unwrapped by putting the cut at
    the midpoint of its largest empty arc.  Uniform phase bins are then formed,
    and the RMS x of the most populated bin is evaluated relative to the closed
    orbit at that bin's mean energy.

    The first FFT record is nominal turn 10. This is the exact calculation
    that originally produced
    sigma_x = 15.70582824293744 mm for ``simulation1``.
    """
    files = sorted(dump_dir.glob("particle_*_angle_90.0_rank_*.npz"))
    if not files:
        raise FileNotFoundError(f"no selected-particle tune dumps in {dump_dir}")

    rows = []
    for filename in files:
        particles = np.load(filename)["particles"]
        if particles.ndim != 2 or particles.shape[1] != 17 or len(particles) == 0:
            raise ValueError(f"{filename.name}: expected nonempty (N, 17) data")
        rows.append(particles[0, [0, 2, 5, 9]])  # r, z, Ek, periodic phase
    data = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite first-record data in {dump_dir}")
    radius, _, ek, phase = data.T

    phase = np.mod(phase, 2.0 * np.pi)
    sorted_phase = np.sort(phase)
    gaps = np.diff(np.r_[sorted_phase, sorted_phase[0] + 2.0 * np.pi])
    cut = np.mod(
        sorted_phase[int(np.argmax(gaps))] + 0.5 * float(gaps.max()),
        2.0 * np.pi,
    )
    continuous_phase = np.mod(phase - cut, 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, n_slices + 1)
    slice_index = np.clip(
        np.searchsorted(edges, continuous_phase, side="right") - 1,
        0,
        n_slices - 1,
    )
    counts = np.bincount(slice_index, minlength=n_slices)
    densest = int(np.argmax(counts))
    selected = slice_index == densest
    mean_ek = float(ek[selected].mean())

    seo_ini = np.loadtxt(seo_dir / "SEO_ini.txt", skiprows=2)
    energy_axis = seo_ini[:, 1]
    reference_radius = float(np.interp(mean_ek, energy_axis, seo_ini[:, 11]))
    reference_vr_over_v = float(np.interp(mean_ek, energy_axis, seo_ini[:, 12]))
    reference_pr_geom = reference_vr_over_v / np.sqrt(
        max(1.0 - reference_vr_over_v**2, 1.0e-30)
    )
    x = (radius[selected] - reference_radius) / np.sqrt(
        1.0 + reference_pr_geom**2
    )
    x -= x.mean()
    return SliceSigmaMeasurement(
        sigma_x_m=float(x.std(ddof=0)),
        slice_index=densest,
        slice_count=int(counts[densest]),
        slice_mean_ek_mev=mean_ek,
        n_particles=len(data),
    )


def measure_turn1_densest_slice_sigma_x(
    dump_dir: Path,
    seo_dir: Path,
    *,
    n_slices: int = 128,
) -> SliceSigmaMeasurement:
    """Return physical RMS sigma_x of the most populated turn-1 time slice.

    At a fixed dump azimuth, arrival time is the longitudinal coordinate up to
    a constant scale and sign, so uniform time bins give the same slice
    membership as uniform longitudinal-s bins.  Each slice is referenced to
    the closed orbit at its own mean kinetic energy before x is evaluated.

    The measured RMS is intentionally used instead of reconstructing sigma_x
    only from emittance and Twiss parameters.  Energy-dependent closed-orbit
    displacement (dispersion) mixes the betatron and energy distributions, so
    the physical transverse distribution of a slice need not remain an exact
    single Gaussian.  Its actual second moment is therefore the appropriate
    effective Gaussian width for this reduced toy model.
    """
    files = sorted(dump_dir.glob("crossing_angle_*_rank_*.npz"))
    if not files:
        raise FileNotFoundError(f"no turn-1 bunch dumps in {dump_dir}")

    blocks = []
    for filename in files:
        particles = np.load(filename)["particles"]
        if particles.ndim != 2 or particles.shape[1] != 17:
            raise ValueError(f"{filename.name}: expected (N, 17), got {particles.shape}")
        blocks.append(
            particles[:, [0, 5, 7, 15, 16]]
        )  # r, Ek, survive, global ID, arrival time
    data = np.concatenate(blocks, axis=0).astype(np.float64, copy=False)
    valid = np.isfinite(data).all(axis=1) & (data[:, 2] == 1.0)
    data = data[valid]
    if len(data) == 0:
        raise ValueError(f"no finite surviving particles in {dump_dir}")

    # PositionDumpBunch may contain more than one crossing per particle in one
    # folder.  Select the earliest record of every Global ID so "turn 1" is
    # unambiguous and a particle is never counted twice.
    order = np.argsort(data[:, 4], kind="stable")
    data = data[order]
    _, first = np.unique(data[:, 3].astype(np.int64), return_index=True)
    data = data[np.sort(first)]

    radius, ek, _, _, arrival_time = data.T
    time_min = float(arrival_time.min())
    time_max = float(arrival_time.max())
    if not time_max > time_min:
        raise ValueError("turn-1 arrival-time extent is zero")
    edges = np.linspace(time_min, time_max, n_slices + 1)
    slice_index = np.clip(
        np.searchsorted(edges, arrival_time, side="right") - 1,
        0,
        n_slices - 1,
    )
    counts = np.bincount(slice_index, minlength=n_slices)
    densest = int(np.argmax(counts))
    selected = slice_index == densest
    mean_ek = float(ek[selected].mean())

    seo_ini = np.loadtxt(seo_dir / "SEO_ini.txt", skiprows=2)
    energy_axis = seo_ini[:, 1]
    reference_radius = float(np.interp(mean_ek, energy_axis, seo_ini[:, 11]))
    reference_vr_over_v = float(np.interp(mean_ek, energy_axis, seo_ini[:, 12]))
    reference_pr_geom = reference_vr_over_v / np.sqrt(
        max(1.0 - reference_vr_over_v**2, 1.0e-30)
    )
    x = (radius[selected] - reference_radius) / np.sqrt(
        1.0 + reference_pr_geom**2
    )
    return SliceSigmaMeasurement(
        sigma_x_m=float(x.std(ddof=0)),
        slice_index=densest,
        slice_count=int(counts[densest]),
        slice_mean_ek_mev=mean_ek,
        n_particles=len(data),
    )
