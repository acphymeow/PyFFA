"""Application tune footprint based directly on ``test_fft_tune6.py``.

The tune extraction below deliberately preserves the original production
script's numerical convention: time sorting, all available records, the
``x + i p`` complex signal, the subsequent ``1 - tune`` conversion, and the
same horizontal integer-branch rule. Only path handling, CSV/PDF output, and
the reviewer-requested resonance overlay are added.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.signal import savgol_filter


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bundle_paths import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from FFAG_MathTools import Bilinear_interp_2D_vect_uniform
from paper_resonance_overlay import draw_paper_resonances


DEFAULT_DUMP = (
    REPO_ROOT.parent / "output" / "FD16_paint_1_1" / "FFT_Tune_Dump_0"
)
DEFAULT_SEO = REPO_ROOT / "Bmap_FD16" / "resultsSEO_noError"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs"
DEFAULT_RESULTS = SCRIPT_DIR / "results"

COL_X, COL_XP = 0, 1
COL_Z, COL_ZP = 2, 3
COL_FI, COL_EK = 4, 5

PAD_FACTOR = 8
USE_SAVGOL = False
SG_WINDOW, SG_ORDER = 18, 4
QX_LIMITS = (2.30, 3.30)
QZ_LIMITS = (1.90, 2.90)
RESONANCE_LINE_COLOR = "#9f9f9f"
RESONANCE_TEXT_COLOR = "#858585"
RESONANCE_LINEWIDTH = 0.58
RESONANCE_LINE_ALPHA = 0.95
RESONANCE_LABEL_FONTSIZE = 7.8
RESONANCE_TEXT_ALPHA = 0.82
RESONANCE_LABEL_BOX_ALPHA = 0.36
ADDITIONAL_THIRD_ORDER_LABELS = {
    (2, 1, 9),    # 2Qx + Qz = 9
    (1, -2, -2),  # Qx - 2Qz = -2
    (1, 2, 8),    # Qx + 2Qz = 8
}


def particle_id(path: Path) -> int:
    match = re.search(r"particle_(\d+)_", path.name)
    return int(match.group(1)) if match else -1


def parabolic_interpolate(amplitude: np.ndarray, index: int) -> float:
    y0, y1, y2 = amplitude[index - 1:index + 2]
    denominator = y0 - 2.0 * y1 + y2
    if denominator == 0.0:
        return float(index)
    return float(index + 0.5 * (y0 - y2) / denominator)


def tune_from_xpx(
    x: np.ndarray,
    px: np.ndarray,
    *,
    pad_factor: int = PAD_FACTOR,
    ignore_dc: bool = True,
    px_scale: float | None = None,
) -> float:
    """Original ``test_fft_tune6.py`` complex FFT tune estimator."""
    x = np.asarray(x)
    px = np.asarray(px)
    if len(x) != len(px):
        raise ValueError("x and px must have the same length")
    if px_scale is None:
        sigma_x = x.std(ddof=0)
        sigma_px = px.std(ddof=0)
        px_scale = sigma_px / sigma_x if sigma_x > 0.0 and sigma_px > 0.0 else 1.0

    signal = x + 1j * (px / px_scale)
    signal = signal - signal.mean()
    nfft = int(pad_factor * len(signal))
    spectrum = np.fft.fft(signal, n=nfft)
    frequencies = np.fft.fftfreq(nfft, d=1.0)
    amplitude = np.abs(spectrum)
    if ignore_dc:
        amplitude[0] = 0.0
    index = int(np.argmax(amplitude))
    index = min(max(index, 1), nfft - 2)
    peak = parabolic_interpolate(amplitude, index)
    return float((peak % nfft) * (frequencies[1] - frequencies[0]))


def load_seo(seo_dir: Path) -> tuple[np.ndarray, ...]:
    seo_r = np.loadtxt(seo_dir / "SEO_r.txt", skiprows=1)
    seo_pr = np.loadtxt(seo_dir / "SEO_pr.txt", skiprows=1)
    return (
        seo_r[0, 1:],
        seo_r[1:, 0],
        seo_r[1:, 1:],
        seo_pr[1:, 1:],
    )


def subtract_closed_orbit(
    x: np.ndarray,
    px: np.ndarray,
    fi: np.ndarray,
    energy: np.ndarray,
    seo: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    fi_axis, energy_axis, r_matrix, pr_matrix = seo
    fi_mod = np.mod(fi, 2.0 * np.pi)
    r0 = np.zeros_like(x)
    pr0 = np.zeros_like(px)
    flag_r = np.zeros_like(x)
    flag_pr = np.zeros_like(px)

    Bilinear_interp_2D_vect_uniform(
        energy_axis[0], energy_axis[1] - energy_axis[0], len(energy_axis),
        fi_axis[0], fi_axis[1] - fi_axis[0], len(fi_axis),
        r_matrix, energy, fi_mod, r0, flag_r,
    )
    Bilinear_interp_2D_vect_uniform(
        energy_axis[0], energy_axis[1] - energy_axis[0], len(energy_axis),
        fi_axis[0], fi_axis[1] - fi_axis[0], len(fi_axis),
        pr_matrix, energy, fi_mod, pr0, flag_pr,
    )
    return x - r0, px - pr0


def select_files(paths: list[Path], maximum: int | None) -> list[Path]:
    if maximum is None or maximum >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, maximum, dtype=int)
    return [paths[index] for index in indices]


def extract_application_tunes(
    dump_dir: Path,
    seo_dir: Path,
    *,
    maximum_files: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Numeric sorting changes only CSV row order, not per-particle results.
    paths = sorted(dump_dir.glob("particle_*.npz"), key=particle_id)
    if not paths:
        raise FileNotFoundError(f"No application particle dumps found in {dump_dir}")
    paths = select_files(paths, maximum_files)
    seo = load_seo(seo_dir)
    identifiers: list[int] = []
    horizontal_fractional: list[float] = []
    vertical_fractional: list[float] = []

    for index, path in enumerate(paths, start=1):
        particle = np.load(path)["particles"]
        # Preserve the explicit chronological sorting in test_fft_tune6.py.
        particle = particle[np.argsort(particle[:, -1])]
        if len(particle) < SG_WINDOW:
            continue

        x = particle[:, COL_X].copy()
        px = particle[:, COL_XP].copy()
        z = particle[:, COL_Z].copy()
        pz = particle[:, COL_ZP].copy()
        fi = particle[:, COL_FI]
        energy = particle[:, COL_EK]

        if USE_SAVGOL:
            window = SG_WINDOW if SG_WINDOW % 2 else SG_WINDOW + 1
            x -= savgol_filter(x, window, SG_ORDER)
            px -= savgol_filter(px, window, SG_ORDER)
            z -= savgol_filter(z, window, SG_ORDER)
            pz -= savgol_filter(pz, window, SG_ORDER)

        x, px = subtract_closed_orbit(x, px, fi, energy, seo)
        horizontal_fractional.append(tune_from_xpx(x, px))
        vertical_fractional.append(tune_from_xpx(z, pz))
        identifiers.append(particle_id(path))
        if index % 1000 == 0 or index == len(paths):
            print(f"Processed {index}/{len(paths)} application particles", flush=True)

    vx = np.asarray(horizontal_fractional)
    vz = np.asarray(vertical_fractional)

    # Preserve the exact absolute-tune conversion in test_fft_tune6.py.
    qx_fraction = 1.0 - vx
    qz_fraction = 1.0 - vz
    qx_fraction[qx_fraction > 0.45] -= 1.0
    qx = qx_fraction + 3.0
    qz = qz_fraction + 2.0
    return np.asarray(identifiers, dtype=int), qx, qz


def plot_application_tunes(
    qx: np.ndarray,
    qz: np.ndarray,
    output_dir: Path,
    *,
    output_stem: str,
    label_mode: str,
) -> Path:
    # Match the original application figure geometry and density convention.
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.hist2d(
        qx, qz, bins=(256, 256), range=(QX_LIMITS, QZ_LIMITS),
        norm=LogNorm(), cmap="viridis", zorder=-1,
    )
    axis.set_xlabel(r"$Q_x$", fontsize=22)
    axis.set_ylabel(r"$Q_z$", fontsize=22)
    axis.tick_params(axis="both", labelsize=20)
    axis.set_xlim(QX_LIMITS)
    axis.set_ylim(QZ_LIMITS)
    draw_paper_resonances(
        axis,
        qx,
        qz,
        QX_LIMITS,
        QZ_LIMITS,
        label_mode=label_mode,
        line_color=RESONANCE_LINE_COLOR,
        text_color=RESONANCE_TEXT_COLOR,
        linewidth=RESONANCE_LINEWIDTH,
        line_alpha=RESONANCE_LINE_ALPHA,
        label_fontsize=RESONANCE_LABEL_FONTSIZE,
        label_zorder=-1,
        bold_math_labels=False,
        text_alpha=RESONANCE_TEXT_ALPHA,
        label_fontweight="normal",
        label_box_alpha=RESONANCE_LABEL_BOX_ALPHA,
        extra_label_keys=ADDITIONAL_THIRD_ORDER_LABELS,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / output_stem
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--seo-dir", type=Path, default=DEFAULT_SEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-stem", default="application_tune_shift_with_resonances")
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--label-mode",
        choices=("all", "low-order", "highlighted", "none"),
        default="low-order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identifiers, qx, qz = extract_application_tunes(
        args.dump_dir, args.seo_dir, maximum_files=args.max_files,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_name = (
        "application_tunes.csv"
        if args.max_files is None
        else f"application_tunes_sample{args.max_files}.csv"
    )
    result_path = args.results_dir / result_name
    np.savetxt(
        result_path,
        np.column_stack((identifiers, qx, qz)),
        delimiter=",",
        header="particle_id,Qx,Qz",
        comments="",
    )
    output = plot_application_tunes(
        qx, qz, args.output_dir,
        output_stem=args.output_stem,
        label_mode=args.label_mode,
    )
    print(f"Saved {result_path}")
    print(f"Saved {output.with_suffix('.png')}")
    print(
        f"Tune ranges: Qx={qx.min():.6f}--{qx.max():.6f}, "
        f"Qz={qz.min():.6f}--{qz.max():.6f}"
    )
    plt.show()


if __name__ == "__main__":
    main()
