#!/usr/bin/env python
"""Plot the configured mid-plane Bz map together with extraction-kicker regions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as path_effects
from matplotlib.patches import Wedge
import numpy as np

from FFAG_Field import FFAG_BField_Error, FFAG_BumpKickerField
from main_track import load_bmap_auto


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read a PyFFA tracking config and plot the base Bz map, configured "
            "kicker locations, and the kicker-only delta-Bz map."
        )
    )
    parser.add_argument(
        "-j", "--config", default="config_track.json",
        help="Tracking JSON file (default: config_track.json).",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output PNG path (default: next to the config file).",
    )
    parser.add_argument("--nr", type=int, default=220, help="Number of radial samples.")
    parser.add_argument(
        "--nphi", type=int, default=600,
        help="Number of azimuth samples across the plotted kicker region.",
    )
    parser.add_argument("--z", type=float, default=0.0, help="Vertical plot plane in m.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    parser.add_argument(
        "--zoom-margin-deg", type=float, default=2.0,
        help="Azimuth margin around the kicker group in the zoom panels.",
    )
    parser.add_argument(
        "--zoom-margin-m", type=float, default=0.10,
        help="Radial margin around the kicker group in the zoom panels.",
    )
    parser.add_argument("--show", action="store_true", help="Also open the Matplotlib window.")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def resolve_config_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    relative_to_config = (config_path.parent / path).resolve()
    if relative_to_config.exists():
        return relative_to_config
    return path.resolve()


def load_optional_harmonics(config: dict, bmap_path: Path):
    seo_folder = config.get("BmapAndSEO", {}).get("SEO_FileName")
    if not seo_folder:
        return None, None

    seo_config_path = bmap_path / seo_folder / "config_SEO.json"
    if not seo_config_path.exists():
        return None, seo_config_path

    seo_config = load_json(seo_config_path)
    harmonics = seo_config.get("BHarmonics", {"enable": False, "harmonics": []})
    source = FFAG_BField_Error(
        harmonics.get("harmonics", []), harmonics.get("enable", False)
    )
    return source, seo_config_path


def evaluate_base_field(bmap, max_order, r_grid, phi_grid, z_value):
    shape = r_grid.shape
    r_flat = np.ascontiguousarray(r_grid.ravel(), dtype=np.float64)
    phi_flat = np.ascontiguousarray(phi_grid.ravel(), dtype=np.float64)
    z_flat = np.full(r_flat.shape, z_value, dtype=np.float64)

    bz_flat = np.zeros_like(r_flat)
    br_flat = np.zeros_like(r_flat)
    bfi_flat = np.zeros_like(r_flat)

    # Do not query interpolation tables outside their radial domain. This also
    # lets a kicker region extend slightly beyond the available base Bmap.
    valid = (r_flat >= bmap.r_min) & (r_flat <= bmap.r_max)
    if np.any(valid):
        r_query = np.ascontiguousarray(r_flat[valid])
        phi_global = np.ascontiguousarray(phi_flat[valid])
        z_query = np.ascontiguousarray(z_flat[valid])
        inj = np.ones(r_query.size, dtype=np.float64)
        survive = np.ones(r_query.size, dtype=np.float64)
        bz_query = np.zeros_like(r_query)
        br_query = np.zeros_like(r_query)
        bfi_query = np.zeros_like(r_query)

        field_type = getattr(bmap, "field_type", "analytical")
        if field_type == "opera3d":
            phi_query = phi_global
        elif field_type == "analytical":
            phi_query = np.mod(phi_global, 2.0 * np.pi / bmap.Nsectors)
        else:
            raise ValueError(f"Unsupported Bmap field_type: {field_type!r}")

        bmap.GetBfields(
            r_query, phi_query, z_query, inj, survive,
            bz_query, br_query, bfi_query, SearchMod=False,
        )
        if field_type == "analytical" and max_order >= 1:
            bmap.AddHigherOrderBmaps(
                r_query, phi_query, z_query,
                bz_query, br_query, bfi_query, max_order,
            )

        bz_flat[valid] = bz_query
        br_flat[valid] = br_query
        bfi_flat[valid] = bfi_query

    return (
        bz_flat.reshape(shape),
        br_flat.reshape(shape),
        bfi_flat.reshape(shape),
    )


def evaluate_kicker_field(kicker_config, r_grid, phi_grid, z_value):
    shape = r_grid.shape
    r_flat = np.ascontiguousarray(r_grid.ravel(), dtype=np.float64)
    phi_flat = np.ascontiguousarray(phi_grid.ravel(), dtype=np.float64)
    z_flat = np.full(r_flat.shape, z_value, dtype=np.float64)
    inj = np.ones(r_flat.size, dtype=np.float64)
    survive = np.ones(r_flat.size, dtype=np.float64)
    bz_flat = np.zeros_like(r_flat)
    br_flat = np.zeros_like(r_flat)
    bfi_flat = np.zeros_like(r_flat)

    # Always evaluate the configured regions for this diagnostic, even when
    # BumpKicker.enable is false in the tracking config.
    source = FFAG_BumpKickerField(kicker_config, EnableFlag=True)
    source.AddBumpKicker_to_Bmap_njit(
        r_flat, phi_flat, z_flat, inj, survive,
        bz_flat, br_flat, bfi_flat,
    )
    return (
        bz_flat.reshape(shape),
        br_flat.reshape(shape),
        bfi_flat.reshape(shape),
    )


def kicker_bz(kicker):
    return float(kicker["sign"]) * float(kicker["BL_Tm"]) / float(kicker["L_eff_m"])


def add_kicker_regions_cartesian(ax, kickers, labels=True):
    for kicker in kickers:
        start_deg = float(kicker["phi_start_deg"])
        end_deg = start_deg + float(kicker["phi_span_deg"])
        rmin = float(kicker["rmin_m"])
        rmax = float(kicker["rmax_m"])
        bz = kicker_bz(kicker)
        patch = Wedge(
            (0.0, 0.0), rmax, start_deg, end_deg, width=rmax - rmin,
            facecolor="none", edgecolor="black", linewidth=1.6,
        )
        patch.set_path_effects([
            path_effects.Stroke(linewidth=3.4, foreground="white"),
            path_effects.Normal(),
        ])
        ax.add_patch(patch)

        if labels:
            phi_mid = np.deg2rad(0.5 * (start_deg + end_deg))
            r_mid = 0.5 * (rmin + rmax)
            polarity = "+" if bz >= 0.0 else "−"
            ax.text(
                r_mid * np.cos(phi_mid), r_mid * np.sin(phi_mid),
                f"{str(kicker.get('name', 'K'))} {polarity}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.1},
            )


def symmetric_norm(values):
    vmax = float(np.nanmax(np.abs(values)))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def kicker_cartesian_limits(kickers, margin_m):
    x_values = []
    y_values = []
    for kicker in kickers:
        start = np.deg2rad(float(kicker["phi_start_deg"]))
        end = start + np.deg2rad(float(kicker["phi_span_deg"]))
        phi = np.linspace(start, end, 80)
        for radius in (float(kicker["rmin_m"]), float(kicker["rmax_m"])):
            x_values.extend((radius * np.cos(phi)).tolist())
            y_values.extend((radius * np.sin(phi)).tolist())
    return (
        (min(x_values) - margin_m, max(x_values) + margin_m),
        (min(y_values) - margin_m, max(y_values) + margin_m),
    )


def configure_cartesian_axis(ax, title, x_limits, y_limits):
    ax.set_title(title)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)


def print_kicker_summary(kickers, enabled):
    print(f"BumpKicker.enable = {enabled} (regions are plotted regardless of this switch)")
    print("name   phi_start -> phi_end [deg]    r range [m]       Bz [T]")
    for kicker in kickers:
        start = float(kicker["phi_start_deg"])
        end = start + float(kicker["phi_span_deg"])
        print(
            f"{str(kicker.get('name', 'K')):<6} "
            f"{start:9.3f} -> {end:9.3f}      "
            f"{float(kicker['rmin_m']):.3f}..{float(kicker['rmax_m']):.3f}    "
            f"{kicker_bz(kicker):+.7f}"
        )


def main():
    args = parse_args()
    if args.nr < 2 or args.nphi < 3:
        raise ValueError("--nr must be >= 2 and --nphi must be >= 3")

    config_path = Path(args.config).expanduser().resolve()
    config = load_json(config_path)
    bmap_config = config["BmapAndSEO"]
    bmap_path = resolve_config_path(bmap_config["maps"], config_path)
    if bmap_path.is_file() and bmap_path.name == "config_Bmap.json":
        bmap_path = bmap_path.parent

    kicker_group = config.get("BumpKicker", {"enable": False, "kickers": []})
    kickers = kicker_group.get("kickers", [])
    if not kickers:
        raise ValueError(f"{config_path} does not contain any BumpKicker.kickers entries")

    max_order = int(bmap_config.get("max_order", 0))
    bmap = load_bmap_auto(str(bmap_path), max_order)

    kicker_rmin = min(float(k["rmin_m"]) for k in kickers)
    kicker_rmax = max(float(k["rmax_m"]) for k in kickers)
    start_deg = min(float(k["phi_start_deg"]) for k in kickers)
    end_deg = max(
        float(k["phi_start_deg"]) + float(k["phi_span_deg"]) for k in kickers
    )
    theta_zoom = (
        start_deg - args.zoom_margin_deg,
        end_deg + args.zoom_margin_deg,
    )
    r_zoom = (
        max(0.0, kicker_rmin - args.zoom_margin_m),
        kicker_rmax + args.zoom_margin_m,
    )

    # Only sample the local Cartesian view requested for kicker placement.
    r_axis = np.linspace(r_zoom[0], r_zoom[1], args.nr)
    phi_axis = np.deg2rad(np.linspace(theta_zoom[0], theta_zoom[1], args.nphi))
    r_grid, phi_grid = np.meshgrid(r_axis, phi_axis, indexing="ij")

    bz_base, _, _ = evaluate_base_field(bmap, max_order, r_grid, phi_grid, args.z)
    harmonics, harmonics_path = load_optional_harmonics(config, bmap_path)
    if harmonics is not None:
        flat_r = np.ascontiguousarray(r_grid.ravel())
        flat_phi = np.ascontiguousarray(phi_grid.ravel())
        flat_z = np.full(flat_r.shape, args.z)
        flat_bz = bz_base.ravel()
        flat_br = np.zeros_like(flat_bz)
        flat_bfi = np.zeros_like(flat_bz)
        harmonics.AddBHarmonics_to_Bmap_njit(
            flat_r, flat_phi, flat_z, flat_bz, flat_br, flat_bfi
        )

    bz_kicker, _, _ = evaluate_kicker_field(kickers, r_grid, phi_grid, args.z)
    x_zoom_grid = r_grid * np.cos(phi_grid)
    y_zoom_grid = r_grid * np.sin(phi_grid)
    x_limits, y_limits = kicker_cartesian_limits(kickers, args.zoom_margin_m)

    fig, (ax_zoom, ax_kicker) = plt.subplots(
        1, 2, figsize=(12.8, 7.2), constrained_layout=True,
    )

    base_norm = symmetric_norm(bz_base)
    base_zoom_map = ax_zoom.contourf(
        x_zoom_grid, y_zoom_grid, bz_base,
        levels=np.linspace(base_norm.vmin, base_norm.vmax, 101),
        cmap="RdBu_r", norm=base_norm, extend="both",
    )
    kicker_norm = symmetric_norm(bz_kicker)
    kicker_map = ax_kicker.contourf(
        x_zoom_grid, y_zoom_grid, bz_kicker,
        levels=np.linspace(kicker_norm.vmin, kicker_norm.vmax, 81),
        cmap="RdBu_r", norm=kicker_norm, extend="both",
    )

    add_kicker_regions_cartesian(ax_zoom, kickers, labels=True)
    add_kicker_regions_cartesian(ax_kicker, kickers, labels=True)

    config_state = "enabled" if kicker_group.get("enable", False) else "disabled in tracking"
    configure_cartesian_axis(
        ax_zoom, f"Base Bz and kicker boundaries ({config_state})", x_limits, y_limits,
    )
    configure_cartesian_axis(
        ax_kicker, "Kicker-only delta Bz", x_limits, y_limits,
    )

    fig.colorbar(base_zoom_map, ax=ax_zoom, pad=0.02, shrink=0.92, label="Base Bz [T]")
    fig.colorbar(kicker_map, ax=ax_kicker, pad=0.02, shrink=0.92, label="Kicker delta Bz [T]")
    fig.suptitle(
        f"{getattr(bmap, 'field_type', 'analytical')} Bmap + horizontal extraction kickers, "
        f"z = {args.z:.4g} m",
        fontsize=14,
    )

    if args.output is None:
        output_path = config_path.parent / "bmap_with_kickers.png"
    else:
        output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)

    print(f"Config          : {config_path}")
    print(f"Bmap            : {bmap_path}")
    if harmonics_path is not None:
        print(f"SEO harmonics   : {harmonics_path}")
    print_kicker_summary(kickers, bool(kicker_group.get("enable", False)))
    print(f"Saved figure    : {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
