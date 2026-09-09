# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np

from FFAG_Field import FFAG_Bfield_3D_opera


def parse_args():
    parser = argparse.ArgumentParser(description="Visual test for PyFFA OPERA 3D field-map npz data.")
    parser.add_argument("--field-dir", default="OperaFieldMap", help="Directory containing FD16.json and FD16.npz.")
    parser.add_argument("--dataset", default="FD16", help="Dataset basename, for example FD16.")
    parser.add_argument("--out-dir", default="Opera3D_field_plots", help="Directory for output figures.")
    parser.add_argument("--y-plot", type=float, default=0.010, help="Off-mid-plane vertical coordinate in m.")
    parser.add_argument("--r-compare", type=float, nargs="+", default=[11.2, 11.5, 11.8, 12.0])
    parser.add_argument("--show", action="store_true", help="Show figures interactively.")
    return parser.parse_args()


def local_xz_mesh(bmap):
    r_grid, phi_grid = np.meshgrid(bmap.r_axis, bmap.phi_axis_rad, indexing="ij")
    dphi = phi_grid - bmap.model_origin_phi_rad
    x_local = r_grid * np.cos(dphi) - bmap.model_origin_radius_m
    z_local = r_grid * np.sin(dphi)
    return x_local, z_local


def save_or_show(fig, path, show):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        import matplotlib.pyplot as plt
        plt.show()


def plot_local_field_map(ax, fig, x_local, z_local, values, label):
    mesh = ax.tripcolor(
        x_local.ravel(),
        z_local.ravel(),
        values.ravel(),
        shading="gouraud",
    )
    fig.colorbar(mesh, ax=ax, label=label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("OPERA local X (m)")
    ax.set_ylabel("OPERA local Z (m)")


def main():
    args = parse_args()

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bmap = FFAG_Bfield_3D_opera(field_dir=args.field_dir, dataset=args.dataset)
    os.makedirs(args.out_dir, exist_ok=True)

    print("OPERA 3D field-map summary:")
    for key, value in bmap.describe().items():
        print(f"  {key}: {value}")

    x_local, z_local = local_xz_mesh(bmap)
    phi_axis_deg = np.rad2deg(bmap.phi_axis_rad)
    phi_cell = bmap.phi_axis_rad - bmap.fi_min

    iy_mid = 0
    iy_plot = int(np.argmin(np.abs(bmap.y_axis - args.y_plot)))
    y_plot = bmap.y_axis[iy_plot]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_local_field_map(ax, fig, x_local, z_local, bmap.bz_map[iy_mid], "Bz (T)")
    ax.set_title("Mid-plane Bz")
    save_or_show(fig, os.path.join(args.out_dir, "midplane_Bz.png"), args.show)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_local_field_map(ax, fig, x_local, z_local, bmap.br_map[iy_plot], "Br (T)")
    ax.set_title(f"Off-mid-plane Br, y = {y_plot * 1000:.1f} mm")
    save_or_show(fig, os.path.join(args.out_dir, "offplane_Br.png"), args.show)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_local_field_map(ax, fig, x_local, z_local, bmap.bfi_map[iy_plot], "Bphi (T)")
    ax.set_title(f"Off-mid-plane Bphi, y = {y_plot * 1000:.1f} mm")
    save_or_show(fig, os.path.join(args.out_dir, "offplane_Bphi.png"), args.show)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r0 in args.r_compare:
        ir = int(np.argmin(np.abs(bmap.r_axis - r0)))
        ax.plot(phi_axis_deg, bmap.bz_map[iy_mid, ir], label=f"R = {bmap.r_axis[ir]:.3f} m")
    ax.set_xlabel("Table azimuth phi (deg)")
    ax.set_ylabel("Bz (T)")
    ax.set_title("Mid-plane Bz(phi)")
    ax.grid(True)
    ax.legend()
    save_or_show(fig, os.path.join(args.out_dir, "midplane_Bz_phi_curves.png"), args.show)

    r_test = 12.0
    ir_test = int(np.argmin(np.abs(bmap.r_axis - r_test)))
    n = bmap.fi_size
    r_query = np.full(n, bmap.r_axis[ir_test], dtype=np.float64)
    z_query = np.full(n, y_plot, dtype=np.float64)
    inj = np.ones(n, dtype=np.bool_)
    survive = np.ones(n, dtype=np.bool_)
    bz_interp = np.zeros(n, dtype=np.float64)
    br_interp = np.zeros(n, dtype=np.float64)
    bfi_interp = np.zeros(n, dtype=np.float64)

    bmap.GetBfields(r_query, phi_cell, z_query, inj, survive, bz_interp, br_interp, bfi_interp)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(phi_axis_deg, bmap.bz_map[iy_plot, ir_test], label="npz grid")
    axes[0].plot(phi_axis_deg, bz_interp, "--", label="GetBfields")
    axes[0].set_ylabel("Bz (T)")
    axes[0].legend()

    axes[1].plot(phi_axis_deg, bmap.br_map[iy_plot, ir_test], label="npz grid")
    axes[1].plot(phi_axis_deg, br_interp, "--", label="GetBfields")
    axes[1].set_ylabel("Br (T)")

    axes[2].plot(phi_axis_deg, bmap.bfi_map[iy_plot, ir_test], label="npz grid")
    axes[2].plot(phi_axis_deg, bfi_interp, "--", label="GetBfields")
    axes[2].set_ylabel("Bphi (T)")
    axes[2].set_xlabel("Table azimuth phi (deg)")

    for ax in axes:
        ax.grid(True)
    fig.suptitle(f"Interpolation check, R = {bmap.r_axis[ir_test]:.3f} m, y = {y_plot * 1000:.1f} mm")
    save_or_show(fig, os.path.join(args.out_dir, "getbfields_interpolation_check.png"), args.show)

    bz_neg = np.zeros(n, dtype=np.float64)
    br_neg = np.zeros(n, dtype=np.float64)
    bfi_neg = np.zeros(n, dtype=np.float64)
    bmap.GetBfields(r_query, phi_cell, -z_query, inj, survive, bz_neg, br_neg, bfi_neg)

    print("Symmetry check at R = %.3f m, |z| = %.4f m:" % (bmap.r_axis[ir_test], y_plot))
    print("  max |Bz(+z)-Bz(-z)|      =", np.max(np.abs(bz_interp - bz_neg)))
    print("  max |Br(+z)+Br(-z)|      =", np.max(np.abs(br_interp + br_neg)))
    print("  max |Bphi(+z)+Bphi(-z)|  =", np.max(np.abs(bfi_interp + bfi_neg)))
    print(f"Figures saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
