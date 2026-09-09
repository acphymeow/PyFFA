# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np

from FFAG_Field import FFAG_Bfield_3D_opera


def print_bz_cell_boundary_report(bmap, radius, z_value):
    cell_period = 2.0 * np.pi / bmap.Nsectors
    phi_query = np.array(
        [
            bmap.fi_min,
            bmap.fi_min + cell_period - 1e-12,
            bmap.fi_min + cell_period,
        ],
        dtype=np.float64,
    )
    r_query = np.full(phi_query.size, radius, dtype=np.float64)
    z_query = np.full(phi_query.size, z_value, dtype=np.float64)
    inj = np.ones(phi_query.size, dtype=np.bool_)
    survive = np.ones(phi_query.size, dtype=np.bool_)
    bz_query = np.zeros(phi_query.size, dtype=np.float64)
    br_query = np.zeros(phi_query.size, dtype=np.float64)
    bphi_query = np.zeros(phi_query.size, dtype=np.float64)

    bmap.GetBfields(
        r_query, phi_query, z_query, inj, survive,
        bz_query, br_query, bphi_query,
        SearchMod=False,
    )

    ir = int(np.argmin(np.abs(bmap.r_axis - radius)))
    iy = int(np.argmin(np.abs(bmap.y_axis - abs(z_value))))
    raw_head = bmap.bz_map[iy, ir, 0]
    raw_tail = bmap.bz_map[iy, ir, -1]
    raw_diff = raw_tail - raw_head

    raw_diff_all_r = bmap.bz_map[iy, :, -1] - bmap.bz_map[iy, :, 0]
    imax = int(np.argmax(np.abs(raw_diff_all_r)))

    print("\nBz cell-boundary continuity check")
    print(f"  query R = {radius:.6f} m, z = {z_value:.6f} m")
    print(
        f"  GetBfields head          : {bz_query[0]:+.12e} T "
        f"at {np.rad2deg(phi_query[0]):.12f} deg"
    )
    print(
        f"  GetBfields tail - eps    : {bz_query[1]:+.12e} T "
        f"at {np.rad2deg(phi_query[1]):.12f} deg"
    )
    print(
        f"  GetBfields folded end    : {bz_query[2]:+.12e} T "
        f"at {np.rad2deg(phi_query[2]):.12f} deg"
    )
    print(f"  tail-eps minus head      : {bz_query[1] - bz_query[0]:+.12e} T")
    print(f"  folded-end minus tail-eps: {bz_query[2] - bz_query[1]:+.12e} T")
    print(
        f"  raw nearest R,y          : R={bmap.r_axis[ir]:.6f} m, "
        f"y={bmap.y_axis[iy]:.6f} m"
    )
    print(f"  raw table head           : {raw_head:+.12e} T")
    print(f"  raw table tail           : {raw_tail:+.12e} T")
    print(f"  raw tail minus head      : {raw_diff:+.12e} T")
    print(
        f"  max raw |tail-head| at y : {np.abs(raw_diff_all_r[imax]):.12e} T "
        f"at R={bmap.r_axis[imax]:.6f} m"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot OPERA 3D B-field components along azimuth using vectorized GetBfields."
    )
    parser.add_argument("--field-dir", default=os.path.join("Bmaps", "OperaFieldMap"))
    parser.add_argument("--dataset", default="FD16")
    parser.add_argument("--radius", type=float, default=11.5, help="Query radius in m.")
    parser.add_argument("--z", type=float, default=0.010, help="Vertical coordinate in m.")
    parser.add_argument("--cells", type=float, default=4.0, help="Number of FFAG cells to plot.")
    parser.add_argument("--points", type=int, default=1441, help="Number of azimuth samples.")
    parser.add_argument("--out-dir", default="Bmap_class_plots")
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window.")
    parser.add_argument("--backend", default=None, help="Optional Matplotlib backend, for example TkAgg or QtAgg.")
    parser.add_argument("--no-boundary-report", action="store_true", help="Do not print the Bz cell boundary report.")
    return parser.parse_args()


def main():
    args = parse_args()

    import matplotlib
    if args.backend:
        matplotlib.use(args.backend)
    elif args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bmap = FFAG_Bfield_3D_opera(field_dir=args.field_dir, dataset=args.dataset)

    cell_period = 2.0 * np.pi / bmap.Nsectors
    phi_span = args.cells * cell_period
    phi_global = np.linspace(bmap.fi_min, bmap.fi_min + phi_span, args.points, dtype=np.float64)
    phi_cell_deg = np.rad2deg(phi_global - bmap.fi_min)

    r = np.full(args.points, args.radius, dtype=np.float64)
    z = np.full(args.points, args.z, dtype=np.float64)
    inj = np.ones(args.points, dtype=np.bool_)
    survive = np.ones(args.points, dtype=np.bool_)

    bz = np.zeros(args.points, dtype=np.float64)
    br = np.zeros(args.points, dtype=np.float64)
    bphi = np.zeros(args.points, dtype=np.float64)

    # Vectorized interpolation: all azimuth samples are queried in one GetBfields call.
    bmap.GetBfields(r, phi_global, z, inj, survive, bz, br, bphi, SearchMod=False)

    if not args.no_boundary_report:
        print_bz_cell_boundary_report(bmap, args.radius, args.z)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"Opera3D_GetBfields_R{args.radius:.3f}_z{args.z:+.3f}_{args.cells:g}cells_phi.png",
    )

    components = [
        ("Bz", bz, "Bz (T)"),
        ("Br", br, "Br (T)"),
        ("Bphi", bphi, "Bphi (T)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, (name, values, ylabel) in zip(axes, components):
        ax.plot(phi_cell_deg, values, linewidth=1.6)
        # for boundary_index in range(1, int(np.floor(args.cells)) + 1):
        #     ax.axvline(np.rad2deg(boundary_index * cell_period), color="0.65", linewidth=0.8, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        ax.grid(True, alpha=0.35)

    axes[-1].set_xlabel("cell azimuth phi - phi_min (deg)")
    fig.suptitle(
        f"OPERA 3D GetBfields at R = {args.radius:.3f} m, "
        f"z = {args.z * 1000:.1f} mm, {args.cells:g} cells"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()

    print(f"Saved plot: {os.path.abspath(out_path)}")
    print(f"phi range: {phi_cell_deg[0]:.3f} deg to {phi_cell_deg[-1]:.3f} deg")
    print(f"Bz range: {bz.min(): .6e} to {bz.max(): .6e} T")
    print(f"Br range: {br.min(): .6e} to {br.max(): .6e} T")
    print(f"Bphi range: {bphi.min(): .6e} to {bphi.max(): .6e} T")

if __name__ == "__main__":
    main()
