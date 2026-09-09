# -*- coding: utf-8 -*-

import numpy as np


# =========================================================
# Grid settings
# =========================================================

N_CELLS = 16
CELL_CENTER_DEG = 90.0
CELL_WIDTH_DEG = 360.0 / N_CELLS

PHI_MIN_DEG = CELL_CENTER_DEG - CELL_WIDTH_DEG / 2.0
PHI_MAX_DEG = CELL_CENTER_DEG + CELL_WIDTH_DEG / 2.0

R_MIN_M = 10.8
R_MAX_M = 12.2
DR_M = 0.005

DPHI_DEG = 0.1

Y_MIN_M = 0.000
Y_MAX_M = 0.050
DY_M = 0.001

MODEL_ORIGIN_R_M = 11.8
MODEL_ORIGIN_PHI_DEG = 90.0

OUTPUT_FILE = "FD_top_90deg_Xradial_3D.table"
WRITE_PYFFA_HEADER = True


def build_pyffa_header(r_axis, phi_axis_deg, y_axis):
    """Metadata consumed by PyFFA after OPERA has filled Bx/By/Bz."""
    return [
        "# PyFFA_BFIELD3D_GRID version=1",
        "# PyFFA source_format=OPERA_TABLE",
        "# PyFFA table_role=opera_query_grid",
        "# PyFFA data_order=y_r_phi",
        "# PyFFA coordinate_columns=X,Y,Z",
        "# PyFFA expected_field_columns=Bx,By,Bz",
        "# PyFFA coordinate_unit=m",
        "# PyFFA field_unit=T",
        "# PyFFA local_x_axis=radial",
        "# PyFFA local_y_axis=vertical",
        "# PyFFA local_z_axis=tangential",
        "# PyFFA cylindrical_components=Br,Bphi,Bz",
        "# PyFFA midplane_symmetry=Bz_even_Br_Bphi_odd",
        f"# PyFFA nsectors={N_CELLS}",
        f"# PyFFA cell_center_deg={CELL_CENTER_DEG:.12g}",
        f"# PyFFA cell_width_deg={CELL_WIDTH_DEG:.12g}",
        f"# PyFFA r_min_m={r_axis[0]:.12g}",
        f"# PyFFA r_max_m={r_axis[-1]:.12g}",
        f"# PyFFA dr_m={DR_M:.12g}",
        f"# PyFFA nr={len(r_axis)}",
        f"# PyFFA phi_min_deg={phi_axis_deg[0]:.12g}",
        f"# PyFFA phi_max_deg={phi_axis_deg[-1]:.12g}",
        f"# PyFFA dphi_deg={DPHI_DEG:.12g}",
        f"# PyFFA nphi={len(phi_axis_deg)}",
        f"# PyFFA y_min_m={y_axis[0]:.12g}",
        f"# PyFFA y_max_m={y_axis[-1]:.12g}",
        f"# PyFFA dy_m={DY_M:.12g}",
        f"# PyFFA ny={len(y_axis)}",
        f"# PyFFA model_origin_phi_deg={MODEL_ORIGIN_PHI_DEG:.12g}",
        f"# PyFFA model_center_radius_m={MODEL_ORIGIN_R_M:.12g}",
    ]


# =========================================================
# Generate grid
# =========================================================

r_axis = np.arange(R_MIN_M, R_MAX_M + DR_M / 2.0, DR_M)

phi_axis_deg = np.arange(
    PHI_MIN_DEG,
    PHI_MAX_DEG + DPHI_DEG / 2.0,
    DPHI_DEG,
)

y_axis = np.arange(
    Y_MIN_M,
    Y_MAX_M + DY_M / 2.0,
    DY_M,
)

yy, rr, phi_deg = np.meshgrid(
    y_axis,
    r_axis,
    phi_axis_deg,
    indexing="ij",
)

delta_phi_rad = np.deg2rad(
    phi_deg - MODEL_ORIGIN_PHI_DEG
)

# Opera local coordinates:
# X: radial
# Y: vertical
# Z: tangential
x_opera = rr * np.cos(delta_phi_rad) - MODEL_ORIGIN_R_M
y_opera = yy
z_opera = rr * np.sin(delta_phi_rad)

coordinates = np.column_stack((
    x_opera.ravel(),
    y_opera.ravel(),
    z_opera.ravel(),
))


# =========================================================
# Write Opera TABLE
# =========================================================

with open(OUTPUT_FILE, "w", encoding="ascii", newline="\n") as file:
    file.write(f"{len(coordinates)} 1 1 2\n")
    file.write("1 X [METRE]\n")
    file.write("2 Y [METRE]\n")
    file.write("3 Z [METRE]\n")
    if WRITE_PYFFA_HEADER:
        for line in build_pyffa_header(r_axis, phi_axis_deg, y_axis):
            file.write(line + "\n")
    file.write("0\n")
    np.savetxt(file, coordinates, fmt="%.12e")


# =========================================================
# Information
# =========================================================

print(f"Output file      : {OUTPUT_FILE}")
print(f"Azimuth range    : {phi_axis_deg[0]:.2f} to {phi_axis_deg[-1]:.2f} deg")
print(f"Radius range     : {r_axis[0]:.3f} to {r_axis[-1]:.3f} m")
print(f"Vertical range   : {y_axis[0] * 1000:.0f} to {y_axis[-1] * 1000:.0f} mm")
print(f"Radial points    : {len(r_axis)}")
print(f"Azimuthal points : {len(phi_axis_deg)}")
print(f"Vertical planes  : {len(y_axis)}")
print(f"Total points     : {len(coordinates)}")
