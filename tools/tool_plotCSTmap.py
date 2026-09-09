# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Settings
# =========================================================

FIELD_FILE = "FD16_cell.table"

R_MIN_M = 10.8
R_MAX_M = 12.2
DR_M = 0.005

PHI_MIN_DEG = 78.75
PHI_MAX_DEG = 101.25
DPHI_DEG = 0.1

Y_MIN_M = 0.000
Y_MAX_M = 0.050
DY_M = 0.001

MODEL_ORIGIN_PHI_DEG = 90.0

Y_PLOT_M = 0.010   # non-mid-plane: 10 mm
R_COMPARE_M = [11.2, 11.5, 11.8, 12.0]


# =========================================================
# Read Opera TABLE
# =========================================================

with open(FIELD_FILE, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

data_start = next(i for i, line in enumerate(lines) if line.strip() == "0") + 1
data = np.loadtxt(lines[data_start:])


# =========================================================
# Restore axes
# =========================================================

r_axis = np.arange(R_MIN_M, R_MAX_M + DR_M / 2.0, DR_M)
phi_axis_deg = np.arange(PHI_MIN_DEG, PHI_MAX_DEG + DPHI_DEG / 2.0, DPHI_DEG)
y_axis = np.arange(Y_MIN_M, Y_MAX_M + DY_M / 2.0, DY_M)

ny = len(y_axis)
nr = len(r_axis)
nphi = len(phi_axis_deg)

print("Loaded points   :", len(data))
print("Expected points :", ny * nr * nphi)


# =========================================================
# Restore 3D matrices
# Data order: Y plane -> R -> phi
# =========================================================

x = data[:, 0].reshape(ny, nr, nphi)
y = data[:, 1].reshape(ny, nr, nphi)
z = data[:, 2].reshape(ny, nr, nphi)

bx = data[:, 3].reshape(ny, nr, nphi)
by = data[:, 4].reshape(ny, nr, nphi)
bz = data[:, 5].reshape(ny, nr, nphi)


# =========================================================
# Convert local Cartesian components to global cylindrical
# Local:
#   X = radial
#   Y = vertical
#   Z = tangential
# Global cylindrical:
#   BR, Bphi, Bz
# =========================================================

delta_phi_rad = np.deg2rad(phi_axis_deg - MODEL_ORIGIN_PHI_DEG)

cos_dphi = np.cos(delta_phi_rad)[None, None, :]
sin_dphi = np.sin(delta_phi_rad)[None, None, :]

br = bx * cos_dphi + bz * sin_dphi
bphi = -bx * sin_dphi + bz * cos_dphi
bz_global = by

print("Matrix shape:", br.shape)


# =========================================================
# Select planes
# =========================================================

iy_mid = 0
iy_plot = np.argmin(np.abs(y_axis - Y_PLOT_M))

print("Mid-plane Y      =", y_axis[iy_mid], "m")
print("Selected Y plane =", y_axis[iy_plot], "m")


# =========================================================
# 1. Mid-plane 2D map: Bz
# =========================================================

plt.figure(figsize=(10, 6))
mesh = plt.pcolormesh(
    x[iy_mid],
    z[iy_mid],
    bz_global[iy_mid],
    shading="auto",
)
plt.colorbar(mesh, label=r"$B_z$ (T)")
plt.xlabel("Opera local X (m)")
plt.ylabel("Opera local Z (m)")
plt.title(r"Mid-plane map: $B_z(X,Z)$ at $Y=0$")
plt.axis("equal")
plt.tight_layout()
plt.show()


# =========================================================
# 2. Off-mid-plane 2D map: BR
# =========================================================

plt.figure(figsize=(10, 6))
mesh = plt.pcolormesh(
    x[iy_plot],
    z[iy_plot],
    br[iy_plot],
    shading="auto",
)
plt.colorbar(mesh, label=r"$B_R$ (T)")
plt.xlabel("Opera local X (m)")
plt.ylabel("Opera local Z (m)")
plt.title(rf"Off-mid-plane map: $B_R(X,Z)$ at $Y={y_axis[iy_plot]*1000:.0f}$ mm")
plt.axis("equal")
plt.tight_layout()
plt.show()


# =========================================================
# 3. Off-mid-plane 2D map: Bphi
# =========================================================

plt.figure(figsize=(10, 6))
mesh = plt.pcolormesh(
    x[iy_plot],
    z[iy_plot],
    bphi[iy_plot],
    shading="auto",
)
plt.colorbar(mesh, label=r"$B_\phi$ (T)")
plt.xlabel("Opera local X (m)")
plt.ylabel("Opera local Z (m)")
plt.title(rf"Off-mid-plane map: $B_\phi(X,Z)$ at $Y={y_axis[iy_plot]*1000:.0f}$ mm")
plt.axis("equal")
plt.tight_layout()
plt.show()


# =========================================================
# 4. Mid-plane 1D curves: Bz(phi) at different radii
# =========================================================

plt.figure(figsize=(10, 6))

for r0 in R_COMPARE_M:
    ir = np.argmin(np.abs(r_axis - r0))
    plt.plot(
        phi_axis_deg,
        bz_global[iy_mid, ir, :],
        label=f"R = {r_axis[ir]:.3f} m",
    )

plt.xlabel(r"Azimuth $\phi$ (deg)")
plt.ylabel(r"$B_z$ (T)")
plt.title(r"Mid-plane: $B_z(\phi)$ at different radii")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# 5. Off-mid-plane 1D curves: BR(phi) at different radii
# =========================================================

plt.figure(figsize=(10, 6))

for r0 in R_COMPARE_M:
    ir = np.argmin(np.abs(r_axis - r0))
    plt.plot(
        phi_axis_deg,
        br[iy_plot, ir, :],
        label=f"R = {r_axis[ir]:.3f} m",
    )

plt.xlabel(r"Azimuth $\phi$ (deg)")
plt.ylabel(r"$B_R$ (T)")
plt.title(rf"Off-mid-plane: $B_R(\phi)$ at different radii, $Y={y_axis[iy_plot]*1000:.0f}$ mm")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# 6. Off-mid-plane 1D curves: Bphi(phi) at different radii
# =========================================================

plt.figure(figsize=(10, 6))

for r0 in R_COMPARE_M:
    ir = np.argmin(np.abs(r_axis - r0))
    plt.plot(
        phi_axis_deg,
        bphi[iy_plot, ir, :],
        label=f"R = {r_axis[ir]:.3f} m",
    )

plt.xlabel(r"Azimuth $\phi$ (deg)")
plt.ylabel(r"$B_\phi$ (T)")
plt.title(rf"Off-mid-plane: $B_\phi(\phi)$ at different radii, $Y={y_axis[iy_plot]*1000:.0f}$ mm")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()