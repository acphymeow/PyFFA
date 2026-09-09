"""Calculate closed-orbit Dx and Dx' in the local Frenet frame.

Different-energy closed orbits are projected onto the normal of the reference
closed orbit.  This follows the geometry used by the slice SC projection more
closely than differentiating the cylindrical radius R at fixed azimuth.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bundle_paths import REPO_ROOT as REPO

# --------------------------- editable parameters ---------------------------
SEO = REPO / "Bmap_FD16" / "resultsSEO_noError_X"
REFERENCE_EK_MEV = 300.0
ENERGY_HALF_STEP_MEV = 1.0
OUTPUT = HERE / "outputs" / "closed_orbit_dispersion_x"
# --------------------------------------------------------------------------


def energy_row(matrix: np.ndarray, energy_axis: np.ndarray,
               energy: float) -> np.ndarray:
    """Interpolate one complete azimuth row at the requested energy."""
    return np.array([
        np.interp(energy, energy_axis, matrix[:, j])
        for j in range(matrix.shape[1])
    ])


def periodic_interp(phi_axis: np.ndarray, values: np.ndarray,
                    query: np.ndarray) -> np.ndarray:
    return np.interp(np.mod(query, 2.0 * np.pi), phi_axis, values,
                     period=2.0 * np.pi)


def project_closed_orbit(
    energy: float,
    phi: np.ndarray,
    energy_axis: np.ndarray,
    radius_matrix: np.ndarray,
    velocity_pr_matrix: np.ndarray,
    reference_radius: np.ndarray,
    reference_pr_geom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local (x, x') of an off-energy closed orbit.

    ``velocity_pr_matrix`` stores vr/v.  The reference geometric slope is
    r'/r=(vr/v)/sqrt(1-(vr/v)^2).  The off-energy velocity is rotated from its
    own cylindrical basis at phi into the reference basis at theta_star before
    the local normal/tangent ratio x' is formed.
    """
    radius = energy_row(radius_matrix, energy_axis, energy)
    velocity_pr = energy_row(velocity_pr_matrix, energy_axis, energy)
    velocity_pphi = np.sqrt(np.maximum(1.0 - velocity_pr**2, 1e-30))

    reference_rprime = reference_radius * reference_pr_geom
    denominator = reference_radius**2 + reference_rprime**2
    delta_theta = ((radius - reference_radius) * reference_rprime
                   / np.maximum(denominator, 1e-30))
    theta_star = np.mod(phi + delta_theta, 2.0 * np.pi)

    radius_star = periodic_interp(phi, reference_radius, theta_star)
    pr_geom_star = periodic_interp(phi, reference_pr_geom, theta_star)
    norm_star = np.sqrt(1.0 + pr_geom_star**2)
    x = (radius - radius_star) / norm_star

    # Rotate the off-energy velocity components from basis(phi) to
    # basis(theta_star), then resolve them along the reference normal/tangent.
    basis_angle = phi - theta_star
    cos_angle = np.cos(basis_angle)
    sin_angle = np.sin(basis_angle)
    velocity_r_star = (velocity_pr * cos_angle
                       - velocity_pphi * sin_angle)
    velocity_phi_star = (velocity_pr * sin_angle
                         + velocity_pphi * cos_angle)
    velocity_normal = velocity_r_star - pr_geom_star * velocity_phi_star
    velocity_tangent = pr_geom_star * velocity_r_star + velocity_phi_star
    xprime = velocity_normal / np.maximum(velocity_tangent, 1e-30)
    return x, xprime


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def centered_dispersion_curves(
    seo: Path = SEO,
    reference_ek_mev: float = REFERENCE_EK_MEV,
    energy_half_step_mev: float = ENERGY_HALF_STEP_MEV,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return native SEO azimuth, centered Dx and centered Dx'."""
    radius_table = np.loadtxt(seo / "SEO_r.txt", skiprows=1)
    pr_table = np.loadtxt(seo / "SEO_pr.txt", skiprows=1)
    phi = radius_table[0, 1:]
    energy_axis = radius_table[1:, 0]
    radius_matrix = radius_table[1:, 1:]
    velocity_pr_matrix = pr_table[1:, 1:]
    if not np.allclose(phi, pr_table[0, 1:]):
        raise ValueError("SEO_r and SEO_pr azimuth axes differ")
    if not np.allclose(energy_axis, pr_table[1:, 0]):
        raise ValueError("SEO_r and SEO_pr energy axes differ")

    reference_radius = energy_row(radius_matrix, energy_axis, reference_ek_mev)
    reference_velocity_pr = energy_row(
        velocity_pr_matrix, energy_axis, reference_ek_mev
    )
    reference_pr_geom = reference_velocity_pr / np.sqrt(
        np.maximum(1.0 - reference_velocity_pr**2, 1e-30)
    )
    x_minus, xp_minus = project_closed_orbit(
        reference_ek_mev - energy_half_step_mev, phi, energy_axis,
        radius_matrix, velocity_pr_matrix, reference_radius, reference_pr_geom,
    )
    x_plus, xp_plus = project_closed_orbit(
        reference_ek_mev + energy_half_step_mev, phi, energy_axis,
        radius_matrix, velocity_pr_matrix, reference_radius, reference_pr_geom,
    )
    scale = 1.0 / (2.0 * energy_half_step_mev)
    return phi, (x_plus - x_minus) * scale, (xp_plus - xp_minus) * scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    radius_table = np.loadtxt(SEO / "SEO_r.txt", skiprows=1)
    pr_table = np.loadtxt(SEO / "SEO_pr.txt", skiprows=1)
    phi = radius_table[0, 1:]
    energy_axis = radius_table[1:, 0]
    radius_matrix = radius_table[1:, 1:]
    velocity_pr_matrix = pr_table[1:, 1:]
    if not np.allclose(phi, pr_table[0, 1:]):
        raise ValueError("SEO_r and SEO_pr azimuth axes differ")
    if not np.allclose(energy_axis, pr_table[1:, 0]):
        raise ValueError("SEO_r and SEO_pr energy axes differ")

    e0 = REFERENCE_EK_MEV
    de = ENERGY_HALF_STEP_MEV
    reference_radius = energy_row(radius_matrix, energy_axis, e0)
    reference_velocity_pr = energy_row(velocity_pr_matrix, energy_axis, e0)
    reference_pr_geom = reference_velocity_pr / np.sqrt(
        np.maximum(1.0 - reference_velocity_pr**2, 1e-30)
    )

    x_minus, xp_minus = project_closed_orbit(
        e0 - de, phi, energy_axis, radius_matrix, velocity_pr_matrix,
        reference_radius, reference_pr_geom,
    )
    x_zero, xp_zero = project_closed_orbit(
        e0, phi, energy_axis, radius_matrix, velocity_pr_matrix,
        reference_radius, reference_pr_geom,
    )
    x_plus, xp_plus = project_closed_orbit(
        e0 + de, phi, energy_axis, radius_matrix, velocity_pr_matrix,
        reference_radius, reference_pr_geom,
    )

    dx_left = (x_zero - x_minus) / de
    dx_right = (x_plus - x_zero) / de
    dx_center = (x_plus - x_minus) / (2.0 * de)
    dxp_left = (xp_zero - xp_minus) / de
    dxp_right = (xp_plus - xp_zero) / de
    dxp_center = (xp_plus - xp_minus) / (2.0 * de)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "closed_orbit_Dx_Dxp_300MeV.csv"
    columns = (
        "phi_rad", "phi_deg", "reference_radius_m", "reference_pr_geom",
        "Dx_left_m_per_MeV", "Dx_center_m_per_MeV",
        "Dx_right_m_per_MeV", "Dxp_left_rad_per_MeV",
        "Dxp_center_rad_per_MeV", "Dxp_right_rad_per_MeV",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(
            phi, np.rad2deg(phi), reference_radius, reference_pr_geom,
            dx_left, dx_center, dx_right,
            dxp_left, dxp_center, dxp_right,
        ))

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True,
                             constrained_layout=True)
    for values, label, style in (
        (dx_left, "left", "--"),
        (dx_center, "centered", "-"),
        (dx_right, "right", ":"),
    ):
        axes[0].plot(np.rad2deg(phi), values * 1e3, style, label=label)
    for values, label, style in (
        (dxp_left, "left", "--"),
        (dxp_center, "centered", "-"),
        (dxp_right, "right", ":"),
    ):
        axes[1].plot(np.rad2deg(phi), values * 1e3, style, label=label)
    axes[0].set_ylabel(r"$D_x$ [mm/MeV]")
    axes[1].set_ylabel(r"$D'_x$ [mrad/MeV]")
    axes[1].set_xlabel(r"Azimuth $\phi$ [deg]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, ncol=3)
    fig.savefig(OUTPUT / "closed_orbit_Dx_Dxp_300MeV.png", dpi=220)
    fig.savefig(OUTPUT / "closed_orbit_Dx_Dxp_300MeV.pdf")

    i0 = int(np.argmin(np.abs(phi)))
    print(f"reference energy = {e0:.6f} MeV, half step = {de:.6f} MeV")
    print("at phi=0 deg:")
    print(
        "  Dx  left/center/right [mm/MeV] = "
        f"{dx_left[i0] * 1e3:.9f}, {dx_center[i0] * 1e3:.9f}, "
        f"{dx_right[i0] * 1e3:.9f}"
    )
    print(
        "  Dx' left/center/right [mrad/MeV] = "
        f"{dxp_left[i0] * 1e3:.9f}, {dxp_center[i0] * 1e3:.9f}, "
        f"{dxp_right[i0] * 1e3:.9f}"
    )
    print("centered derivative over the full ring:")
    print(
        "  Dx  min/max/mean/RMS [mm/MeV] = "
        f"{dx_center.min() * 1e3:.9f}, {dx_center.max() * 1e3:.9f}, "
        f"{dx_center.mean() * 1e3:.9f}, {rms(dx_center) * 1e3:.9f}"
    )
    print(
        "  Dx' min/max/mean/RMS [mrad/MeV] = "
        f"{dxp_center.min() * 1e3:.9f}, {dxp_center.max() * 1e3:.9f}, "
        f"{dxp_center.mean() * 1e3:.9f}, {rms(dxp_center) * 1e3:.9f}"
    )
    print(f"CSV: {csv_path}")
    print(f"plot: {OUTPUT / 'closed_orbit_Dx_Dxp_300MeV.png'}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
