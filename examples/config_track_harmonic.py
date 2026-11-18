"""Run a small 300 MeV longitudinal tracking example with a second-harmonic cavity."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np

from config_Emap import config_data as emap_template
from config_track import cf as track_template
from FFAG_Bunch import FFAG_ManageBunchAttribute
from FFAG_Distribution import LoadSEOParams
from main_Emap import (
    build_frequency_curve_from_config,
    build_second_harmonic_map,
    build_spiral_map_from_config,
)
from main_track import main as track_bunch


BMAP_DIR = Path('Bmaps/Bmap_test')
SEO_FOLDER = 'resultsSEO'
OUTPUT_DIR = Path('output/harmonic_tracking')
ENERGY_MEV = 300.0
N_PARTICLES = 8
N_TURNS = 300
TIME_STEP_NS = 0.20
SEED = 417
SHOW_PLOT = True

MAIN_GAP_DEG = 25.0
MAIN_GAP_VOLTAGE_V = 100e3  # Per gap, with two main RF gaps.
N_CELLS_AFTER_MAIN = 1
SECOND_HARMONIC_VOLTAGE_V = 160e3  # V2/V1 = 0.8 for a total main RF voltage of 200 kV.
SECOND_HARMONIC_PHASE_DEG = 180.0


def plot_potential_with_trajectories(run_dir):
    """Plot the turn-averaged RF potential and the saved tracking trajectories."""
    run_dir = Path(run_dir)
    emap_config = json.loads((run_dir / 'config_Emap_harmonic.json').read_text(encoding='utf-8'))
    track_config = json.loads((run_dir / 'config_track_harmonic.json').read_text(encoding='utf-8'))
    emap = emap_config['Emap']
    second = emap_config['SecondHarmonic']
    dump = track_config['DumpPara']['modules'][0]
    sample_deg = dump['dump_azimuth'][0]
    phase_ref = np.deg2rad(sample_deg - emap['spiral_s_ref_deg']) * emap['harmonic']
    energy_ref = track_config['track']['start_EkMeV']
    v1 = emap['Ngap'] * emap['acc_voltage_start'] / 1e6  # Total main RF voltage per turn [MV].
    v2 = second['voltage_V'] / 1e6
    ratio = v2 / v1
    phase2 = np.deg2rad(second['phase_at_sync_deg'])

    trajectories = []
    for particle_id in dump['num_particles_to_dump_global']:
        path = Path(dump['save_folder']) / f'particle_{particle_id}_angle_{sample_deg:.1f}_rank_0.npz'
        with np.load(path) as data:
            particles = data['particles']
        phase = (particles[:, 9] - phase_ref + np.pi) % (2.0 * np.pi) - np.pi
        energy = particles[:, 5] - energy_ref
        trajectories.append((phase, energy))

    # Fit dphi/dturn = -a * DeltaE from the production tracking data.
    # This sets the energy-axis scale of the turn-averaged Hamiltonian contours.
    energy_mid = np.concatenate([0.5 * (e[1:] + e[:-1]) for p, e in trajectories])
    phase_step = np.concatenate([np.diff(np.unwrap(p)) for p, e in trajectories])
    drift_fit = np.polyfit(energy_mid, phase_step, 1)
    a = -drift_fit[0]
    phase_limit_deg = 1.15 * max(np.max(np.abs(np.rad2deg(p))) for p, e in trajectories)
    energy_limit = 1.15 * max(np.max(np.abs(e)) for p, e in trajectories)
    phase_axis = np.linspace(-np.deg2rad(phase_limit_deg), np.deg2rad(phase_limit_deg), 601)
    phase_deg = np.rad2deg(phase_axis)
    # U'(phi) = V(phi)/V1, with U(0) = 0.
    potential = 1.0 - np.cos(phase_axis) + 0.5 * ratio * (np.cos(phase2) - np.cos(2.0 * phase_axis + phase2))
    single_potential = 1.0 - np.cos(phase_axis)
    flat_potential = 1.0 - np.cos(phase_axis) - 0.25 * (1.0 - np.cos(2.0 * phase_axis))

    fig, (well_ax, orbit_ax) = plt.subplots(
        2, 1, figsize=(8.2, 8.0), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1.65]), layout='constrained',
    )
    well_ax.plot(phase_deg, single_potential, '--', color='0.5', label='Main RF only')
    well_ax.plot(phase_deg, potential, color='#125a94', lw=2, label=f'Current $V_2/V_1={ratio:g}$')
    well_ax.plot(phase_deg, flat_potential, ':', color='#a15b14', lw=2, label=r'Flat center, $V_2/V_1=0.5$')
    well_ax.set_ylabel(r'Normalized potential $U(\phi)$')
    well_ax.set_title(f'Turn-averaged RF potential at {energy_ref:g} MeV')
    well_ax.legend(frameon=False, fontsize=9)
    well_ax.grid(alpha=0.2)

    energy_axis = np.linspace(-energy_limit, energy_limit, 401)
    phase_grid, energy_grid = np.meshgrid(phase_axis, energy_axis)
    hamiltonian = a * energy_grid**2 / (2.0 * v1) + potential[np.newaxis, :]
    orbit_energy_levels = []
    trajectory_hamiltonians = []
    for phase, energy in trajectories:
        u = 1.0 - np.cos(phase) + 0.5 * ratio * (np.cos(phase2) - np.cos(2.0 * phase + phase2))
        h = a * energy**2 / (2.0 * v1) + u
        trajectory_hamiltonians.append(h)
        orbit_energy_levels.append(np.median(h))
    levels = np.unique(np.sort(orbit_energy_levels))
    orbit_ax.contour(np.rad2deg(phase_grid), energy_grid, hamiltonian, levels=levels,
                     colors='0.60', linewidths=1.1, linestyles='--')
    colors = plt.get_cmap('turbo')(np.linspace(0.1, 0.9, len(trajectories)))
    for (phase, energy), color in zip(trajectories, colors):
        line, = orbit_ax.plot(np.rad2deg(phase), energy, color=color, lw=1.2, alpha=0.9)
        orbit_ax.plot(np.rad2deg(phase[0]), energy[0], 'o', color=line.get_color(), ms=4)
    orbit_ax.set_xlim(-phase_limit_deg, phase_limit_deg)
    orbit_ax.set_ylim(-energy_limit, energy_limit)
    orbit_ax.set_xlabel(r'Fundamental RF phase offset $\Delta\phi$ [deg]')
    orbit_ax.set_ylabel(r'Kinetic energy offset $\Delta E_k$ [MeV]')
    orbit_ax.set_title('Tracking trajectories over Hamiltonian contours (gray dashed)', fontsize=11)
    orbit_ax.grid(alpha=0.15)
    fig.supxlabel(f'V1 = {v1 * 1e3:g} kV per turn, V2 = {v2 * 1e3:g} kV; '
                  'contour energy scale uses the tracked phase-slip coefficient', fontsize=9)
    path = run_dir / 'longitudinal_potential_with_trajectories.png'
    fig.savefig(path, dpi=180)
    print(f'Potential plot: {path.resolve()}')
    print(f'Fitted phase-slip coefficient: {a:.8g} rad / (MeV turn)')

    # Lift each saved trajectory onto H(phi, DeltaE) and project it onto the floor.
    projection_fig = plt.figure(figsize=(10, 8), layout='constrained')
    surface_ax = projection_fig.add_subplot(111, projection='3d', computed_zorder=False)
    h_min = np.min(hamiltonian)
    h_max = np.max(hamiltonian)
    floor = h_min - 0.30 * (h_max - h_min)
    surface_ax.plot_surface(np.rad2deg(phase_grid), energy_grid, hamiltonian,
                            rcount=55, ccount=75, cmap='coolwarm', alpha=0.22,
                            linewidth=0, antialiased=True, zorder=1)
    surface_ax.plot_wireframe(np.rad2deg(phase_grid), energy_grid, hamiltonian,
                              rstride=35, cstride=40, color='0.45', lw=0.45, alpha=0.25, zorder=2)
    surface_ax.plot(phase_deg, np.zeros_like(phase_axis), potential,
                    color='0.15', lw=2.2, label=r'Potential $U(\phi)=H(\phi,0)$', zorder=4)
    for (phase, energy), h, color in zip(trajectories, trajectory_hamiltonians, colors):
        surface_ax.plot(np.rad2deg(phase), energy, h, color=color, lw=1.5, zorder=5)
        surface_ax.plot(np.rad2deg(phase), energy, np.full_like(h, floor),
                        color=color, lw=1.3, zorder=3)
        surface_ax.plot([np.rad2deg(phase[0])] * 2, [energy[0]] * 2, [floor, h[0]],
                        color=color, lw=0.8, ls=':', alpha=0.6, zorder=3)
    surface_ax.set_xlabel(r'RF phase offset $\Delta\phi$ [deg]', labelpad=10)
    surface_ax.set_ylabel(r'Energy offset $\Delta E_k$ [MeV]', labelpad=10)
    surface_ax.set_zlabel(r'Normalized Hamiltonian $H$', labelpad=10)
    surface_ax.set_xlim(-phase_limit_deg, phase_limit_deg)
    surface_ax.set_ylim(-energy_limit, energy_limit)
    surface_ax.set_zlim(floor, h_max)
    surface_ax.view_init(elev=25, azim=-58)
    surface_ax.set_box_aspect((1.5, 1, 0.9))
    surface_ax.legend(loc='upper left', frameon=False)
    surface_ax.set_title(f'Hamiltonian surface and trajectory projections, $V_2/V_1={ratio:g}$\n'
                         'Colored curves on the surface; phase-space projections on the floor')
    projection_fig.supxlabel('Turn-averaged model; phase-slip coefficient fitted from tracking', fontsize=9)
    projection_path = run_dir / 'hamiltonian_projection.png'
    projection_fig.savefig(projection_path, dpi=180)
    print(f'Hamiltonian projection: {projection_path.resolve()}')

    # Show three fundamental RF periods on the extended phase axis.
    wide_phase = np.linspace(-3.0 * np.pi, 3.0 * np.pi, 2401)
    wide_deg = np.rad2deg(wide_phase)
    wide_potential = 1.0 - np.cos(wide_phase) + 0.5 * ratio * (
        np.cos(phase2) - np.cos(2.0 * wide_phase + phase2)
    )
    wide_energy_limit = 1.12 * np.sqrt(2.0 * v1 * np.ptp(wide_potential) / a)
    wide_energy = np.linspace(-wide_energy_limit, wide_energy_limit, 601)
    wide_hamiltonian = a * wide_energy[:, np.newaxis]**2 / (2.0 * v1) + wide_potential
    wide_fig, (potential_ax, contour_ax) = plt.subplots(
        2, 1, figsize=(12, 7.6), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 2]), layout='constrained',
    )
    potential_ax.plot(wide_deg, wide_potential, color='#125a94', lw=1.8)
    potential_ax.set_ylabel(r'Potential $U(\phi)$')
    potential_ax.set_title(f'Three RF periods on the extended phase axis, $V_2/V_1={ratio:g}$')
    contour_levels = np.linspace(np.min(wide_potential) + 0.025, np.max(wide_potential) - 0.025, 15)
    contour_ax.contour(wide_deg, wide_energy, wide_hamiltonian,
                       levels=contour_levels, colors='0.72', linewidths=0.65)
    # Locate potential maxima on this uniform grid to draw the saddle-energy contours.
    maxima = np.flatnonzero((wide_potential[1:-1] > wide_potential[:-2]) &
                           (wide_potential[1:-1] > wide_potential[2:])) + 1
    saddle_levels = np.unique(np.round(wide_potential[maxima], 10))
    contour_ax.contour(wide_deg, wide_energy, wide_hamiltonian,
                       levels=saddle_levels, colors='#9b391e', linewidths=1.3)
    for (phase, energy), color in zip(trajectories, colors):
        contour_ax.plot(np.rad2deg(phase), energy, color=color, lw=1.3)
    for boundary in (-540, -180, 180, 540):
        potential_ax.axvline(boundary, color='0.4', ls='--', lw=0.8)
        contour_ax.axvline(boundary, color='0.4', ls='--', lw=0.8)
    for period in (-1, 0, 1):
        potential_ax.text(360 * period, 0.85 * np.max(wide_potential), f'RF period {period:+d}',
                          ha='center', fontsize=10)
    potential_ax.grid(alpha=0.15)
    contour_ax.set_xlim(-540, 540)
    contour_ax.set_xticks(np.arange(-540, 541, 90))
    contour_ax.set_xlabel(r'Extended fundamental RF phase $\phi$ [deg]')
    contour_ax.set_ylabel(r'Energy offset $\Delta E_k$ [MeV]')
    contour_ax.set_title('Hamiltonian contours and saddle levels; colored curves are tracked particles', fontsize=11)
    contour_ax.grid(alpha=0.15)
    wide_fig.supxlabel(f'Fundamental harmonic h = {emap["harmonic"]}; '
                        'adjacent periods are periodic copies; tracking is shown only in period 0', fontsize=9)
    wide_path = run_dir / 'hamiltonian_extended_phase.png'
    wide_fig.savefig(wide_path, dpi=180)
    print(f'Extended Hamiltonian contours: {wide_path.resolve()}')
    return fig


def main():
    run_dir = OUTPUT_DIR / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True)
    map_dir = run_dir / 'Emaps'
    curve_path = map_dir / 'FrequencyCurve_LSC.txt'
    dump_dir = run_dir / 'PositionDump'
    print(f'Output directory: {run_dir.resolve()}', flush=True)

    seo_file = BMAP_DIR / SEO_FOLDER / 'SEO_ini.txt'
    seo_ref, _, _ = LoadSEOParams(str(seo_file), ENERGY_MEV)
    turn_period = seo_ref[3]
    bmap_config = json.loads((BMAP_DIR / 'config_Bmap.json').read_text(encoding='utf-8'))
    cell_deg = 360.0 / bmap_config['Bmap']['NSector']
    # Sample one full cell downstream of the second-harmonic gap.
    phase_at_sample_deg = (N_CELLS_AFTER_MAIN + 1) * cell_deg
    sample_deg = (MAIN_GAP_DEG + phase_at_sample_deg) % 360.0

    # Generate both maps in the run directory using a fixed-energy RF program.
    emap_config = deepcopy(emap_template)
    emap_config['config'].update(filename=str(curve_path), plot_frequency_curve=False, plot_spiral_map=False)
    emap_config['machine'].update(energy_inj=ENERGY_MEV, energy_ext=ENERGY_MEV)
    emap_config['Emap'].update(
        Bmap_fold=str(BMAP_DIR), SEO_folder=SEO_FOLDER, emap_save_dir=str(map_dir),
        gap_model='spiral', Ngap=2, harmonic=1, gap_width=1.0,
        spiral_alpha_deg=0.0, spiral_s_ref_deg=MAIN_GAP_DEG, spiral_FWHM=0.60, spiral_U=1.0,
        acc_phase_start=0.0, acc_phase_end=0.0,
        acc_voltage_start=MAIN_GAP_VOLTAGE_V, acc_voltage_end=MAIN_GAP_VOLTAGE_V,
        num_turns=N_TURNS + 3,
    )
    emap_config['SecondHarmonic'].update(
        enable=True, n_cells_after_main=N_CELLS_AFTER_MAIN, spiral_FWHM=0.20,
        voltage_V=SECOND_HARMONIC_VOLTAGE_V, phase_at_sync_deg=SECOND_HARMONIC_PHASE_DEG,
    )
    (run_dir / 'config_Emap_harmonic.json').write_text(json.dumps(emap_config, indent=2), encoding='utf-8')
    config = emap_config['config']
    emap = emap_config['Emap']
    harmonic_header = build_second_harmonic_map(config, emap, emap_config['SecondHarmonic'])
    build_frequency_curve_from_config(config, emap, emap_config['machine'], harmonic_header)
    build_spiral_map_from_config(config, emap)

    # Use the same JSON format and tracking entry point as config_track.py.
    track_config = deepcopy(track_template)
    start_time = -100e-9
    time_step = TIME_STEP_NS * 1e-9
    dump_end = N_TURNS * turn_period
    # PositionDump writes its buffered data on the first step after dump_end.
    track_end = dump_end + 0.5 * turn_period
    track_config['track'].update(
        start_EkMeV=ENERGY_MEV, start_azimuth=0.0, time_start=start_time, time_step=time_step, solver_type=1,
        stop_condition=dict(max_stepsN=int(np.ceil((track_end - start_time) / time_step)),
                            max_turn=N_TURNS + 2, max_time=track_end),
    )
    track_config['BmapAndSEO'].update(maps=str(BMAP_DIR), SEO_FileName=SEO_FOLDER, max_order=0)
    track_config['Emap'].update(maps=str(curve_path), enable=True)
    track_config['SC'].update(enable_SC=False, grid_size_rfz=[8, 8, 8, 10])
    track_config['Paint']['enable'] = False
    track_config['BumpKicker'] = dict(enable=False, kickers=[])
    track_config['BunchPara'].update(
        ParticleNum=N_PARTICLES, TransverseREmit=0.01, TransverseZEmit=0.01,
        LongitudeT=80.0, LongitudeDEk=1.2, LongitudeDistType='waterbag',
    )
    track_config['DumpPara']['modules'] = [dict(
        type='PositionDump', start_time=start_time, end_time=dump_end,
        num_particles_to_dump_global=list(range(N_PARTICLES)),
        dump_azimuth=[sample_deg], save_folder=str(dump_dir),
    )]
    track_path = run_dir / 'config_track_harmonic.json'
    track_path.write_text(json.dumps(track_config, indent=2), encoding='utf-8')

    random.seed(SEED)
    np.random.seed(SEED)
    plot_backend = plt.get_backend()
    plt.switch_backend('Agg')  # Suppress the initial-distribution window during tracking.
    track_bunch(str(track_path))
    plt.close('all')
    plt.switch_backend(plot_backend)

    columns = FFAG_ManageBunchAttribute().Attribute
    phase_ref = np.deg2rad(phase_at_sample_deg)
    fig, ax = plt.subplots(figsize=(7.6, 5.5), layout='constrained')
    for particle_id in range(N_PARTICLES):
        dump_path = dump_dir / f'particle_{particle_id}_angle_{sample_deg:.1f}_rank_0.npz'
        with np.load(dump_path) as data:
            particles = data['particles']
        phase = particles[:, columns['RF_phase']] - phase_ref
        phase_deg = np.rad2deg((phase + np.pi) % (2.0 * np.pi) - np.pi)
        energy_offset = particles[:, columns['Ek']] - ENERGY_MEV
        line, = ax.plot(phase_deg, energy_offset, lw=1.1, alpha=0.8)
        ax.plot(phase_deg[0], energy_offset[0], 'o', ms=4, color=line.get_color())
        print(f'Particle {particle_id}: {len(particles)} turn-by-turn samples')

    ax.set_xlabel(r'Fundamental RF phase offset $\Delta\phi$ [deg]')
    ax.set_ylabel(r'Kinetic energy offset $\Delta E_k$ [MeV]')
    ax.set_title(f'Dual-harmonic RF at {ENERGY_MEV:g} MeV, sampled at {sample_deg:g}°\n'
                 f'{N_PARTICLES} particles over {N_TURNS} turns (dots mark first samples)')
    ax.axhline(0.0, color='0.72', lw=0.8, zorder=0)
    ax.axvline(0.0, color='0.72', lw=0.8, zorder=0)
    ax.grid(alpha=0.2)
    plot_path = run_dir / 'longitudinal_phase_space.png'
    fig.savefig(plot_path, dpi=180)
    print(f'Phase-space plot: {plot_path.resolve()}')
    plot_potential_with_trajectories(run_dir)
    if SHOW_PLOT:
        plt.show()


if __name__ == '__main__':
    main()
