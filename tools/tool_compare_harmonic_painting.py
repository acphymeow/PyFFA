"""Compare longitudinal offset injection with and without a second-harmonic cavity, without SC."""

from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d


BMAP_DIR = Path('Bmaps/Bmap_test')
OUTPUT_DIR = Path('output/harmonic_painting')
REFERENCE_ENERGY_MEV = 300.0
REST_ENERGY_MEV = 938.2720813
MOMENTUM_OFFSET = -0.0015
RMS_MOMENTUM_SPREAD = 0.001
CHOPPING_FRACTION = 0.50
INJECTION_DURATION_US = 500.0
PARTICLES_PER_TURN = 128
POST_INJECTION_TURNS = 500
MAIN_GAP_VOLTAGE_V = 100e3
SECOND_HARMONIC_RATIO = 0.8
N_CELLS_AFTER_MAIN = 1
RECORD_EVERY = 5
N_BINS = 128
SMOOTH_SIGMA_BINS = 1.5
SEED = 417
SHOW_PLOTS = True


@njit(cache=True)
def track_offset_injection(injection_phase, injection_energy, injection_turns, total_turns,
                           per_turn, v_gap, v_second, cell_fraction, k, f_ref, p_ref, beta_ref,
                           snapshots_at):
    """Track thin RF gaps with the scaling-FFA energy-dependent revolution period."""
    count = len(injection_phase)
    phase = injection_phase.copy()
    energy = injection_energy.copy()
    n_records = total_turns // RECORD_EVERY + 3
    histograms = np.zeros((n_records, N_BINS))
    moments = np.zeros((n_records, 5))
    snapshots = np.full((len(snapshots_at), count, 2), np.nan)
    n_active = 0
    record = 0
    snap = 0
    two_pi = 2.0 * np.pi
    fractions = (cell_fraction, 0.5 - cell_fraction, 0.5)
    voltages = (v_gap, v_second, v_gap)
    exponent = -1.0 / (k + 1.0)

    for turn in range(total_turns + 1):
        if turn < injection_turns:
            n_active += per_turn
        if turn % RECORD_EVERY == 0 or turn == injection_turns or turn == total_turns:
            sum_phase2, sum_energy, sum_energy2 = 0.0, 0.0, 0.0
            for i in range(n_active):
                wrapped = (phase[i] + np.pi) % two_pi - np.pi
                index = int((wrapped + np.pi) / two_pi * N_BINS)
                histograms[record, index] += 1
                sum_phase2 += wrapped * wrapped
                sum_energy += energy[i]
                sum_energy2 += energy[i] * energy[i]
            moments[record, 0] = turn
            moments[record, 1] = n_active
            moments[record, 2] = np.sqrt(sum_phase2 / n_active)
            moments[record, 3] = sum_energy / n_active
            moments[record, 4] = np.sqrt(sum_energy2 / n_active - (sum_energy / n_active)**2)
            record += 1
        if snap < len(snapshots_at) and turn == snapshots_at[snap]:
            snapshots[snap, :n_active, 0] = phase[:n_active]
            snapshots[snap, :n_active, 1] = energy[:n_active]
            snap += 1
        if turn == total_turns:
            break

        # Observe immediately upstream of main gap 1. The second-harmonic gap
        # follows by n lattice cells, and main gap 2 follows by half a turn.
        for event in range(3):
            for i in range(n_active):
                rf_phase = phase[i]
                if event == 1:
                    kick = -voltages[event] * np.sin(2.0 * rf_phase)
                else:
                    kick = voltages[event] * np.sin(rf_phase)
                energy[i] += kick
                momentum = np.sqrt(energy[i] * (energy[i] + 2.0 * REST_ENERGY_MEV))
                beta = momentum / (energy[i] + REST_ENERGY_MEV)
                frequency = f_ref * beta / beta_ref * (momentum / p_ref)**exponent
                phase[i] += two_pi * fractions[event] * (f_ref / frequency - 1.0)

    return histograms[:record], moments[:record], snapshots


def main():
    run_dir = OUTPUT_DIR / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True)
    bmap = json.loads((BMAP_DIR / 'config_Bmap.json').read_text(encoding='utf-8'))['Bmap']
    seo = np.loadtxt(BMAP_DIR / 'resultsSEO/SEO_ini.txt', skiprows=2)
    f_ref = np.interp(REFERENCE_ENERGY_MEV, seo[:, 1], seo[:, 4])
    p_ref = np.sqrt(REFERENCE_ENERGY_MEV * (REFERENCE_ENERGY_MEV + 2.0 * REST_ENERGY_MEV))
    beta_ref = p_ref / (REFERENCE_ENERGY_MEV + REST_ENERGY_MEV)
    k = bmap['k_value']
    momentum_nodes = np.sqrt(seo[:, 1] * (seo[:, 1] + 2.0 * REST_ENERGY_MEV))
    beta_nodes = momentum_nodes / (seo[:, 1] + REST_ENERGY_MEV)
    predicted_frequency = f_ref * beta_nodes / beta_ref * (momentum_nodes / p_ref)**(-1.0 / (k + 1.0))
    scaling_error = np.max(np.abs(predicted_frequency / seo[:, 4] - 1.0))
    injection_turns = int(np.round(INJECTION_DURATION_US * 1e-6 * f_ref))
    total_turns = injection_turns + POST_INJECTION_TURNS
    snapshots_at = np.array([injection_turns // 4, injection_turns // 2,
                             injection_turns, total_turns], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    n_particles = injection_turns * PARTICLES_PER_TURN
    phase_in = rng.uniform(-np.pi * CHOPPING_FRACTION, np.pi * CHOPPING_FRACTION, n_particles)
    momentum_in = p_ref * (1.0 + MOMENTUM_OFFSET + RMS_MOMENTUM_SPREAD * rng.standard_normal(n_particles))
    energy_in = np.sqrt(momentum_in**2 + REST_ENERGY_MEV**2) - REST_ENERGY_MEV
    v_gap = MAIN_GAP_VOLTAGE_V / 1e6
    phase_axis = (np.arange(N_BINS) + 0.5) / N_BINS * 360.0 - 180.0
    settings = dict(model='1D thin-gap kick/drift, scaling FFA, no SC, stationary RF, no phase sweep',
                    bmap=str(BMAP_DIR), reference_energy_MeV=REFERENCE_ENERGY_MEV,
                    momentum_offset=MOMENTUM_OFFSET, rms_momentum_spread=RMS_MOMENTUM_SPREAD,
                    injected_mean_energy_MeV=float(np.mean(energy_in)), chopping_fraction=CHOPPING_FRACTION,
                    injection_turns=injection_turns, injection_duration_us=injection_turns/f_ref*1e6,
                    particles_per_turn=PARTICLES_PER_TURN, total_particles=n_particles,
                    post_injection_turns=POST_INJECTION_TURNS, main_gap_voltage_V=MAIN_GAP_VOLTAGE_V,
                    second_harmonic_ratio=SECOND_HARMONIC_RATIO, n_cells_after_main=N_CELLS_AFTER_MAIN,
                    histogram_bins=N_BINS, smoothing_sigma_bins=SMOOTH_SIGMA_BINS, seed=SEED,
                    scaling_law_max_relative_error_on_SEO=float(scaling_error),
                    bunching_factor_definition='mean current over a full fundamental RF period / peak current')
    (run_dir / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    summary = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for name, ratio, color in [('Main RF only', 0.0, '#3366aa'),
                                ('Dual harmonic, r=0.8', SECOND_HARMONIC_RATIO, '#ce542b')]:
        print(f'Tracking {name}: {n_particles} particles, {injection_turns} injection turns', flush=True)
        hist, moments, snapshots = track_offset_injection(
            phase_in, energy_in, injection_turns, total_turns, PARTICLES_PER_TURN,
            v_gap, 2.0*v_gap*ratio, N_CELLS_AFTER_MAIN/bmap['NSector'],
            k, f_ref, p_ref, beta_ref, snapshots_at)
        smooth = gaussian_filter1d(hist, SMOOTH_SIGMA_BINS, axis=1, mode='wrap')
        bf_raw = np.mean(hist, axis=1) / np.max(hist, axis=1)
        bf = np.mean(smooth, axis=1) / np.max(smooth, axis=1)
        turns = moments[:, 0]
        time_us = turns / f_ref * 1e6
        at_injection_end = np.argmin(np.abs(turns - injection_turns))
        during_injection = (turns >= 20) & (turns < injection_turns)
        after = turns >= injection_turns
        stem = 'single' if ratio == 0 else 'dual'
        np.savez_compressed(run_dir / f'{stem}.npz', histograms=hist, moments=moments,
                            snapshots=snapshots, snapshot_turns=snapshots_at, phase_bin_centers_deg=phase_axis)
        np.savetxt(run_dir / f'{stem}_bunching_factor.csv',
                   np.column_stack((turns, time_us, bf, bf_raw, moments[:, 1:])), delimiter=',',
                   header='turn,time_us,Bf_smoothed,Bf_raw,N,sigma_phase_rad,mean_Ek_MeV,sigma_Ek_MeV', comments='')
        summary[stem] = dict(Bf_end_injection=float(bf[at_injection_end]),
                             Bf_end_tracking=float(bf[-1]),
                             Bf_min_during_injection_after_turn20=float(np.min(bf[during_injection])),
                             Bf_mean_after_injection=float(np.mean(bf[after])),
                             Bf_min_after_injection=float(np.min(bf[after])),
                             final_sigma_Ek_MeV=float(moments[-1, 4]),
                             final_histogram_count=int(np.sum(hist[-1])))
        axes[0, 0].plot(time_us, bf, color=color, label=name)
        axes[0, 1].plot(phase_axis, smooth[at_injection_end]/np.mean(smooth[at_injection_end]), color=color, label=name)
        axes[1, 0].plot(phase_axis, smooth[-1]/np.mean(smooth[-1]), color=color, label=name)
        ax = axes[1, 1]
        final = snapshots[-1, ::8]
        final_phase = (final[:, 0] + np.pi) % (2*np.pi)-np.pi
        ax.scatter(np.rad2deg(final_phase), final[:, 1]-REFERENCE_ENERGY_MEV,
                   s=1.0, alpha=0.3, color=color, rasterized=True)
        print(name, summary[stem], flush=True)

    axes[0, 0].axvline(injection_turns/f_ref*1e6, color='0.4', ls='--', lw=1)
    axes[0, 0].set(xlabel='Time [us]', ylabel='Bunching factor', ylim=(0, 1))
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(title='At the end of injection', xlabel='RF phase [deg]', ylabel='Current / ring-average current')
    axes[1, 0].set(title='500 turns after injection', xlabel='RF phase [deg]', ylabel='Current / ring-average current')
    axes[1, 1].set(title='Final longitudinal phase space', xlabel='RF phase [deg]', ylabel='Energy offset [MeV]')
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle('Offset longitudinal painting, no SC: main RF 200 kV/turn, second harmonic 0 or 160 kV\n'
                 '300 MeV reference; momentum offset -0.15%, rms spread 0.10%; 50% chopping')
    fig.supxlabel(f'1D thin-gap model; Bf uses {N_BINS} bins and periodic Gaussian smoothing '
                  f'(sigma = {SMOOTH_SIGMA_BINS:g} bins)', fontsize=9)
    fig.savefig(run_dir / 'bunching_factor_comparison.png', dpi=180)
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Results: {run_dir.resolve()}', flush=True)
    if SHOW_PLOTS:
        plt.show()


if __name__ == '__main__':
    main()
