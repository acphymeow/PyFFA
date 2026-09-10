# PyFFA

PyFFA is a Python research code for fixed-field alternating-gradient (FFAG) accelerator simulation. It provides analytical and tabulated magnetic-field models, closed-orbit and optics calculations, RF acceleration, multiparticle tracking, painting injection, and three-dimensional or slice-by-slice space-charge solvers.

## Contents

- [Features](#features)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Environment setup](#environment-setup)
- [Quick start](#quick-start)
- [Magnetic-field and closed-orbit workflows](#magnetic-field-and-closed-orbit-workflows)
- [RF electric-field workflow](#rf-electric-field-workflow)
- [Multiparticle tracking](#multiparticle-tracking)
- [Outputs](#outputs)
- [Research projects](#research-projects)
- [Benchmarks and reproducibility](#benchmarks-and-reproducibility)
- [Validation](#validation)
- [Data and repository policy](#data-and-repository-policy)

## Features

- Analytical scaling or isochronous FFAG fields generated from a mid-plane model and extended in three dimensions with high-order Taylor maps.
- Direct interpolation of single-cell three-dimensional OPERA field tables.
- Static equilibrium orbit (SEO), tune, and Twiss-function calculations.
- Fixed or spiral RF gaps, multiple gaps per turn, and programmable voltage, phase, and frequency curves.
- Fourth-order Runge-Kutta and second-order symplectic Boris tracking.
- Three-dimensional Cartesian PIC/FFT and 2.5D slice-by-slice PIC/FFT space-charge models.
- Painting injection, apertures, field harmonics, local extraction kickers, and configurable diagnostics.
- Dedicated reverse-injection and kicker/septum study packages.
- Public benchmark and paper-reproduction source code without large simulation datasets.

## Repository layout

```text
PyFFA/
├── src/                         # Core FFAG modules
├── examples/                    # main_*.py entry points and config_*.py templates
├── benchmarks/                  # Numerical benchmarks and validation scripts
│   └── reproducibility/scripts/ # Public reproducibility source tree
├── tools/                       # General analysis, conversion, and plotting tools
├── project_reverse_injection/   # Reverse-injection studies and documentation
├── project_bump/                # Kicker, bump, and septum studies
├── PaintCurvesNew/              # Generated painting-curve examples
├── SetEnvCmd.ps1                # Windows PowerShell environment helper
└── SetEnvCmd.sh                 # Linux/HPC environment helper
```

Magnetic maps, electric maps, HPC working copies, generated JSON files, simulation outputs, and local research archives are intentionally excluded from Git. Typical local data directories are `Bmaps/`, `Emaps/`, and `output/`.

All production commands should be launched from the repository root. Relative paths such as `./Bmaps`, `./Emaps`, and `./output` are resolved from the current working directory.

## Requirements

- Python 3.12 or newer
- OpenMP
- MPI implementation
- NumPy
- SciPy
- Numba
- mpi4py
- Matplotlib
- pyFFTW

Some plotting and presentation utilities have additional optional dependencies. Individual scripts report missing optional packages when required.

## Environment setup

The environment helpers add `src/` and `examples/` to `PYTHONPATH` for the current shell only. This keeps the source tree importable without installing PyFFA as a package and allows tools to reuse helpers such as `main_track.load_bmap_auto`.

Windows PowerShell:

```powershell
Set-Location C:\path\to\PyFFA
.\SetEnvCmd.ps1
```

Linux or HPC shell:

```bash
cd /path/to/PyFFA
source ./SetEnvCmd.sh
```

The shell script must be sourced; executing it in a child shell will not retain the updated `PYTHONPATH`.

## Quick start

The standard workflow is:

```text
config_Bmap.py -> config_Bmap.json -> main_Bmap.py
                                      |
config_SEO.py  -> config_SEO.json  -> main_SEO.py
                                      |
config_Emap.py -> config_Emap.json -> main_Emap.py
                                      |
config_track.py -> config_track.json -> main_track.py
```

From the repository root:

```bash
# 1. Generate an analytical magnetic-field configuration and map.
python examples/config_Bmap.py
python examples/main_Bmap.py -j config_Bmap.json

# 2. Search closed orbits and calculate optical functions.
python examples/config_SEO.py
python examples/main_SEO.py -j config_SEO.json

# MPI can be used for the SEO energy scan.
mpirun -np 6 python examples/main_SEO.py -j config_SEO.json

# 3. Generate the RF frequency curve and optional spiral-gap maps.
python examples/config_Emap.py
python examples/main_Emap.py -j config_Emap.json

# 4. Generate the tracking configuration and run tracking.
python examples/config_track.py
python examples/main_track.py -j config_track.json

# 5. Merge distributed detector output when required.
python examples/main_MergeData.py output/simulation1/Bunch_Position/
```

The checked-in configuration scripts are research templates, not universal zero-configuration examples. Before a production run, review their energy range, field-map paths, SEO result directory, RF curve, aperture, dump destinations, MPI settings, and particle count. A fresh clone does not contain the large field maps needed by the current FD16 examples.

Checkpoint restart accepts either NPZ or CSV particle data:

```bash
python examples/main_track.py -j config_track.json -r time_2596.84_rank_0.npz
python examples/main_track.py -j config_track.json -r time_2596.84_rank_0.csv
```

## Magnetic-field and closed-orbit workflows

### Analytical fields

`examples/config_Bmap.py` defines the machine energy range and the analytical lattice. Important entries include:

| Key | Meaning |
|---|---|
| `config.folder` | Output directory for the generated magnetic map |
| `config.expand_order` | Requested Taylor-map expansion order |
| `machine.energy_inj`, `machine.energy_ext` | Injection and extraction kinetic energies in MeV |
| `Bmap.Type` | `SCALE` or `ISOCHRONOUS` |
| `Bmap.NSector` | Number of lattice sectors |
| `Bmap.theta_step_rad` | Azimuthal grid step in radians |
| `Bmap.rmin_max_step_m` | Radial range and step in metres |
| `Bmap.interval_1` | Relative cell-element lengths |
| `Bmap.positive_or_negative_1` | Relative element polarities and strengths |
| `Bmap.fringe_width_1` | Fringe-field parameters |
| `Bmap.SpiralAngle_deg` | Spiral angle in degrees |
| `Bmap.k_value` | Scaling-field index |
| `Bmap.B0_max_T`, `Bmap.R0_m` | Reference field and radius |

For a scaling field,

$$
B(r) = B_0 \left(\frac{r}{r_0}\right)^k.
$$

For an isochronous field, the radial field is represented by a polynomial whose coefficients are determined from the requested orbital-frequency behaviour.

### OPERA 3D fields

`examples/main_SEO.py` and `examples/main_track.py` automatically distinguish the two supported map layouts:

- A directory containing `config_Bmap.json` is loaded as an analytical/Taylor map.
- Otherwise, a directory containing one matching OPERA metadata JSON and NPZ dataset is loaded as an OPERA 3D map.

OPERA queries use the global azimuth. The field class folds it into the real single-cell table using the table's `phi_min` and `phi_max` metadata. Analytical fields continue to use the cell-local azimuth.

See [OPERA 3D field-data guide](OPERA_3D_FIELD_DATA_GUIDE_CN_GENERIC.md) for the current OPERA conversion and metadata notes.

### SEO and Twiss calculations

`examples/config_SEO.py` controls the energy scan and field-error harmonics:

```python
config['start_Ek'] = 300.0
config['end_Ek'] = 600.0
config['delta_Ek'] = 50.0
config['extra_Ek'] = ()
config['Bmap_path'] = './Bmaps/OperaFieldMap'
config['SEO_folder'] = 'resultsSEO'
```

The Bmap path may point to either an analytical map directory or an OPERA 3D dataset directory. Results are written below `<Bmap_path>/<SEO_folder>/`.

Typical SEO outputs are:

| File | Content |
|---|---|
| `SEO_ini.txt` | Closed-orbit summary, including tunes, frequency, and mean radius |
| `SEO_r.txt` | Closed-orbit radius as a function of azimuth |
| `SEO_pr.txt` | Normalized radial momentum as a function of azimuth |
| `BetaFuncR.txt`, `BetaFuncZ.txt` | Radial and vertical beta functions |
| `AlphaFuncR.txt`, `AlphaFuncZ.txt` | Radial and vertical alpha functions |
| `Bz.txt`, `Br.txt`, `Bf.txt` | Field components sampled on the closed orbit |

The current implementation saves full-turn SEO and Twiss curves. Strictly cell-periodic fields may use a one-cell search before the orbit is expanded to a full turn. With enabled field harmonics, final closed-orbit refinement is performed over a full turn.

## RF electric-field workflow

`examples/config_Emap.py` can generate an RF frequency curve and an optional spiral-gap field map. It supports:

- independent enable flags for the frequency curve and spiral map;
- voltage and synchronous-phase ramps specified by turn number;
- multiple RF gaps and harmonic number `h`;
- fixed rectangular gaps or spiral gaps derived from SEO intersections;
- Gaussian transverse gap profiles;
- automatic per-gap phase offsets;
- separate calculation and saved radial ranges.

The primary configuration groups are:

| Key | Meaning |
|---|---|
| `config.filename` | Frequency-curve output path |
| `Emap.Bmap_fold` | Magnetic-map directory containing SEO data |
| `Emap.acc_phase_start/end` | Synchronous phase program |
| `Emap.acc_voltage_start/end` | Voltage program |
| `Emap.phase_ramp_start/end` | Phase-ramp turn interval |
| `Emap.voltage_ramp_start/end` | Voltage-ramp turn interval |
| `Emap.harmonic` | RF harmonic number |
| `Emap.Ngap` | Number of accelerating gaps |
| `Emap.gap_model` | `fixed` or `spiral` |
| `Emap.emap_save_dir` | Spiral-map output directory |

For multiple gaps, the code applies a common synchronous-phase program and an automatically generated fixed offset

$$
\Delta\phi_i = \frac{2\pi h i}{N_{\mathrm{gap}}},
\qquad i=0,\ldots,N_{\mathrm{gap}}-1.
$$

The tracker dispatches the RF model from the map header: `fixed` uses `FFAG_EField_new`, while `spiral` uses `FFAG_EField_spiral`.

## Multiparticle tracking

`examples/config_track.py` writes `config_track.json`. The main configuration blocks are:

| Block | Purpose |
|---|---|
| `config` | MPI process count and Numba thread count |
| `track` | Initial energy, start time, time step, solver, and stop conditions |
| `SC` | Space-charge switch, solver type, grid size, and update interval |
| `BmapAndSEO` | Magnetic-map path, SEO directory, and Taylor order |
| `BumpKicker` | Optional global-azimuth local kicker fields |
| `Emap` | RF map path and enable switch |
| `BunchPara` | Macroparticle count, intensity, emittance, and distributions |
| `Paint` | Painting curve and injection timing |
| `Aperture` | Radial and vertical acceptance limits |
| `DumpPara` | Position- or time-triggered diagnostics |

### Tracking solvers

- `solver_type = 0`: fourth-order Runge-Kutta, fourth-order single-step accuracy, not symplectic.
- `solver_type = 1`: Boris pusher, second-order single-step accuracy and symplectic.

The simulation stops when any configured limit in `max_stepsN`, `max_turn`, or `max_time` is reached. Tracking starts from `track.time_start`.

### Space charge

- `SC_type = 0`: three-dimensional Cartesian PIC/FFT, intended for compact bunches.
- `SC_type = 1`: 2.5D slice-by-slice PIC/FFT with energy-dependent closed-orbit projection, intended for long curved bunches.
- `SC_type = 2`: Gaussian toy model used for controlled comparisons.

For `SC_type = 1`, the production chain uses `Build_2_5D_SliceGrid_sbs_new_fixProjection`. It computes transverse and longitudinal space-charge components and writes them to particle columns 10, 11, and 12 as `Esc_r`, `Esc_z`, and `Esc_fi`. The FFTW plan is reused, while the Green-function kernel is rebuilt for the current transverse grid spacing at each space-charge update.

`SC.step` controls how many tracking steps elapse between field recomputations. The field is frozen during an individual RK4 or Boris step.

### Kicker model

`BumpKicker` defines optional hard-edge, static local fields in global azimuth. The default vertical component is

$$
B_z = \mathrm{sign}\,\frac{BL}{L_{\mathrm{eff}}}.
$$

The region is limited by `phi_start_deg`, `phi_span_deg`, `rmin_m`, and `rmax_m`. Constant `Br_T` and `Bfi_T` components may also be supplied. The current model has no time waveform and no vertical cutoff.

### Dump modules

Four detector types are available:

| Type | Trigger | File organisation | Typical use |
|---|---|---|---|
| `PositionDumpBunch` | Crossing selected azimuths | One file per bunch snapshot | Bunch diagnostics at fixed locations |
| `PositionDump` | Crossing selected azimuths | One file per particle | Turn-by-turn particle motion |
| `StepDumpBunch` | Fixed time interval | One file per bunch snapshot | Bunch evolution and restart data |
| `StepDump` | Fixed time interval | One file per particle | Short single-particle trajectory studies |

Each MPI rank writes its own NPZ files. Use `examples/main_MergeData.py` or `examples/main_MergeDataBatch.py` to create merged CSV files for analysis.

## Outputs

Particle dump arrays use 17 columns:

| Index | Name | Unit | Description |
|---:|---|---|---|
| 0 | `r` | m | Radial position |
| 1 | `pr` | rad | Radial trajectory angle or normalized radial slope |
| 2 | `z` | m | Vertical position |
| 3 | `pz` | rad | Vertical trajectory angle or slope |
| 4 | `fi` | rad | Global azimuth |
| 5 | `Ek` | MeV | Kinetic energy |
| 6 | `inj_t` | s | Injection time |
| 7 | `Inj_flag` | - | Injection-state flag |
| 8 | `Survive` | - | Survival flag |
| 9 | `RF_phase` | rad | RF phase using the sine convention |
| 10 | `Esc_r` | V/m | Radial space-charge field |
| 11 | `Esc_z` | V/m | Vertical space-charge field |
| 12 | `Esc_fi` | V/m | Azimuthal space-charge field |
| 13 | `Bunch_ID` | - | Injected sub-bunch identifier |
| 14 | `Local_ID` | - | Rank-local particle identifier |
| 15 | `Global_ID` | - | Global particle identifier |
| 16 | `t` | s | Recording time |

Detector filenames encode the trigger position or interval, time bounds, and MPI rank. Merged CSV output is saved in a `merged_files/` subdirectory by the merge utilities.

## Research projects

### Reverse injection

[`project_reverse_injection/`](project_reverse_injection/) contains the 300 MeV stripping-injection reverse-tracking tools, analytical-field parameter scans, OPERA 3D reverse tracking, and dynamic-aperture studies. Start with:

- [Reverse-injection guide](project_reverse_injection/README.md)
- [Dynamic-aperture guide](project_reverse_injection/DA_README.md)

These scripts expect local Bmap and result datasets and should be run from the repository root.

### Kicker, bump, and septum studies

[`project_bump/`](project_bump/) contains production-Boris single-particle and bunch tracking for extraction kickers, septum placement, kicker-split comparisons, and two- or three-straight layouts. Start with the [bump-project guide](project_bump/README.md).

The current scripts include the four-kicker `2+2`, `1+3`, `3+1`, and `0+4` scans; the three-straight `2+1+1`, `1+2+1`, and `1+1+2` scans; and the six-kicker `2+2+2` comparison. Generated scan directories are intentionally excluded from Git.

## Benchmarks and reproducibility

General numerical benchmarks are stored in [`benchmarks/`](benchmarks/). The self-contained public source snapshot for longitudinal-space-charge and related paper checks is located at [`benchmarks/reproducibility/scripts/`](benchmarks/reproducibility/scripts/).

The reproducibility tree contains Python and Markdown source only. Generated PNG, PDF, CSV, NPZ, JSON, and text results are excluded. Several 3D comparisons require local HPC output and SEO tables that are not part of the repository; the individual README files identify those inputs.

Example offline validation:

```bash
python benchmarks/reproducibility/scripts/02_harmonic_continuity/test_core_bin_enclosing_arc.py --no-show
```

## Validation

After changing core tracking or space-charge code, run at least:

```bash
python -m compileall -q src examples benchmarks tools project_reverse_injection/scripts project_bump/scripts
python -c "import FFAG_Bunch, FFAG_SC_new, FFAG_track; print('imports ok')"
python examples/main_track.py --help
```

The environment helper must be active for the import and entry-point checks.

Useful targeted checks include:

```bash
python tools/tool_smoke_sc_fixProjection.py
python tools/tool_check_emap_voltage.py
python tools/tool_test_Opera3D_field.py --field-dir Bmaps/OperaFieldMap
python project_bump/scripts/tool_compare_kicker_three_straights.py --help
```

The targeted field and tracking checks require their corresponding local datasets.

## Data and repository policy

- Do not commit generated configuration JSON, magnetic/electric maps, HPC copies, caches, or simulation output.
- Keep production modules in `src/`, executable workflows in `examples/`, reusable utilities in `tools/`, and project-specific scripts below their `project_*` directory.
- Run workflows from the repository root unless a script explicitly documents another working directory.
- Avoid force-pushing or rewriting published history.
- Preserve the particle space-charge interface: columns 10, 11, and 12 are `Esc_r`, `Esc_z`, and `Esc_fi`.

PyFFA is active research software. Numerical settings and example configurations should be reviewed for each lattice, beam, and computing environment before results are used in an accelerator design study.
