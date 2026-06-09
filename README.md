# FFAG Program User Manual

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Command Line Usage](#2-command-line-usage)
- [3. Configuration File Description](#3-configuration-file-description)
  - [3.1 Magnetic Field Configuration File: config_Bmap.py](#31-magnetic-field-configuration-file-config_bmappy)
  - [3.2 Closed Orbit Calculation Configuration: config_SEO.py](#32-closed-orbit-calculation-configuration-config_seopy)
  - [3.3 Electric Field Configuration: config_Emap.py](#33-electric-field-configuration-config_emappy)
  - [3.4 Multiparticle Tracking](#34-multiparticle-tracking)
- [4. Output File Format and Plotting Instructions](#4-output-file-format-and-plotting-instructions)

## 1. Introduction

### 1.1 Overview

This program is used to simulate the three-dimensional motion of particles in accelerator electromagnetic field maps, including space charge effects. Its main functions include:

- Using a mid-plane magnetic field map as input and extrapolating it into a three-dimensional magnetic field through the source-free and curl-free properties of a static magnetic field combined with Taylor expansion.
- RF electric field acceleration.
- Two selectable particle motion solvers:
  - Fourth-order Runge-Kutta method: fourth-order single-step accuracy, but not symplectic.
  - Boris method: second-order single-step accuracy, symplectic, and slightly faster.
- A 2.5D PIC algorithm accelerated by FFT/iFFT for space charge calculation.
- Painting injection.

### 1.2 Dependencies

- Python >= 3.12
- OpenMP
- MPI
- NumPy (Python package)
- SciPy (Python package)
- Numba (Python package)
- mpi4py (Python package)
- matplotlib (Python package)
- pyfftw (Python package)

### 1.3 Source Code

- HPC

- CSNS GitLab platform

### 1.4 Update Log

- 2025.04.14
  - Initial submission.
  - Completed basic simulation functions: beam optics, multiparticle tracking, three-dimensional space charge effects, and painting injection.

- 2025.06.05
  - Added 2.5D space charge effects.
  - Performance optimization: preallocated and reused all temporary array variables inside functions to improve computational performance.
  - Performance optimization: replaced `scipy.fft` with FFTW.
  - Parallel method optimization: replaced pure MPI multiprocessing with a hybrid MPI multiprocessing + `numba.prange` multithreading mode. Within each node, `prange` multithreading is used without duplicating memory; between nodes, MPI multithreaded communication is used, and memory is copied once per node.

- Current work in 2025.06:
  - Adding a magnetic field error module.
  - Adding a phase stability region calculation module.
  - Optimizing the performance of high-order magnetic field interpolation.
  - Performing full-process tracking tests.

---

## 2. Command Line Usage

In the **project directory**, files named `main_*.py` are executable programs, and files named `config_*.py` are used to configure parameters. The generated `config_*.json` files are used as input files for the executable programs. The specific format of the `config_*.py` configuration files is described in the next section.

All command-line operations are listed below:

```bash
source ./SetEnvCmd.sh              # Load the virtual environment on the CSNS HPC server

python config_Bmap.py             # Generate the magnetic field configuration
python main_Bmap.py -j config_Bmap.json     # Generate the magnetic field map

python config_SEO.py              # Set closed orbit calculation parameters, such as which magnetic field to use and which energy points to calculate
(mpirun -np 6) python main_SEO.py -j config_SEO.json   # Calculate closed orbits; mpirun can be used for parallel execution, or the script can be run directly

python config_Emap.py             # Generate the electric field configuration
python main_Emap.py -j config_Emap.json     # Generate the electric field map

# For testing on a personal computer:
python config_track.py            # Generate the tracking configuration, including the initial small bunch, painting, SC, and electromagnetic field maps
python main_track.py -j config_track.json   # Multiparticle tracking
python main_MergeData.py output/simulation1/Bunch_Position/  # Merge multithreaded output files after the calculation is completed

# For running on the CSNS HPC:
python config_track.py            # Generate the tracking configuration, including the initial small bunch, painting, SC, and electromagnetic field maps
sbatch test.submit        # Submit the job on the CSNS HPC server

# The -r option can be used for restart from a checkpoint:
python .\main_track.py -j .\config_track.json -r .\time_2596.84_rank_0.npz
python .\main_track.py -j .\config_track.json -r .\time_2596.84_rank_0.csv
# Here, .\time_2596.84_rank_0.npz or .\time_2596.84_rank_0.csv is the Bunch file at the checkpoint time t = 2596.84 ns
```

## 3. Configuration File Description

### 3.1 Magnetic Field Configuration File: config_Bmap.py

The magnetic field map parameters are defined in `config_Bmap.py` and should include the following dictionary variables.

`machine` defines the machine parameters, including injection and extraction energies in MeV.

```python
# machine parameters
machine = dict()
machine['energy_inj'] = 300
machine['energy_ext'] = 600
```

`Bmap` contains all lattice parameters, including the lattice type, arrangement, and field strength:

```python
# Bmap configuration
Bmap = dict()
Bmap['Type'] = 'SCALE'
# Options: isochronous 'ISOCHRONOUS', or scaling 'SCALE'
Bmap['NSector'] = 12

Bmap['theta_step_rad'] = np.deg2rad(0.005)  # unit: rad
Bmap['rmin_max_step_m'] = (8.0, 9.3, 0.001)  # unit: m; slightly larger than the actual Bmap range to avoid exceeding the Bmap boundary during iteration

Bmap['interval_1'] = (7.5/30.0, 9.375/30.0, 2.8125/30.0, 2.8125/30.0, 7.5/30.0)  # unit: 1
Bmap['positive_or_negative_1'] = (0, 1.0, 0, -1.201/1.65, 0)  # unit: 1
Bmap['fringe_width_1'] = (0, 0.20, 0, 0.35, 0)  # unit: 1
Bmap['SpiralAngle_deg'] = 40.0  # unit: deg

Bmap['orbital_freq_MHz'] = 4.5  # unit: MHz; effective when Type is isochronous 'ISOCHRONOUS'
Bmap['k_value'] = 5.714  # unit: 1; effective when Type is scaling 'SCALE'
Bmap['B0_max_T'] = 1.65  # unit: T; B0 at the reference radius, effective when Type is scaling 'SCALE'
Bmap['R0_m'] = 9.0   # unit: m; reference radius, effective when Type is scaling 'SCALE'
```

| Parameter | Meaning and Description |
|---|---|
| **Type** | Defines the magnetic field map type. Options are isochronous `ISOCHRONOUS` or scaling `SCALE`. |
| **NSector** | Number of lattice periods, i.e., the number of magnetic field sectors. In this example, there are 12 periods. |
| **theta_step_rad** | Azimuthal step size of the cylindrical grid for the map, in rad. In this example, it is 0.005° converted to rad. |
| **rmin_max_step_m** | Radial range and step size of the cylindrical grid for the map, in m. In this example, the range is from 8.0 m to 9.3 m with a step size of 0.001 m. |
| **interval_1** | Division of one period into several segments, expressed as the fractional angular width of each segment. In this example, one cell is divided into five segments. The first segment occupies 7.5/30 of the cell angle, and the second segment occupies 9.375/30. |
| **positive_or_negative_1** | Magnetic pole polarity of each segment: `1` represents a positive pole, `0` represents a drift section, and a negative value represents a negative pole. The magnitude of the negative value gives the relative strength. In this example, the sequence is drift, positive pole, drift, negative pole with a field strength of 1.201/1.65 of the positive pole, and drift. |
| **fringe_width_1** | Fringe field width relative to the flat-top region of the magnet pole. For example, `0.5` means 50%. This parameter can be adjusted according to the plotted Bz-theta distribution to ensure that the field distributions between magnet poles do not interfere with each other. |
| **SpiralAngle_deg** | Spiral angle of the magnet pole, in degrees. |
| **orbital_freq_MHz** | Effective only for isochronous magnetic fields. It specifies the target orbital frequency, for example 4.5 MHz. |
| **k_value** | Effective only for scaling magnetic fields. This is the k value in the scaling field formula. |
| **B0_max_T** | Effective only for scaling magnetic fields. It is the magnetic field strength at the reference radius, i.e., B_0 in the formula. |
| **R0_m** | Effective only for scaling magnetic fields. It is the reference radius, i.e., r_0 in the formula. |

For a **scaling magnetic field**, the radial dependence of the magnetic field is

$$
B(r) = B_0 \left(\frac{r}{r_0}\right)^k
$$

where $B_0$ is the magnetic field strength at the reference radius $r_0$, and $k$ is the field index.

For an **isochronous magnetic field**, the magnetic field is expressed as a polynomial in $r$:

$$
B(r) = a_0 + a_1 (r - r_{\text{min}}) + a_2 (r - r_{\text{min}})^2 + a_3 (r - r_{\text{min}})^3 + \cdots
$$

where $r_{\text{min}}$ is the starting radius, and $a_0, a_1, a_2, \dots$ are polynomial coefficients. The user specifies the target frequency, and the program automatically searches for a set of coefficients that satisfies the requirement of a constant orbital frequency.

### 3.2 Closed Orbit Calculation Configuration: config_SEO.py

Before performing multiparticle simulations, closed orbits and optical parameters need to be calculated, because some closed orbit parameters are required as references during injection and tracking.

The closed orbit configuration file contains the following parameters:

```python
# some general parameters
config = dict()
config['start_Ek'] = 300.0
config['end_Ek'] = 600.0
config['delta_Ek'] = 30.0
config['extra_Ek'] = ()
config['Bmap_path'] = './Bmap'
```

| Parameter | Meaning and Description |
|---|---|
| **start_Ek** | Starting energy point, in MeV. In this example, it is 300 MeV. |
| **end_Ek** | Ending energy point, in MeV. In this example, it is 600 MeV. |
| **delta_Ek** | Energy spacing, in MeV. In this example, it is 30 MeV. |
| **extra_Ek** | Additional energy points. In this example, there are no additional energy points. Additional points can be added, for example `config['extra_Ek'] = (310,320,350,)`. |
| **Bmap_path** | Path of the magnetic field to be used. |

The calculated closed orbit data are saved in the `resultsSEO` folder under the **Bmap_path** directory.

### 3.3 Electric Field Configuration: config_Emap.py

### 3.4 Multiparticle Tracking Configuration

The multiparticle tracking parameters are defined in `config_track.py`. They include the injected small bunch information, magnetic field map and closed orbit information, electric field information, painting curve, and space charge configuration. The contents are as follows:

```python
# tracking parameters
track = dict()
stop_condition = {'max_stepsN': 50000000,  # Maximum number of tracking steps
                  'max_turn': 1000.5, #  Maximum number of turns
                  'max_time': float('inf')}  #  Maximum tracking time (second)

track['start_EkMeV'] = 300.0  # Initial kinetic energy (MeV)
track['stop_condition'] = stop_condition  # The simulation stops when any one of the three termination conditions is satisfied
track['time_step'] = 0.02e-9  # Fixed time step, 0.02 ns --> orbital period about 273 ns --> about 13650 steps per turn, about 1137 steps per cell
track['start_azimuth'] = 0.0  # Injection azimuth
track['solver_type'] = 1  # 0 ---> RK4, fourth-order single-step accuracy but not symplectic; 1 ---> Boris, second-order single-step accuracy but symplectic and slightly faster
```

| Parameter | Meaning and Description |
|---|---|
| **start_EkMeV** | Initial kinetic energy of the injected particles, in MeV. In this example, it is 300 MeV. |
| **stop_condition** | Simulation termination conditions, including the maximum number of steps, maximum number of turns, and maximum time. The simulation stops when any one of these conditions is satisfied. |
| **time_step** | Time step, in seconds. |
| **start_azimuth** | Injection azimuth, i.e., where injection starts. |
| **solver_type** | Type of ordinary differential equation solver. `0` means the fourth-order Runge-Kutta method, and `1` means the Boris pusher. |

The following parameters configure the space charge effect:

```python
# SC parameters
SC = dict()
SC['enable_SC'] = True  # True False: whether to include space charge effects
SC['SC_type'] = 1
# SC_type=0: Cartesian grid + FFT (3D, suitable for small bunches)
# SC_type=1: Cylindrical grid + FFT (2.5D, suitable for long bunches)
SC['grid_size_rfz'] = [256, 256, 128]  # Grid size; for Cartesian coordinates: x y z; for cylindrical coordinates: r z phi
```

| Parameter | Meaning and Description |
|---|---|
| **enable_SC** | Whether to include space charge effects. |
| **SC_type** | Space charge solver type. `0` means Cartesian grid + FFT, i.e., 3D and suitable for small bunches. `1` means cylindrical grid + FFT, i.e., 2.5D and suitable for long bunches. |
| **grid_size_rfz** | Number of grid cells in each direction. The order is x, y, z for Cartesian coordinates, and r, phi, z for cylindrical coordinates. |

The following parameters configure the magnetic field:

```python
# Magnetic field and high-order nonlinear term configuration
BmapAndSEO = dict()
BmapAndSEO['maps'] = './Bmap'  # Magnetic field path
BmapAndSEO['max_order'] = 1  # Expansion order; n means expansion up to order (2*n+1) [0 --> highest order 1, 1 --> highest order 3, 2 --> highest order 5]
```

| Parameter | Meaning and Description |
|---|---|
| **maps** | Path of the magnetic field map to be used. |
| **max_order** | Expansion order. `n` means the highest expansion order is (2*n+1), because Bz has only even-order terms, while Br and Bf have only odd-order terms. |

The following parameters configure the electric field:

```python
# Electric field configuration
Emap = dict()
Emap['enable'] = True    # True False
Emap['maps'] = './Emap/FrequencyCurve.txt'  # Electric field frequency curve; the file header contains parameters such as voltage, acceleration phase, and RF cavity position
```

| Parameter | Meaning and Description |
|---|---|
| **maps** | Path of the electric field map. The electric field uses a theoretical model. |
| **enable** | Whether to include the electric field. |

The following parameters configure the small bunch:

```python
# Injected small bunch parameters
BunchPara = dict()
BunchPara['ParticleDensity'] = 1.0e4  # Number of real particles represented by each macroparticle
BunchPara['ParticleNum'] = 100  # Number of macroparticles in each sub-bunch
BunchPara['TransverseREmit'] = 0.1  # Transverse emittance (pi*mm*mrad)
BunchPara['TransverseZEmit'] = 0.1  # Transverse emittance (pi*mm*mrad)
BunchPara['LongitudeT'] = 0.3  # Longitudinal length (ns)
BunchPara['LongitudeDEk'] = 0.004  # Longitudinal energy spread (MeV)
BunchPara['InjTimeNanoSec'] = 0  # Initial injection time (ns)
BunchPara['TransverseDistType'] = 'gauss'  # Transverse distribution type; options: 'gauss', 'kv', 'waterbag', 'hollow_waterbag'
BunchPara['LongitudeDistType'] = 'gauss'  # Longitudinal distribution type; options: 'gauss', 'kv', 'waterbag', 'hollow_waterbag'
```

| Parameter | Meaning and Description |
|---|---|
| **ParticleNum** | Number of macroparticles in each small bunch. |
| **ParticleDensity** | Number of real particles represented by each macroparticle. |
| **TransverseREmit** | Emittance in the R direction, i.e., the radial r-r' phase space, in π*mm*mrad. This is the 1-rms emittance. |
| **TransverseZEmit** | Emittance in the Z direction, i.e., the axial z-z' phase space, in π*mm*mrad. This is the 1-rms emittance. |
| **LongitudeT** | Longitudinal time length of the small bunch in the t-Ek phase space, in ns. |
| **LongitudeDEk** | Longitudinal energy spread of the small bunch in the t-Ek phase space, in MeV. |
| **InjTimeNanoSec** | Injection time of the first small bunch, usually set to 0 by default. |
| **TransverseDistType** | Distribution type in the small-bunch r-r' and z-z' phase spaces. Four options are available: 'gauss', 'kv', 'waterbag', and 'hollow_waterbag'. |
| **LongitudeDistType** | Distribution type in the small-bunch t-Ek phase space. Five options are available: 'match', 'gauss', 'kv', 'waterbag', and 'hollow_waterbag'. |

The following parameters configure the painting process:

```python
# Painting configuration
Paint = dict()
Paint['enable'] = True    # True False
Paint['MaxBunchNum'] = 100  # Number of injected sub-bunches
Paint['TimeInterval'] = 273.35  # Time interval in ns
Paint['Curve'] = './PaintCurves/Curve4.paint' # Painting curve; it contains 5 columns. The first column is time, and the remaining columns are offsets of the small bunch in phase space
```

| Parameter | Meaning and Description |
|---|---|
| **enable** | Whether to enable painting. |
| **MaxBunchNum** | Maximum number of small bunches to inject. |
| **TimeInterval** | Time interval between two successive injected small bunches. |
| **Curve** | Painting curve. It contains 5 columns: the first column is time, and the remaining columns are offsets of the small bunch in the r-r' and z-z' phase spaces. |

The following parameters configure dump detectors. First, a detector object is created and then added to the detector list in `DumpPara['modules']`. Multiple detector objects can be added to output multiple sets of information.

```python
# Dump configuration (detectors)
DumpPara = dict()

# Create a detector object:
PositionDumpBunch = {
    "type": "PositionDumpBunch",  # Save the whole bunch when particles pass a specified position; each bunch at each position is saved as a separate file
    # One-shot writing is used, which is faster
    "start_time": 273.35e-9 * (-1.0),
    "end_time": 273.35e-9 * 10.0,
    "dump_azimuth": [5.6, 35.6, 65.6, 95.6],  # Azimuths for saving, in degrees
    "save_folder": "./output/simulation1/Bunch_Position"  # Output path
}
# Add all detectors to the detector list
DumpPara['modules'] = [PositionDumpBunch, ]
```

The parameters of `PositionDumpBunch` are described below:

| Parameter | Meaning and Description |
|---|---|
| **type** | Detector type. In this example, it is `PositionDumpBunch`, meaning that the detector is placed at a given azimuth and outputs Bunch information when particles pass through that azimuth. |
| **start_time** | Start time of the detector. |
| **end_time** | End time of the detector. |
| **dump_azimuth** | Azimuths where the detector is placed. |
| **save_folder** | Directory where output files are saved. |

According to how particle information is saved during tracking, there are four detector types:

| Dump Type | File Output Method | Trigger Condition | Typical Application |
|---|---|---|---|
| **PositionDumpBunch** | One file per bunch | Recorded once when the particles pass through a specified azimuth | Records bunch information at a specified position, similar to PyORBIT. |
| **PositionDump** | One file per particle | Recorded once when the particle passes through a specified azimuth | Records the evolution of particles at specified positions. |
| **StepDumpBunch** | One file per bunch | Recorded at a fixed time interval | Records the time evolution of the bunch. The output files can also be used for restart from a checkpoint. |
| **StepDump** | One file per particle | Recorded at a fixed time interval | Records particle trajectories. The files are large and time-consuming to write. |

In addition to `PositionDumpBunch`, examples of the other three detector types are given below:

```python
# Other dump detectors
# StepDump:
StepDump = {
    "type": "StepDump",  # Save particles at fixed time intervals. Each particle is saved as a separate file and distinguished by ID; it can be used to plot single-particle trajectories
    # Append writing is used and frequent reading/writing is required; it is very slow and occupies a large amount of storage, so it should only be used for short simulations
    "start_time": 273.35e-9 * (-1.0), # Start time
    "end_time": 273.35e-9 * 60.0, # End time
    "interval_time": 50.0e-9,  # Time interval for saving
    "num_particles_to_dump_global": 50,  # Number of particles saved globally
    "save_folder": "./output/simulation1/Particle_time"  # Output path
}

# StepDumpBunch:
StepDumpBunch = {
    "type": "StepDumpBunch",  # Save the whole bunch at fixed time intervals. The bunch is saved as a separate file according to the step number
    # One-shot writing is used, which is faster
    "start_time": 273.35e-9 * (-1.0), # Start time
    "end_time": 273.35e-9 * 60.0, # End time
    "interval_time": 273.35e-9*10.5, # Time interval for saving the current bunch
    "save_folder": "./output/simulation1/Bunch_time"  # Output path
}

# PositionDump:
PositionDump = {
    "type": "PositionDump",  # Save particles when they pass a specified position. Each particle is saved as a separate file
    # Append writing is used. Reading/writing is slower than PositionDumpBunch but less frequent than StepDump
    "start_time": 273.35e-9 * (-1.0),
    "end_time": 273.35e-9 * 600.0,
    "dump_azimuth": (np.array([5.61, 35.61, 65.61, ])).tolist(),  # Azimuths for saving, in degrees
    "num_particles_to_dump_global": 100,  # Number of particles saved globally
    "save_folder": "./output/simulation1/Particle_Position"  # Output path
}
```

## 4. Output File Format and Plotting Instructions

During the simulation, the program outputs magnetic field files, electric field files, intermediate data, and particle information recorded by several detector modules. These outputs are described by module below.

### 4.1 Magnetic and Electric Field File Formats

Magnetic and electric field maps are usually saved in the `./Bmap/` and `./Emap/` directories, respectively. The corresponding configuration files are `config_Bmap.json` and `config_Emap.json`. The magnetic field is a three-dimensional extended map. The electric field is an ideal model based on the frequency curve; it generates a rectangular region at a given position, and a uniform electric field is applied inside the rectangular region.

* Magnetic field map:

  * The file format is `.npz`, which contains expansion coefficients of `Bz`, `Br`, and `Bf` on the cylindrical coordinate grid.

* Electric field map:

  * The file format is `.txt`. The header contains frequency, voltage, acceleration phase, and acceleration region position, i.e., the rectangular region. The data section contains the frequency curve.

---

### 4.2 Closed Orbit and Twiss Function Output File Formats

During the SEO search and Twiss function calculation process, the program generates a `resultsSEO/` folder under the specified Bmap directory. This folder contains the following analysis data files:

| File Name | Content |
|---|---|
| `SEO_ini.txt` | Characteristic parameters of closed orbits at different energies: Qr, Qz, frequency, and MeanR |
| `SEO_r.txt` | r(φ) of closed orbits at different energies |
| `SEO_pr.txt` | pr(φ) of closed orbits at different energies |
| `BetaFuncR.txt` | Radial β<sub>r</sub>(φ) curve |
| `BetaFuncZ.txt` | Axial β<sub>z</sub>(φ) curve |
| `AlphaFuncR.txt` | Radial α<sub>r</sub>(φ) curve |
| `AlphaFuncZ.txt` | Axial α<sub>z</sub>(φ) curve |
| `Bz.txt`, `Br.txt`, `Bf.txt` | Magnetic field components on the closed orbit |

File format description:

* All data are stored in ASCII text format, and the header indicates the units, such as Ek in MeV and angle in degrees.
* The first column is energy Ek, the first row is azimuth φ in rad, and the data region contains the physical quantities corresponding to different energy points and azimuths.

An example method for plotting closed orbits is given below. Other Twiss parameter files have a similar format:

```python
import numpy as np
import matplotlib.pyplot as plt

# Read r(φ) data and plot closed orbits
data = np.loadtxt('./Bmap/resultsSEO/SEO_r.txt')

# Row 0 is phi in rad; subsequent rows are r(phi) at the corresponding energy points
phi_rad = data[0, 1:]
phi_deg = np.rad2deg(phi_rad)
r_matrix = data[1:, 1:]  # shape: [n_energy, n_phi]
ek_list = data[1:, 0]    # Column 1 is Ek in MeV

# r vs fi
plt.figure(figsize=(8, 6))
for i, ek in enumerate(ek_list):
    plt.plot(phi_deg, r_matrix[i], label=f"Ek = {ek:.1f} MeV")
plt.xlabel('φ (deg)')
plt.ylabel('r (m)')
plt.title("Closed Orbit r(φ)")
plt.legend()
plt.grid(True)

# r*cos(fi) vs r*sin(fi)
plt.figure(figsize=(6, 6))
for i, ek in enumerate(ek_list):
    x = r_matrix[i] * np.cos(phi_rad)
    y = r_matrix[i] * np.sin(phi_rad)
    plt.plot(x, y, label=f"Ek = {ek:.1f} MeV")
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title("Closed Orbits in x-y Plane")
plt.legend()
plt.axis('equal')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

### 4.3 Detector Output Data Format

The output files of different detectors have exactly the same format. They are binary `.npz` files with a two-dimensional array of size `n×17`, where each row represents one particle state. However, the **trigger mechanism and data stacking method are different**:

* `StepDump` / `StepDumpBunch`: triggered at fixed time intervals. Data are stacked either for all particles at the same time or for the same particle over time steps.
* `PositionDump` / `PositionDumpBunch`: triggered by azimuth. Data are recorded when particles cross the detector azimuth. `PositionDumpBunch` is similar to PyORBIT and outputs bunch information at one position.

Unlike PyORBIT, which merges and prints data online, this program uses a FLUKA-like method:

* Each process independently outputs local `.npz` files.
* During post-processing, these files are merged into a readable ASCII `.csv` file.

The structure of the output `.npz` or `.csv` file is as follows:

| Index | Variable Name | Unit | Meaning |
|---|---|---|---|
| 0 | `r` | m | Radial position |
| 1 | `pr` | rad | Radial momentum slope = arctan(vr / vf) |
| 2 | `z` | m | Axial position |
| 3 | `pz` | rad | Axial momentum slope = arctan(vz / vf) |
| 4 | `fi` | rad | Azimuth |
| 5 | `Ek` | MeV | Kinetic energy |
| 6 | `inj_t` | s | Injection time |
| 7 | `Inj_flag` | - | Whether the particle has been injected: 1 means yes, 0 means no |
| 8 | `Survive` | - | Whether the particle survives: 1 means yes, 0 means lost |
| 9 | `RF_phase` | rad | RF phase of the electric field at the current time, using the sine convention |
| 10 | `Esc_r` | V/m | Space charge field Er |
| 11 | `Esc_z` | V/m | Space charge field Ez |
| 12 | `Esc_fi` | V/m | Space charge field Efi |
| 13 | `Bunch_ID` | - | Small bunch ID |
| 14 | `Local_ID` | - | ID within the current MPI process |
| 15 | `Global_ID` | - | Global ID |
| 16 | `t` | s | Current recording time |

#### Output File Naming Rule

```
prefix_start_{start_time}_end_{end_time}_rank_{rank_id}.npz
```

For example:

```
crossing_angle_5.6_start_0_end_1000_rank_1.npz
```

indicates that this file records particle data from rank 1 during the time interval from 0 to 1000 ns when particles pass through the azimuth of 5.6°.

```
crossing_angle_5.6_start_0_end_1000_rank_merged.csv
```

indicates that this file records particle data during the time interval from 0 to 1000 ns when particles pass through the azimuth of 5.6°, with data from all processes merged into one file.

Example output files:

```
output/simulation1/Bunch_Position/
├── crossing_angle_5.6_start_0_end_1000_rank_0.npz
├── crossing_angle_5.6_start_0_end_1000_rank_1.npz
└── ...
```

---

## 5. Data Merging and Visualization Tools

### 5.1 Merging Multithreaded npz Files into CSV

Each thread independently outputs a `.npz` file. If overall visualization and analysis are required, the script `main_MergeData.py` can be used to merge the files into `.csv` format:

```bash
python main_MergeData.py output/simulation1/Bunch_Position/
```

The output is saved in the `merged_files/` subfolder under the original directory, with the following naming rule:

```
crossing_angle_5.6_start_0_end_1000_merged.csv
```

The CSV file contains all particle attributes. The column order is the same as that defined in the npz file, and the last column is the recording time.

---

(20250605, unfinished)
