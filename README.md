# Barotropic Instability Simulation

Python-based simulation of barotropic instability in geophysical flows, developed as part of undergraduate research at [CIGEFI](https://www.cigefi.ucr.ac.cr/) (Center for Geophysical Research, University of Costa Rica), under the supervision of Dr. Tito Maldonado.

## Overview 

This project simulates the Intra-Americas Seas (IAS) region to investigate the interaction between the Caribbean Low-Level Jet (CLLJ) and tropical easterly waves (EWs) within the theoretical framework of barotropic instability, using [FluidSim](https://fluidsim.readthedocs.io/), running on the University of Costa Rica's institutional HPC cluster.

Barotropic instability is a fundamental mechanism in geophysical fluid dynamics by which small perturbations in a horizontally sheared flow grow and extract energy from the mean flow, playing a key role in the formation of eddies and large-scale turbulence in the atmosphere and ocean.


## What this project does

- Sets up and runs 2D barotropic flow simulations with configurable initial conditions (jet/shear profiles, perturbation amplitude, resolution).
- Processes simulation output to analyze growth rates and instability structure.
- Uses ERA5 reanalysis data and least-squares fitting to inform physically realistic parameter choices.
- Reproduces a real laboratory fluid dynamics case study for validation against experimental results.

## Features

- Pseudo-spectral solver
- FluidSim
- MPI parallelization
- ERA5 comparison
- Automatic diagnostics
- Reproducible simulations

## Data Sources

- ERA5 (Datos de reanalisis descargados para el nivel # hPa de las componentes u del jet)

## Scientific Background

### Physical Motivation


Understanding barotropic instability over the Intra-Americas Seas (IAS) is essential for explaining the dynamics of the Caribbean Low-Level Jet (CLLJ) and its interaction with Tropical Easterly Waves (EWs). During boreal summer, the CLLJ is one of the dominant atmospheric circulation features over the Caribbean, strongly influencing regional weather, moisture transport, and the propagation of synoptic-scale disturbances.

The zonal mean flow associated with the CLLJ can become barotropically unstable, allowing perturbations such as tropical easterly waves to extract kinetic energy from the background flow through horizontal shear. This energy transfer provides a mechanism for the maintenance and amplification of the waves as they propagate westward toward the eastern Pacific.

Rivera (2026) showed that the summertime CLLJ provides a barotropically unstable mean-flow environment for tropical easterly waves. In particular, the Rayleigh–Kuo necessary condition for barotropic instability,

$$
\frac{\partial \overline{q}}{\partial y} = 0
\quad \text{or equivalently} \quad
\frac{\partial \overline{q}}{\partial y}
\text{ changes sign},
$$

is satisfied over much of the Caribbean basin, indicating regions where growing disturbances can extract energy from the mean flow.

The objective of this project is to investigate these nonlinear jet–wave interactions using an idealized numerical model, providing a controlled framework for studying the associated energy transfer mechanisms.




### Governing Equations

The simulations solve the two-dimensional barotropic vorticity equation on a $\beta$-plane,

$$
\frac{\partial \zeta}{\partial t} + J(\psi,\zeta+f) = \nu_4\nabla^4\zeta + F
$$

where $\zeta=\nabla^2\psi$ is the relative vorticity, $f=f_0+\beta y$ is the Coriolis parameter, $\nu_4$ is the hyperviscosity coefficient, and $F$ represents the prescribed vorticity forcing that continuously injects easterly-wave energy into the domain.

To diagnose barotropic instability, two key quantities are evaluated. The first is the **Rayleigh–Kuo criterion**,

$$
\beta-\frac{\partial^2 U}{\partial y^2},
$$

whose sign reversal indicates a necessary condition for barotropic instability. The second is the **barotropic kinetic energy conversion**,

$$
CK=-\left\langle u'v' \right\rangle\frac{\partial U}{\partial y},
$$

which measures the transfer of kinetic energy between the mean jet and the perturbations. Positive values of $CK$ indicate that the perturbations extract energy from the mean flow, allowing tropical easterly waves to grow or be maintained.

### Model Assumptions

- Incompressible fluid
- Periodic computational domain
- Two-dimensional approximation
- Pseudo-spectral formulation


### External Forcing - Easterly Waves

The wave is taken as the perturbation and it is represented as a Gaussian envelope in latitude modulated by a sinusoidal carrier in longitude that propagates westward:

$$u' = A \sin(k \lambda + 2\pi t/T) e^{- \frac{(\varphi - \varphi_0)^2}{2 \sigma_y^2}}$$

$$v' = 0.6 A \cos(k \lambda + 2\pi t/T) e^{- \frac{(\varphi - \varphi_0)^2}{2 \sigma_y^2}}$$


The parameters are defined as follows:

- $k$ — wave number ($\mathrm{m^{-1}}$).
- $\lambda$ — wavelength ($\mathrm{m}$).
- $t$ — time ($\mathrm{s}$).
- $T$ — wave period ($\mathrm{s}$).
- $A$ — velocity perturbation amplitude, which defines the initial strength of the prescribed tropical waves through the wind anomalies $u'$ and $v'$.

These perturbations typically have amplitudes of approximately $\pm 5\,\mathrm{m\,s^{-1}}$.

The perturbation amplitude directly influences the **barotropic kinetic energy conversion** through the eddy momentum flux,

$$
\left\langle u'v' \right\rangle
$$

which appears in the energy conversion term,

$$
CK=-\left\langle u'v' \right \rangle \frac{\partial U}{\partial y}
$$

Larger perturbation amplitudes generally produce stronger momentum fluxes, increasing the rate at which tropical easterly waves can extract kinetic energy from the mean Caribbean Low-Level Jet.
   

### Mean Jet Profile

The Caribbean Low-Level Jet (CLLJ) is modeled as a superposition of Gaussian functions, producing a zonally uniform and climatologically realistic mean flow:

$$
\overline{u}(\varphi)
=
\sum_i
A_i
\exp\left(
-\frac{(\varphi-\varphi_i)^2}
{2\sigma_i^2}
\right).
$$

The sign convention is such that $\overline{u}<0$ represents easterly winds. Since the mean flow is assumed to be zonally uniform, it depends only on latitude and is independent of longitude.

The parameters are defined as follows:

- $A_i$ — velocity amplitude of the $i$-th Gaussian component ($\mathrm{m\,s^{-1}}$).
- $\varphi$ — latitude.
- $\varphi_i$ — latitude at which the Gaussian is centered.
- $\sigma_i$ — Gaussian width, which controls the meridional extent of the jet component.


## Repository Structure

```text
CLLJ-Simulations/
│
├── README.md                 # Project documentation
├── LICENSE
│
├── simulation/               # Main numerical model
│   ├── CLLJ_simulation.py
│
├── era5/                     # ERA5 download and preprocessing
│   ├── data.py
│   ├── era5_cllj_925.nc 
├── analysis/                 # Diagnostics and post-processing
│   ├── post-processing-data.py
│
├── cluster/                  # HPC job scripts
│   ├── run_64cores.sh
│   ├── run_debug.sh
│   └── README.md
├── output/                   # Results 
```
## Tech Stack

- Python
- FluidSim
- FluidFFT
- mpi4py
- SLURM

## Installation

### Clone the repository

```bash
git clone https://github.com/3stiven366/Barotropic_Instability.git
cd CLLJ-Simulations
```

### Enviroment Configuration
To set up the FluidSim environment, I strongly recommend using Pixi.

#### Installating FluidSim

```bash
cd where/you/want/to/have/the/pixi/env/directory

uvx install-locked-env https://github.com/fluiddyn/fluidsim/tree/branch/default/pixi-envs/env-fluidsim
```



#### Installing FluidSim-MPI

```bash
cd where/you/want/to/have/the/pixi/env/directory

uvx install-locked-env https://github.com/fluiddyn/fluidsim/tree/branch/default/pixi-envs/env-fluidsim-mpi
```

#### To activate

```bash
cd where/you/installed/the/pixi/env/directory
eval "$(pixi shell-hook)"
```

You can always check [FluidSim Documentation](https://fluidsim.readthedocs.io/).

### Requirements
```bash
pip install numpy
```

## Running Simulations
You can run the simulation with or without MPI, if you installed FluidSim with pixi, make sure to be in the right enviroment. 

### With MPI
```bash
mpirun -n 4 python CLLJ_simulation.py
```
### Without MPI
```bash
python CLLJ_simulation.py
```
## Running on an HPC Cluster

This project was developed and tested on the UCR HPC cluster using
SLURM and an MPI-enabled FluidSim installation.

Example SLURM submission scripts are provided in the
[`cluster/`](cluster/) directory for different execution scenarios,
including:

- Debug jobs
- MPI jobs


These scripts illustrate how to submit FluidSim simulations using the
SLURM workload manager and can be adapted to other HPC clusters with
similar configurations.

To submit a job:

```bash
sbatch run_64cores.sh
```




## References
1. Rivera, E. R., Amador, J. A., et al. (2026). *Interaction between the Caribbean Low-Level Jet and Tropical Easterly Waves during OTREC*. **Meteorology**, 5(1), 6. https://doi.org/10.3390/meteorology5010006
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
## Author

Estiven Hernández Alfaro — Physics student, Universidad de Costa Rica
