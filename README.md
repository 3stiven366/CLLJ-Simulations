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

- ERA5 reanalysis data of the 925 hPa zonal wind component (u), used to construct and calibrate the climatological Caribbean Low-Level Jet (CLLJ) profile.
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




### Easterly Wave Perturbations

The wave is taken as the perturbation and is derived from a streamfunction with a Gaussian envelope in latitude modulated by a sinusoidal carrier in longitude that propagates westward:

$$\psi' = A(y) \sin(k_x x + \omega t + \phi_0), \qquad A(y) = A_0\, \sigma_y \sqrt{e}\; e^{-\frac{(y - y_0)^2}{2\sigma_y^2}}$$

The velocity components and the perturbation vorticity follow by direct differentiation of $\psi'$, which guarantees a non-divergent flow and dynamical consistency between $u'$, $v'$ and $\zeta'$:

$$u' = \frac{\partial \psi'}{\partial y} = A'(y) \sin(k_x x + \omega t + \phi_0)$$

$$v' = -\frac{\partial \psi'}{\partial x} = -k_x A(y) \cos(k_x x + \omega t + \phi_0)$$

$$\zeta' = \nabla^2 \psi' = \left[A''(y) - k_x^2 A(y)\right] \sin(k_x x + \omega t + \phi_0)$$

with the analytical derivatives of the Gaussian envelope:

$$A'(y) = -\frac{(y - y_0)}{\sigma_y^2} A(y), \qquad A''(y) = \left[\frac{(y-y_0)^2}{\sigma_y^4} - \frac{1}{\sigma_y^2}\right] A(y)$$

The parameters are defined as follows:

- $k$ — zonal wavenumber (dimensionless): the number of complete wave cycles fitting in the zonal domain.
- $k_x = 2\pi k / L_x$ — physical zonal wavenumber ($\mathrm{rad\,m^{-1}}$).
- $x$ — zonal coordinate ($\mathrm{m}$).
- $y$ — meridional coordinate ($\mathrm{m}$), with $y_0$ the latitude of the envelope centre.
- $\sigma_y$ — meridional envelope width ($\mathrm{m}$).
- $t$ — time ($\mathrm{s}$).
- $T$ — wave period ($\mathrm{s}$), with $\omega = 2\pi/T$.
- $\phi_0$ — phase offset ($\mathrm{rad}$).
- $A_0$ — velocity perturbation amplitude ($\mathrm{m\,s^{-1}}$), which defines the initial strength of the prescribed tropical waves through the wind anomalies $u'$ and $v'$.

Since $\psi'$ has units of $\mathrm{m^2\,s^{-1}}$, the envelope prefactor must carry those units as well: multiplying the velocity scale $A_0$ by the length scale $\sigma_y$ fixes this. The additional factor $\sqrt{e}$ is a normalization choice made so that $A_0$ equals exactly the peak zonal perturbation amplitude:

$$\max|u'| = A_0 \quad \text{at } y = y_0 \pm \sigma_y, \qquad \max|v'| = k_x \sigma_y \sqrt{e}\, A_0 \quad \text{at } y = y_0$$

Note that $u'$ is the derivative of a Gaussian and therefore has a dipolar structure in latitude — it vanishes at $y_0$ and peaks at $y_0 \pm \sigma_y$ — while $v'$ peaks at the envelope centre. The ratio between the two amplitudes is set by $k_x \sigma_y \sqrt{e}$, i.e. by the spatial scale of the wave, rather than by a prescribed constant.

These perturbations typically have amplitudes of approximately $\pm 5\,\mathrm{m\,s^{-1}}$.

The perturbation amplitude directly influences the **barotropic kinetic energy conversion** through the eddy momentum flux,

$$
\left\langle u'v' \right\rangle
$$

which appears in the energy conversion term,

$$
CK=-\left\langle u'v' \right \rangle \frac{\partial U}{\partial y}
$$

Larger perturbation amplitudes generally produce stronger momentum fluxes, increasing the rate at which tropical easterly waves can extract kinetic energy from the mean Caribbean Low-Level Jet. The quadrature phase relationship between $u'$ and $v'$ that sustains $\langle u'v'\rangle \neq 0$ now emerges from the streamfunction formulation itself rather than being imposed externally.

### Vorticity Forcing Implementation

The perturbations described above define a wave structure, but they do not by
themselves define how that structure is *injected* into the flow. This section
makes that distinction explicit, since it is the source of a subtle but critical
dimensional issue.

#### The forcing term is a rate, not a field

The governing equation solved by the model is

$$
\frac{\partial \zeta}{\partial t} + J(\psi,\zeta+f) = \nu_4\nabla^4\zeta + F
$$

Every term in this equation carries units of $\mathrm{s^{-2}}$. In particular
$\partial\zeta/\partial t$ has units of $[\zeta]/[t] = \mathrm{s^{-1}}/\mathrm{s}
= \mathrm{s^{-2}}$, and therefore $F$ must also be $\mathrm{s^{-2}}$ — otherwise
it could not be added to the other terms.

This is a property of the vorticity equation itself, not of any particular
solver. $F$ is a **source term in an evolution equation**: it is the *rate* at
which vorticity is injected, not a vorticity field.

The wave vorticity computed analytically from the streamfunction,

$$
\zeta' = \nabla^2\psi' = \left[A''(y) - k_x^2 A(y)\right]\sin(k_x x + \omega t + \phi_0),
$$

has units of $\mathrm{s^{-1}}$, as expected for a vorticity. Passing $\zeta'$
directly as $F$ would be dimensionally inconsistent, and numerically it injects
vorticity orders of magnitude larger than the mean flow within a few time steps,
causing the adaptive time step to collapse.

#### Normalisation by the wave period

The forcing is therefore defined as the wave vorticity divided by a
characteristic time scale:

$$
F = \frac{\zeta'}{T}, \qquad [F] = \frac{\mathrm{s^{-1}}}{\mathrm{s}} = \mathrm{s^{-2}}
$$

The interpretation is direct: sustained over a time $T$, the forcing accumulates
a vorticity of magnitude $\zeta'$, i.e. it rebuilds one full wave structure in
one characteristic time. For multiple superposed waves, each mode is normalised
by its own period $T_i$ rather than by a single global constant, so that each
wave is replenished on its own dynamical time scale:

$$
F = \sum_i \frac{\zeta'_i}{T_i}
$$

This removes any free tuning parameter from the forcing: $T_i$ is already
prescribed by the wave period $T_{\mathrm{days},i}$, which is set from the
observed 2.5–6 day easterly-wave band.

Physically, the forcing continuously replenishes the prescribed wave structure
while the mean jet deforms it and exchanges energy with it through the
barotropic conversion term $CK$. The waves are therefore maintained rather than
allowed to decay, and the diagnosed $CK$ reflects the jet–wave interaction under
a statistically steady wave supply.

#### FluidSim implementation notes

FluidSim offers two relevant in-script forcing modes, and the distinction matters:

| Mode | Grid | Normalisation |
|------|------|---------------|
| `in_script_coarse` | Reduced grid, upscaled internally | Renormalised internally by FluidSim |
| `in_script` | Full simulation grid | None — the returned array is used as-is |

With `in_script_coarse`, FluidSim builds a coarse spectral grid (whose size it
determines internally from `nkmax_forcing`) and rescales the forcing to maintain
a prescribed injection rate. The returned amplitude is therefore not under the
user's control, and for a spatially structured forcing such as a prescribed wave
the coarse-to-fine projection introduces large amplification factors.

With `in_script`, the array returned by `compute_forcing_each_time` is
transformed by `oper.fft()` (which is normalised, so a physical-space value of
magnitude $a$ maps to a spectral coefficient of the same magnitude) and added
directly to the nonlinear tendencies in `tendencies_nonlin`:

```python
if self.params.forcing.enable:
    tendencies_fft += self.forcing.get_forcing()
```

`tendencies_fft` is $\partial\zeta/\partial t$. This is the concrete reason the
returned array must be $\mathrm{s^{-2}}$: it is summed with the Jacobian and
dissipation terms and then multiplied by $\Delta t$ inside the RK4 integrator.

For this reason the model uses `in_script`, which gives full control over the
injected amplitude and avoids the hidden renormalisation.

Under MPI, every rank evaluates the forcing on its own local subdomain — the
arrays `x` and `y` already hold each rank's local coordinates — and FluidSim
assembles the global field internally. No rank-0 guard is used.
  

### Mean Jet Profile

The Caribbean Low-Level Jet (CLLJ) is modeled as a superposition of Gaussian functions, producing a zonally uniform and climatologically realistic mean flow:

$$
\overline{u}(\varphi)=\sum_i A_i \exp\left(-\frac{(\varphi-\varphi_i)^2} {2\sigma_i^2}\right)
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
