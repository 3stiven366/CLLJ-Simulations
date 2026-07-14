# Barotropic Instability Simulation

Python-based simulation of barotropic instability in geophysical flows, developed as part of undergraduate research at [CIGEFI](https://www.cigefi.ucr.ac.cr/) (Center for Geophysical Research, University of Costa Rica), under the supervision of Dr. Tito Maldonado.

## Overview

Barotropic instability is a fundamental mechanism in geophysical fluid dynamics by which small perturbations in a horizontally sheared flow grow and extract energy from the mean flow, playing a key role in the formation of eddies and large-scale turbulence in the atmosphere and ocean.

This project simulates the evolution of barotropic instabilities using [FluidSim](https://fluidsim.readthedocs.io/), running on the University of Costa Rica's institutional HPC cluster.

## What this project does

- Sets up and runs 2D barotropic flow simulations with configurable initial conditions (jet/shear profiles, perturbation amplitude, resolution).
- Processes simulation output to analyze growth rates and instability structure.
- Uses ERA5 reanalysis data and least-squares fitting to inform physically realistic parameter choices.
- Reproduces a real laboratory fluid dynamics case study for validation against experimental results.

## Tech stack

- **Python** — NumPy, Matplotlib
- **FluidSim** — pseudo-spectral simulation framework
- **HPC** — UCR institutional cluster (SLURM job submission)
- **ERA5** reanalysis data for parameter calibration

## Running a simulation

```bash
python FludSim_Barotropic.py
```

## Context

This work is part of an ongoing undergraduate research position at CIGEFI, UCR, focused on translating physical models from the scientific literature into working numerical simulations.

## Author

Estiven Hernández Alfaro — Physics student, Universidad de Costa Rica
