# Running Simulations on an HPC Cluster

## Overview

This directory contains example SLURM job scripts used to run the complete
simulation workflow on the UCR High Performance Computing (HPC) cluster.

The scripts cover the three main stages of the project:

- Running the numerical simulation.
- Debugging new configurations.
- Performing automated post-processing and diagnostics after the simulation has finished.

Although these scripts were developed for the UCR HPC cluster, they can be
easily adapted to other systems using the SLURM workload manager.

## Available Job Scripts

| Script | Description |
|---------|-------------|
| `run_64cores.sh` | Launches the FluidSim simulation using MPI on the HPC cluster. |
| `run_debug.sh` | Runs a short debug simulation for testing new configurations. |
| `run_analysis.sh` | Executes the post-processing pipeline after the simulation finishes. |

## Post-processing

The `run_analysis.sh` script launches the automated analysis pipeline after
a simulation has completed.

It activates the required Python environment and executes the analysis
program, which computes diagnostics, generates figures, and stores the
results for later inspection.

## Requirements

- SLURM workload manager
- MPI-enabled FluidSim installation
- Python environment with the required dependencies
## Workflow

```text
Local machine
      │
      ▼
Configure simulation
      │
      ▼
sbatch run_64cores.sh
      │
      ▼
FluidSim simulation
      │
      ▼
Simulation output
      │
      ▼
sbatch run_analysis.sh
      │
      ▼
Diagnostics 
```


## Using a Job Script

Submit a job using

```bash
sbatch run_64cores.sh
```

## Understanding the SLURM Directives

Explain things like

- `--partition`
- `--ntasks`
- `--time`
- `--mem`
- `--job-name`
- `--output`

## Monitoring Jobs

```bash
squeue -u $USER
```

## Canceling Jobs

```bash
scancel JOB_ID
```

## Adapting the Scripts

Explain which lines users will most likely need to modify:

- partition
- number of MPI processes
- wall time
- account/project
- simulation path

