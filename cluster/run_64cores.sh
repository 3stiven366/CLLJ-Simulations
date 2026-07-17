#!/bin/bash
#SBATCH --partition=parallel
#SBATCH --account=ID
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --job-name=ex_job_name
#SBATCH --output=ex_job_name_%j.log
#SBATCH --mail-user=user@email.com
#SBATCH --mail-type=END,FAIL


cd ~/fluidsim_pixi/env-fluidsim-mpi
eval "$(pixi shell-hook)"

cd ~/CLLJ-Simulations/simulation
mpirun -n 64 python CLLJ_simulation.py
 

