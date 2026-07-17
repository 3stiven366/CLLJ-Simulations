#!/bin/bash
#SBATCH --partition=debug
#SBATCH --account= ID
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --job-name=ex_job_name
#SBATCH --output=ex_job_name_%j.log
#SBATCH --mail-user=user@email.com
#SBATCH --mail-type=END,FAIL


cd ~/fluidsim_pixi/env-fluidsim-mpi
eval "$(pixi shell-hook)"

cd ~/CLLJ-Simulations/simulation
mpirun -n 8 python CLLJ_simulation.py




