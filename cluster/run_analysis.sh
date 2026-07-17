#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=ID
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=ex_job_name
#SBATCH --output=ex_job_name_%j.log
#SBATCH --mail-user=user@email.com
#SBATCH --mail-type=END,FAIL


cd ~/fluidsim_pixi/env-fluidsim-mpi
eval "$(pixi shell-hook)"

cd ~/CLLJ-Simulations/analysis
python post-processing-data.py

