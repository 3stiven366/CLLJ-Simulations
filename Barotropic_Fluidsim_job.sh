#!/bin/bash
#SBATCH --job-name=fluidsim_barotropic
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1            
#SBATCH --time=24:00:00
#SBATCH --output=fluidsim_%j.log


cd ~/fluidsim_pixi/env-fluidsim-mpi
eval "$(pixi shell-hook)"

cd ~/Inestabilidad_Barotropica
mpirun -n 16 python FluidSim_Barotropic.py

