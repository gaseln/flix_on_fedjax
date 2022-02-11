#!/bin/bash --login
#SBATCH --time=7-0:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:5
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=400GB
#SBATCH --constraint=intel
#SBATCH --partition=batch
#SBATCH --job-name=jup_server
#SBATCH --mail-type=ALL
#SBATCH --output=bin/%x-%j-slurm.out
#SBATCH --error=bin/%x-%j-slurm.err

# use srun to launch Jupyter server in order to reserve a port
srun --resv-ports=1 ./bin/launch-jupyter-server.srun
