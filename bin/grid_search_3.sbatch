#!/bin/bash --login
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:2
#SBATCH --cpus-per-gpu=6  
#SBATCH --mem=128GB
#SBATCH --constraint=intel 
#SBATCH --partition=debug
#SBATCH --job-name=grid_search_3
#SBATCH --mail-type=ALL
#SBATCH --output=bin/%x-%j-slurm.out
#SBATCH --error=bin/%x-%j-slurm.err

#set up the environment
module purge
conda activate ./env

#launch the script
python ./src/EMNIST_grid_search.py --alpha 0.3
