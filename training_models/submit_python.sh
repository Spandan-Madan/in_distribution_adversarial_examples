#!/bin/bash
#SBATCH --array=0
#SBATCH --gres=gpu:1
#SBATCH --partition=cbmm
#SBATCH -N2
#SBATCH -t 40:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH -J data_size_experiments
#SBATCH -o ./slurm_outputs/find_failures_.%N.%j.out # STDOUT
#SBATCH -e ./slurm_outputs/find_failures_.%N.%j.err # STDERR

bash python_script.sh ${SLURM_ARRAY_TASK_ID} 
