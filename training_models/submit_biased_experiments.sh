#!/bin/bash
#SBATCH --array=0-24
#SBATCH --gres=gpu:tesla-k80
#SBATCH --partition=cbmm
#SBATCH -N1
#SBATCH -c2
#SBATCH -t 20:00:00
#SBATCH --mem=8G
#SBATCH -J biased_experiments
#SBATCH -o ./slurm_outputs/slurm.%N.%j.out # STDOUT
#SBATCH -e ./slurm_outputs/slurm.%N.%j.err # STDERR

bash unsymmetric_biased.sh ${SLURM_ARRAY_TASK_ID}
