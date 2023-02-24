#!/bin/bash
#SBATCH --array=1
#SBATCH --gres=gpu:tesla-k80:2
#SBATCH --partition=cbmm
#SBATCH -N1
#SBATCH -c2
#SBATCH -t 50:00:00
#SBATCH --mem=12G
#SBATCH -J transformer_40_experiments
#SBATCH -o ./slurm_outputs/slurm.%N.%j.out # STDOUT
#SBATCH -e ./slurm_outputs/slurm.%N.%j.err # STDERR

bash transformer_experiments.sh ${SLURM_ARRAY_TASK_ID}
