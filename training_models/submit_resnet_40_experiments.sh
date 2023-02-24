#!/bin/bash
#SBATCH --array=0-2
#SBATCH --gres=gpu:tesla-k80
#SBATCH --partition=cbmm
#SBATCH -N1
#SBATCH -c2
#SBATCH -t 50:00:00
#SBATCH --mem=12G
#SBATCH -J resnet_40_experiments
#SBATCH -o ./slurm_outputs/slurm.%N.%j.out # STDOUT
#SBATCH -e ./slurm_outputs/slurm.%N.%j.err # STDERR

bash resnet_40_experiments.sh ${SLURM_ARRAY_TASK_ID}
