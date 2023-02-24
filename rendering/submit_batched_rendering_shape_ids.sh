#!/bin/bash
#SBATCH --array=0-10
#SBATCH --gres=gpu:1
#SBATCH --partition=cox
#SBATCH -N2
#SBATCH -t 48:00:00
#SBATCH --mem=8G
#SBATCH -o ./slurm_outputs/slurm.%N.%j.out # STDOUT
#SBATCH -e ./slurm_outputs/slurm.%N.%j.err # STDERR

bash batched_rendering_shape_ids.sh ${SLURM_ARRAY_TASK_ID}
