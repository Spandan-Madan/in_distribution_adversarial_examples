#!/bin/bash
#SBATCH --array=0-119
#SBATCH --gres=gpu:1
#SBATCH --partition=cbmm
#SBATCH -N1
#SBATCH -t 24:00:00
#SBATCH --mem=4G
#SBATCH -o ./slurm_outputs/slurm.%N.%j.out # STDOUT
#SBATCH -e ./slurm_outputs/slurm.%N.%j.err # STDERR

# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_v7_normalized_final.pt categories_10_models_10.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_pretrained_v7_normalized_final.pt categories_10_models_10.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_antialiased_v7_normalized_final.pt categories_10_models_10.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_v7_truly_shift_invariant_normalized_final.pt categories_10_models_10.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} train_v7_transformer_2_final.pt categories_10_models_10.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} train_v7_transformer_2_deit.pt categories_10_models_10.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} train_v7_transformer_2_deit_distilled.pt categories_10_models_10.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_v7_normalized_final.pt categories_10_models_10_unseen_2.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_pretrained_v7_normalized_final.pt categories_10_models_10_unseen_2.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_antialiased_v7_normalized_final.pt categories_10_models_10_unseen_2.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_v7_truly_shift_invariant_normalized_final.pt categories_10_models_10_unseen_2.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} train_v7_transformer_2_final.pt categories_10_models_10_unseen_2.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} train_v7_transformer_2_deit.pt categories_10_models_10_unseen_2.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} train_v7_transformer_2_deit_distilled.pt categories_10_models_10_unseen_2.pkl

bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_v7_40_final.pt categories_10_models_40.pkl
bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_pretrained_v7_40_normalized_final.pt categories_10_models_40.pkl
bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_antialiased_v7_40_normalized_final.pt categories_10_models_40.pkl
bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_v7_40_truly_shift_invariant_normalized.pt categories_10_models_40.pkl
bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} train_v7_transformer_40_2.pt categories_10_models_40.pkl
bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} train_v7_transformer_40_2_deit.pt categories_10_models_40.pkl
bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} train_v7_transformer_40_2_deit_distilled.pt categories_10_models_40.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_v7_normalized_final.pt categories_10_models_10_unseen_2.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_pretrained_v7_normalized_final.pt categories_10_models_10_unseen_2.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_antialiased_v7_normalized_final.pt categories_10_models_10_unseen_2.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} resnet18_v7_truly_shift_invariant_normalized_final.pt categories_10_models_10_unseen_2.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} train_v7_transformer_2_final.pt categories_10_models_10_unseen_2.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} train_v7_transformer_2_deit.pt categories_10_models_10_unseen_2.pkl
# bash batched_cma_light.sh ${SLURM_ARRAY_TASK_ID} train_v7_transformer_2_deit_distilled.pt categories_10_models_10_unseen_2.pkl