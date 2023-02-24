experiment_index=$1
echo $experiment_index
if [ "$experiment_index" = "0" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_around_3d_fixed_lighting.py --dataset_name train_v3_shapenet_fixed_lighting_10_models --model_files_pickle_name categories_10_models_10.pkl --category 02691156
elif [ "$experiment_index" = "1" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_around_3d_fixed_lighting.py --dataset_name train_v3_shapenet_fixed_lighting_10_models --model_files_pickle_name categories_10_models_10.pkl --category 02818832
elif [ "$experiment_index" = "2" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_around_3d_fixed_lighting.py --dataset_name train_v3_shapenet_fixed_lighting_10_models --model_files_pickle_name categories_10_models_10.pkl --category 02958343
elif [ "$experiment_index" = "3" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_around_3d_fixed_lighting.py --dataset_name train_v3_shapenet_fixed_lighting_10_models --model_files_pickle_name categories_10_models_10.pkl --category 03001627
elif [ "$experiment_index" = "4" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_around_3d_fixed_lighting.py --dataset_name train_v3_shapenet_fixed_lighting_10_models --model_files_pickle_name categories_10_models_10.pkl --category 03467517
elif [ "$experiment_index" = "5" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_around_3d_fixed_lighting.py --dataset_name train_v3_shapenet_fixed_lighting_10_models --model_files_pickle_name categories_10_models_10.pkl --category 03624134
elif [ "$experiment_index" = "6" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_around_3d_fixed_lighting.py --dataset_name train_v3_shapenet_fixed_lighting_10_models --model_files_pickle_name categories_10_models_10.pkl --category 03790512
elif [ "$experiment_index" = "7" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_around_3d_fixed_lighting.py --dataset_name train_v3_shapenet_fixed_lighting_10_models --model_files_pickle_name categories_10_models_10.pkl --category 03928116
elif [ "$experiment_index" = "8" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_around_3d_fixed_lighting.py --dataset_name train_v3_shapenet_fixed_lighting_10_models --model_files_pickle_name categories_10_models_10.pkl --category 03948459
elif [ "$experiment_index" = "9" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_around_3d_fixed_lighting.py --dataset_name train_v3_shapenet_fixed_lighting_10_models --model_files_pickle_name categories_10_models_10.pkl --category 04256520
elif [ "$experiment_index" = "10" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_around_3d_fixed_lighting.py --dataset_name train_v3_shapenet_fixed_lighting_10_models --model_files_pickle_name categories_10_models_10.pkl --category 04379243
fi
