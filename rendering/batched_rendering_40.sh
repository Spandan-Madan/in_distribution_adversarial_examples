experiment_index=$1
echo $experiment_index
if [ "$experiment_index" = "0" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_complexer_light_and_camera_inverted.py --dataset_name train_v7_shapenet_40 --model_files_pickle_name categories_10_models_40.pkl --num_repeats 1000 --category 02691156
elif [ "$experiment_index" = "1" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_complexer_light_and_camera_inverted.py --dataset_name train_v7_shapenet_40 --model_files_pickle_name categories_10_models_40.pkl --num_repeats 1000 --category 02818832
elif [ "$experiment_index" = "2" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_complexer_light_and_camera_inverted.py --dataset_name train_v7_shapenet_40 --model_files_pickle_name categories_10_models_40.pkl --num_repeats 1000 --category 02958343
elif [ "$experiment_index" = "3" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_complexer_light_and_camera_inverted.py --dataset_name train_v7_shapenet_40 --model_files_pickle_name categories_10_models_40.pkl --num_repeats 1000 --category 03001627
elif [ "$experiment_index" = "4" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_complexer_light_and_camera_inverted.py --dataset_name train_v7_shapenet_40 --model_files_pickle_name categories_10_models_40.pkl --num_repeats 1000 --category 03467517
elif [ "$experiment_index" = "5" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_complexer_light_and_camera_inverted.py --dataset_name train_v7_shapenet_40 --model_files_pickle_name categories_10_models_40.pkl --num_repeats 1000 --category 03624134
elif [ "$experiment_index" = "6" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_complexer_light_and_camera_inverted.py --dataset_name train_v7_shapenet_40 --model_files_pickle_name categories_10_models_40.pkl --num_repeats 1000 --category 03790512
elif [ "$experiment_index" = "7" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_complexer_light_and_camera_inverted.py --dataset_name train_v7_shapenet_40 --model_files_pickle_name categories_10_models_40.pkl --num_repeats 1000 --category 03928116
elif [ "$experiment_index" = "8" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_complexer_light_and_camera_inverted.py --dataset_name train_v7_shapenet_40 --model_files_pickle_name categories_10_models_40.pkl --num_repeats 1000 --category 03948459
elif [ "$experiment_index" = "9" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_complexer_light_and_camera_inverted.py --dataset_name train_v7_shapenet_40 --model_files_pickle_name categories_10_models_40.pkl --num_repeats 1000 --category 04256520
elif [ "$experiment_index" = "10" ]; then
    source ~/.bashrc
    conda activate pytorch3d
    python render_complexer_light_and_camera_inverted.py --dataset_name train_v7_shapenet_40 --model_files_pickle_name categories_10_models_40.pkl --num_repeats 1000 --category 04379243
fi
