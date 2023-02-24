experiment_index=$1
echo $experiment_index
if [ "$experiment_index" = "0" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v6_normalized --normalize --image_size 224 --dataset_name image_train_v6_shapenet --base_lr 0.0003
elif [ "$experiment_index" = "1" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v5_zero_removed --image_size 224 --dataset_name image_train_v5_shapenet_zero_removed --base_lr 0.0003
elif [ "$experiment_index" = "2" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v5_zero_removed_normalized --normalize --image_size 224 --dataset_name image_train_v5_shapenet_zero_removed --base_lr 0.0003
elif [ "$experiment_index" = "3" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v7_normalized --normalize --image_size 224 --dataset_name image_train_v7_shapenet --base_lr 0.0003
fi
