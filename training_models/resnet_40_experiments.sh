experiment_index=$1
echo $experiment_index
if [ "$experiment_index" = "0" ]; then
    python training_simple_CNN.py --num_epochs 50 --model_arch resnet18_antialiased --batch_size 75 --num_classes 11 --base_lr 0.0003 --run_name resnet18_antialiased_v7_40_normalized --dataset_name image_train_v7_shapenet_40 --normalize
elif [ "$experiment_index" = "1" ]; then
    python training_simple_CNN.py --num_epochs 50 --model_arch TRULY_SHIFT_INVARIANT --batch_size 50 --num_classes 11 --base_lr 0.0003 --run_name resnet18_v7_40_truly_shift_invariant_normalized --dataset_name image_train_v7_shapenet_40 --normalize
elif [ "$experiment_index" = "2" ]; then
    python training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --num_classes 11 --base_lr 0.0003 --run_name resnet18_pretrained_v7_40_normalized --dataset_name image_train_v7_shapenet_40 --pretrained --normalize
fi
