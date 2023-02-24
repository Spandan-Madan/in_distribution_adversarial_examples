experiment_index=$1
echo $experiment_index
if [ "$experiment_index" = "0" ]; then
    python training_simple_CNN.py --num_epochs 50 --model_arch VISION_TRANSFORMER --batch_size 50 --num_classes 11 --base_lr 0.0003 --run_name train_v7_transformer_40_2 --dataset_name image_train_v7_shapenet_40 --pretrained --normalize
elif [ "$experiment_index" = "1" ]; then
    python training_simple_CNN.py --num_epochs 50 --model_arch VISION_TRANSFORMER_DEIT --batch_size 50 --num_classes 11 --base_lr 0.0003 --run_name train_v7_transformer_40_2_deit --dataset_name image_train_v7_shapenet_40 --pretrained --normalize
elif [ "$experiment_index" = "2" ]; then
    python training_simple_CNN.py --num_epochs 50 --model_arch VISION_TRANSFORMER_DEIT_DISTILLED --batch_size 50 --num_classes 11 --base_lr 0.0003 --run_name train_v7_transformer_40_2_deit_distilled --dataset_name image_train_v7_shapenet_40 --pretrained --normalize
elif [ "$experiment_index" = "3" ]; then
    python training_simple_CNN.py --num_epochs 50 --model_arch VISION_TRANSFORMER_HYBRID --batch_size 75 --num_classes 11 --base_lr 0.0003 --run_name train_v7_transformer_40_2_hybrid --dataset_name image_train_v7_shapenet_40 --pretrained --normalize
elif [ "$experiment_index" = "4" ]; then
    python training_simple_CNN.py --num_epochs 50 --model_arch VISION_TRANSFORMER_DEIT_DISTILLED --batch_size 50 --num_classes 11 --base_lr 0.0003 --run_name train_v7_transformer_2_deit_distilled --dataset_name image_train_v7_shapenet --pretrained --normalize
fi
