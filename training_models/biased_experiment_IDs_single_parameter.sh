experiment_index=$1
echo $experiment_index
if [ "$experiment_index" = "0" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v3_fixed_camera_symmetric_bias_4_conditions_lighting --pretrained False --image_size 224 --dataset_name image_train_v3_fixed_camera_symmetric_bias_4_conditions_lighting
elif [ "$experiment_index" = "1" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v3_fixed_camera_symmetric_bias_6_conditions_lighting --pretrained False --image_size 224 --dataset_name image_train_v3_fixed_camera_symmetric_bias_6_conditions_lighting
elif [ "$experiment_index" = "2" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v3_fixed_camera_symmetric_bias_8_conditions_lighting --pretrained False --image_size 224 --dataset_name image_train_v3_fixed_camera_symmetric_bias_8_conditions_lighting
elif [ "$experiment_index" = "3" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v3_fixed_lighting_symmetric_bias_4_conditions_camera --pretrained False --image_size 224 --dataset_name image_train_v3_fixed_lighting_symmetric_bias_4_conditions_camera
elif [ "$experiment_index" = "4" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v3_fixed_lighting_symmetric_bias_6_conditions_camera --pretrained False --image_size 224 --dataset_name image_train_v3_fixed_lighting_symmetric_bias_6_conditions_camera
elif [ "$experiment_index" = "5" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v3_fixed_lighting_symmetric_bias_8_conditions_camera --pretrained False --image_size 224 --dataset_name image_train_v3_fixed_lighting_symmetric_bias_8_conditions_camera
elif [ "$experiment_index" = "6" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v3_fixed_camera_unsymmetric_bias_4_conditions_lighting --pretrained False --image_size 224 --dataset_name image_train_v3_fixed_camera_unsymmetric_bias_4_conditions_lighting
elif [ "$experiment_index" = "7" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v3_fixed_camera_unsymmetric_bias_6_conditions_lighting --pretrained False --image_size 224 --dataset_name image_train_v3_fixed_camera_unsymmetric_bias_6_conditions_lighting
elif [ "$experiment_index" = "8" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v3_fixed_camera_unsymmetric_bias_8_conditions_lighting --pretrained False --image_size 224 --dataset_name image_train_v3_fixed_camera_unsymmetric_bias_8_conditions_lighting
elif [ "$experiment_index" = "9" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v3_fixed_lighting_unsymmetric_bias_4_conditions_camera --pretrained False --image_size 224 --dataset_name image_train_v3_fixed_lighting_unsymmetric_bias_4_conditions_camera
elif [ "$experiment_index" = "10" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v3_fixed_lighting_unsymmetric_bias_6_conditions_camera --pretrained False --image_size 224 --dataset_name image_train_v3_fixed_lighting_unsymmetric_bias_6_conditions_camera
elif [ "$experiment_index" = "11" ]; then
    python /om5/user/smadan/differentiable_graphics_ml/training_models/training_simple_CNN.py --num_epochs 50 --model_arch resnet18 --batch_size 125 --run_name resnet18_v3_fixed_lighting_unsymmetric_bias_8_conditions_camera --pretrained False --image_size 224 --dataset_name image_train_v3_fixed_lighting_unsymmetric_bias_8_conditions_camera
fi
