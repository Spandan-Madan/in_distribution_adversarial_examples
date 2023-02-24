import os
import pickle
import numpy as np
import torch
import sys
sys.path.append('../inverse_rendering/')
machine_path = os.getcwd()
user_root_dir = '/'.join(machine_path.split('/')[:-2])
sys.path.insert(0,'%s/redner/'%user_root_dir)
import redner
import pyredner
print(pyredner.__file__)
pyredner.render_pytorch.print_timing = False
from general_imports import *
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_file_name', type = str, required = True)
args = parser.parse_args()


MODEL_FILE_NAME = args.model_file_name
SAVE_FOLDER_NAME = MODEL_FILE_NAME.split('.pt')[0]

with open('../training_models/shapenet_class_num_to_class_name.p','rb') as F:
    shapenet_class_num_to_class_name = pickle.load(F)
with open('../training_models/shapenet_id_to_class_num.p','rb') as F:
    shapenet_id_to_class_num = pickle.load(F)
    
    
model_path = '../training_models/saved_models/%s'%MODEL_FILE_NAME
loaded_model = torch.load(model_path)
softmax_layer = nn.Softmax()



def clone_scene_params(scene_params):
    original_scene_params = [0,0,0,0]
    original_scene_params[0] = scene_params[0]

    original_scene_params[1] = {}
    for key in scene_params[1].keys():
        original_scene_params[1][key] = scene_params[1][key].clone()

    original_scene_params[2] = {}
    for key in scene_params[2].keys():
        original_scene_params[2][key] = [i.clone() for i in scene_params[2][key]]

    original_scene_params[3] = scene_params[3]
    return original_scene_params

def start_up(scene_params, optimized_params):
    model_file, camera_params, light_params, material_settings = scene_params
    variables = []
    var_names_list = []
    for param in optimized_params:
        if param in camera_params.keys():
            optimize_flags(camera_params[param])
            if type(camera_params[param]) == list:
                variables.extend(camera_params[param])
            else:
                variables.append(camera_params[param])
        elif param in light_params.keys():
            optimize_flags(light_params[param])
            if type(light_params[param]) == list:
                variables.extend(light_params[param])
            else:
                variables.append(light_params[param])

    scene_params = [model_file, camera_params, light_params, material_settings]
    scene = setup_scene(scene_params)

    return scene, variables

def load_geometry(model_file, geometry, mat):
    if geometry:
        obj_model_all = model_file
        obj_model = [i for i in obj_model_all if len(i.vertices)>0]
    else:
        obj_model_all = pyredner.load_obj(model_file, return_objects=True)
        obj_model = [i for i in obj_model_all if len(i.vertices)>0]

    for part in obj_model:
        part.material = mat

    return obj_model

def render_input(scene):
    img = pyredner.render_pathtracing(scene = scene, num_samples = 512, seed = 1)
    img = torch.clamp(img, min = 0.00000001)
    img = torch.pow(img, 1.0/2.2)
    img = img*255/torch.max(img)
    inputs = img.permute(2,0,1).unsqueeze(0)
    return inputs, img

def setup_scene(scene_params):

    model_file, camera_params, light_params, material_settings = scene_params
    obj_model = load_geometry(model_file, False, material_settings['diffuse_material'])

    #### Camera Setup ####
    scene_cam = pyredner.automatic_camera_placement(obj_model, resolution = (224, 224),
                                                   fov = torch.tensor([camera_params['fov']]),
                                                   up = camera_params['cam_up'],
                                                   look_at = camera_params['cam_look_at'])
    scene_cam.position = camera_params['camera_position']

    #### Lights Setup ####
    scene_lights = []
    num_lights = len(light_params['all_light_positions'])
    for i in range(num_lights):
        scene_light = pyredner.generate_quad_light(position = light_params['all_light_positions'][i],
                                     look_at = light_params['all_light_look_ats'][i],
                                     size = light_params['all_light_sizes'][i],
                                     intensity = light_params['all_light_intensities'][i],
                                     directly_visible = False)
        scene_lights.append(scene_light)

    all_objects = obj_model + scene_lights
    scene = pyredner.Scene(objects = all_objects, camera = scene_cam)

    return scene

def show_inputs(inp, save_path = False, title='No Title'):
    plt.imshow(inp[0].cpu().permute(1,2,0).int())
    plt.title(title)
    plt.axis('off')
    if save_path != False:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        

varying_camera = True
varying_light = False

def plot_exp(exp_path, save_folder):
    with open(exp_path, 'rb') as F:
        exp_info = pickle.load(F)

    starting_scene_params = exp_info['scene_params']

    
def evaluate_exp(exp_path, render = False, show = False):
    with open(exp_path, 'rb') as F:
        exp_info = pickle.load(F)
    
    starting_scene_params = exp_info['scene_params']
    all_predictions = exp_info['es'].predictions
    all_prediction_settings = exp_info['es'].prediction_settings
    correct_prediction = exp_info['es'].correct_prediction

    ids = np.where(np.array(all_predictions) != correct_prediction)
    
    if len(ids[0]) != 0:
        found_adv = True
        adv_predictions = np.array(all_predictions)[ids]
        adv_prediction_settings = np.array(all_prediction_settings)[ids]

        adv_id = random.choice(range(len(ids)))

        starting_scene_params = exp_info['scene_params']

        adv_scene_params = clone_scene_params(starting_scene_params)
        if varying_camera:
            adv_scene_params[1]['camera_position'] = torch.tensor(adv_prediction_settings[adv_id][:3]).float()
            adv_scene_params[1]['cam_look_at'] = torch.tensor(adv_prediction_settings[adv_id][3:6]).float()
            adv_scene_params[1]['cam_up'] = torch.tensor(adv_prediction_settings[adv_id][6:9]).float()
            adv_scene_params[1]['fov'] = torch.tensor(adv_prediction_settings[adv_id][9]*10).float()
        elif varying_light:
            num_lights = int(adv_prediction_settings[adv_id].shape[0]/11)
            adv_scene_params[2]['all_light_positions'] = torch.tensor(adv_prediction_settings[adv_id][:num_lights*3]).float().view(num_lights,-1)
            adv_scene_params[2]['all_light_look_ats'] = torch.tensor(adv_prediction_settings[adv_id][num_lights*3:num_lights*6]).float().view(num_lights,-1)
            adv_scene_params[2]['all_light_intensities'] = torch.tensor(adv_prediction_settings[adv_id][num_lights*6:num_lights*9]).float().view(num_lights,-1)
            adv_scene_params[2]['all_light_sizes'] = torch.tensor(adv_prediction_settings[adv_id][num_lights*9:]).float().view(num_lights,-1)
        params = [starting_scene_params, adv_scene_params]
    else:
        found_adv = False
        params = [starting_scene_params]
    
    inps = []
    predictions = []
    imgs = []
    if render:
        print('Rendering')
        for i in range(len(params)):
            param = params[i]
            scene, variables = start_up(param, [])

            inputs, img = render_input(scene)
            imgs.append(img)
            
            inputs = inputs.cuda()
            rendered_inputs = inputs.clone()
            inputs = inputs/255.0
            im_means = torch.mean(inputs).unsqueeze(0).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            im_stds = torch.std(inputs).unsqueeze(0).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            inputs_ = (inputs - im_means)/im_stds
            outputs = loaded_model(inputs_)
            probability = outputs[0][correct_prediction].item()
            prediction = torch.argmax(outputs[0]).item()
            predictions.append(prediction)
            inps.append(rendered_inputs)    
            
        if len(inps) == 1:
            inps.append(inps[0])
            predictions.append(predictions[0])
    
    if show:
        print('Showing')
        fig,ax = plt.subplots(nrows=1,ncols=2)
        ax[0].imshow(inps[0][0].cpu().permute(1,2,0).int())
        ax[0].axis('off')
        ax[0].set_title('Starting: %s'%(shapenet_class_num_to_class_name[predictions[0]].split(',')[0]))
        ax[1].imshow(inps[1][0].cpu().permute(1,2,0).int())
        ax[1].axis('off')        
        ax[1].set_title('Adversarial: %s'%(shapenet_class_num_to_class_name[predictions[1]].split(',')[0]))
        plt.show()
        
    return found_adv, inps, predictions, imgs

folder = '/om5/user/smadan/differentiable_graphics_ml/other_optimization_methods/cma_adversarial_cam/neurips_results/%s'%SAVE_FOLDER_NAME


save_folder = "%s/saved_images/"%folder
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
    
rendered_save_folder = "%s/saved_renders/"%folder
if not os.path.isdir(rendered_save_folder):
    os.mkdir(rendered_save_folder)
    
success_count = 0
total = 0
for exp in tqdm(os.listdir(folder)):
    if '.p' in exp:
        exp_p = "%s/%s"%(folder, exp)        
        found_adv, rendered_inputs, predictions, images = evaluate_exp(exp_p, render = True, show = False)
        
        starting_render_path = "%s/%s_starting_%s.png"%(rendered_save_folder, exp_p.split('/')[-1].split('.p')[0], predictions[0])
        start_image = Image.fromarray(images[0].cpu().numpy().astype('uint8'))
        start_image.save(starting_render_path)

        if found_adv == True:
            adv_render_path = "%s/%s_adversarial_%s.png"%(rendered_save_folder, exp_p.split('/')[-1].split('.p')[0], predictions[1])
            adv_image = Image.fromarray(images[1].cpu().numpy().astype('uint8'))
            adv_image.save(adv_render_path)
            success_count += 1
        total += 1
        
attack_accuracy = success_count/total
model_accuracy = 1 - attack_accuracy
save_string = "%s___%s"%(MODEL_FILE_NAME, model_accuracy)
with open('cma_measured_accuracy.txt','a') as F:
    print(save_string, file = F)