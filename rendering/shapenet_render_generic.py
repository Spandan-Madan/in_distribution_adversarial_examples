import os

import torch
import torch.nn as nn
from torch.autograd import Variable, grad
# import pyredner
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import random
from tqdm import tqdm

import sys
sys.path.insert(0,'/om5/user/smadan/redner/')
import pyredner
import pickle
import argparse


print('PyRedner location: %s'%pyredner.__file__)
cam_pos =  torch.tensor([1.0, 1.0, 1.0])
light_pos = torch.tensor([.5, .5, -0.5])
light_intensity = torch.tensor([5000.0, 5000.0, 5000.0])


parser = argparse.ArgumentParser()
parser.add_argument('--num_lights', type = int, default = 2)
parser.add_argument('--num_repeats', type = int, default = 2)
parser.add_argument('--save_path', type = str, default = '/om5/user/smadan/differentiable_graphics_ml/data/shapenet_rendered_simple')
parser.add_argument('--model_files_pickle', type = str, required = True)
parser.add_argument('--start_category', type = str)
args = parser.parse_args()

NUM_LIGHTS = args.num_lights
NUM_REPEATS = args.num_repeats
SAVE_PATH = args.save_path
MODEL_FILES_PICKLE = args.model_files_pickle
if args.start_category:
    START_CATEGORY = args.start_category


SHAPENET_DIR = '/om5/user/smadan/ShapeNetCore.v2'

def get_light_positions(pos, num_lights):
    all_combinations = [[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]]
    x, y, z = pos[0], pos[1], pos[2]
    chosen_combs = random.sample(all_combinations, num_lights)
    
    new_light_positions = []
    for comb in chosen_combs:
        new_light_positions.append(torch.tensor([x*comb[0], y*comb[1], z*comb[2]]))
    return new_light_positions


def render_shapenet_obj(obj_path, camera_position, light_position, light_intensity, show_lights = False):
    obj_model_all = pyredner.load_obj(obj_path, return_objects=True)
    obj_model = [i for i in obj_model_all if len(i.vertices)>0]
    m = pyredner.Material(diffuse_reflectance = torch.tensor((0.8, 0.8, 0.8), device = pyredner.get_device()))
    for part in obj_model:
        part.material = m

    scene_cam = pyredner.automatic_camera_placement(obj_model, resolution = (224, 224))
    scene_cam.position = camera_position
    scene_cam.look_at[0] = random.uniform(-.2,.2) + scene_cam.look_at[0]
    scene_cam.look_at[1] = random.uniform(-.2,.2) + scene_cam.look_at[1]
    scene_cam.look_at[2] = random.uniform(-.2,.2) + scene_cam.look_at[2]
#     scene_light = pyredner.generate_quad_light(position = light_position,
#                                          look_at = torch.zeros(3),
#                                          size = torch.tensor([0.01, 0.01]),
#                                          intensity = light_intensity )
    
    all_light_positions = get_light_positions(light_pos, NUM_LIGHTS)
    scene_lights = []
    
    for l_pos in all_light_positions:
        scene_light = pyredner.generate_quad_light(position = l_pos,
                                         look_at = torch.zeros(3),
                                         size = torch.tensor([0.5, 0.5]),
                                         intensity = light_intensity,
                                         directly_visible = show_lights)
        scene_lights.append(scene_light)
    
    all_objects = obj_model + scene_lights
    # scene_light.light_intensity = li
    scene = pyredner.Scene(objects = all_objects, camera = scene_cam)
    print('staring')
    img = pyredner.render_pathtracing(scene,num_samples=256,use_secondary_edge_sampling=False)
    print('saving')    
    im = torch.pow(img.data, 1.0/2.2).cpu()
    im = im*255/torch.max(im)
    image = Image.fromarray(im.numpy().astype('uint8'))
    return image, torch.sum(im)



with open('/om5/user/smadan/differentiable_graphics_ml/rendering/%s'%MODEL_FILES_PICKLE,'rb') as F:
    sampled_per_category_instances = pickle.load(F)

render_cat = 0
for repeat_num in range(NUM_REPEATS): 
    for category in sorted(sampled_per_category_instances.keys()):
        if repeat_num == 0 and category == START_CATEGORY:
            render_cat = 1
        if render_cat == 0:
            print('skipping category: %s'%category)
        elif render_cat == 1:
            category_dir = "%s/%s"%(SHAPENET_DIR, category)
            instance_model_files = sampled_per_category_instances[category]
            for model_file in tqdm(instance_model_files):
                print(model_file)
                instance = model_file.split('/')[7]
                randomized_light_pos = torch.tensor([random.uniform(-2,2), random.uniform(-2,2), random.uniform(-2,2)])
                randomized_cam_pos = torch.tensor([random.uniform(-2,2), random.uniform(-2,2), random.uniform(-2,2)])
                try:
                    print('trying rendering') 
                    rendered_im, im_sum = render_shapenet_obj(model_file, randomized_cam_pos, randomized_light_pos, light_intensity, False)
                    image_name = "%s/%s_%s_%s_%s_%s.png"%(SAVE_PATH, category, instance, randomized_light_pos[0].item(), randomized_light_pos[1].item(), randomized_light_pos[2].item())
                    rendered_im.save(image_name)
                except:
                    print('failed on %s %s'%(category, instance))
