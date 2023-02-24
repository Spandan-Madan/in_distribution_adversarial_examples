from __future__ import print_function, division
import os

machine_path = os.getcwd()
user_root_dir = '/'.join(machine_path.split('/')[:-2])

import torch
import sys

# from torchvision import datasets, models, transforms

from PIL import Image
import torchvision

sys.path.append('%s/training_scaffold_own/res/'%user_root_dir)
from models.models import get_model
from loader.loader import get_loader
import random
import pickle

# Learning rate parameters
BASE_LR = 0.001
EPOCH_DECAY = 30 # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.1 # factor by which the learning rate is reduced.

# DATASET INFO
NUM_CLASSES = 2 # set the number of classes in your dataset
DATA_DIR = '../data/' # to run with the sample dataset, just set to 'hymenoptera_data'

# DATALOADER PROPERTIES
BATCH_SIZE = 10 # Set as high as possible. If you keep it too high, you'll get an out of memory error.


### GPU SETTINGS
CUDA_DEVICE = 0 # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 1 # set to 1 if want to run on gpu.

# MODEL_ARCH = 'transformer'


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import collections
from collections import OrderedDict
from tqdm import tqdm
import argparse

# DATASET_NAME = 'image_test_v7_cma_resnet18_v7_subsampled'
# 
parser = argparse.ArgumentParser()
parser.add_argument('--model_file_name', type = str, required = True)
parser.add_argument('--batch_size', type = int, default = 100)
args = parser.parse_args()

MODEL_FILE_NAME = args.model_file_name
SAVE_FOLDER_NAME = MODEL_FILE_NAME.split('.pt')[0]
BATCH_SIZE = args.batch_size

model_path = '../training_models/saved_models/%s'%MODEL_FILE_NAME
loaded_model = torch.load(model_path)
loaded_model.eval();
loaded_model.cuda();


DATASET_NAME = 'image_test_v7_cma_%s'%SAVE_FOLDER_NAME
print('DATASET: %s'%DATASET_NAME)

image_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

loader_new = get_loader('multi_attribute_loader_file_list_shapenet')
if 'smadan' in user_root_dir:
    file_list_root = '%s/dataset_lists_openmind'%user_root_dir
elif 'spandan' in user_root_dir:
    file_list_root = "%s/dataset_lists_fasrc"%user_root_dir


att_path = '%s/differentiable_graphics_ml/training_models/shapenet_id_to_class_num.p'%user_root_dir
shuffles = {'train':True,'val':True,'test':False,'seen_test':False}

################ GET FROM USER CONFIG - TODO #####################
file_lists = {}
dsets = {}
dset_loaders = {}
dset_sizes = {}
for phase in ['test']:
    file_lists[phase] = "%s/%s_list_%s.txt"%(file_list_root,phase,DATASET_NAME)
    dsets[phase] = loader_new(file_lists[phase],att_path, image_transform)
    dset_loaders[phase] = torch.utils.data.DataLoader(dsets[phase], batch_size=BATCH_SIZE, shuffle = shuffles[phase], num_workers=0,drop_last=True)
    dset_sizes[phase] = len(dsets[phase])


    
def input_shift_batch(arr, w_shift, h_shift):
    empty_arr = torch.zeros((arr.shape))
    
    if w_shift <0 and h_shift<0:
        empty_arr[:,:,:224+h_shift,:224+w_shift] = arr[:,:,-h_shift:,-w_shift:]
    elif w_shift >=0 and h_shift >=0:
        empty_arr[:,:,h_shift:,w_shift:] = arr[:,:,:arr.shape[2]-h_shift,:arr.shape[3]-w_shift]
    elif w_shift >=0 and h_shift <0:
        empty_arr[:,:,:224+h_shift,w_shift:] = arr[:,:,-h_shift:,:arr.shape[3]-w_shift]
    elif w_shift <0 and h_shift >=0:
        empty_arr[:,:,h_shift:,:224+w_shift] = arr[:,:,:arr.shape[2]-h_shift,-w_shift:]
    
    return empty_arr

all_errors = 0
total = 0
for phase in ['test']:
    for data in tqdm(dset_loaders[phase]):
        count = 0
        batch_predictions = torch.zeros((25, BATCH_SIZE))
        for w_shift in range(-2, 3):
            for h_shift in range(-2, 3):
                if count % 10 == 0:
                    print("%s"%count)
                inputs_orig, labels, image_paths = data
                inputs_shifted = input_shift_batch(inputs_orig, w_shift, h_shift)
                inputs = inputs_shifted.float().cuda()
                labels = labels.long().cuda()
                
                im_means = torch.mean(inputs.view(BATCH_SIZE, -1),dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                im_stds = torch.std(inputs.view(BATCH_SIZE, -1),dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                inputs = (inputs - im_means)/im_stds

                outputs = loaded_model(inputs)
                preds = torch.argmax(outputs,dim=1)
                batch_predictions[count] = preds
                count += 1
                batch_repeated_labels = labels.repeat(25, 1)
        shifted_matches = batch_predictions.cpu() == batch_repeated_labels.cpu()
        batch_errors = torch.sum(torch.sum(shifted_matches,dim=0) != len(shifted_matches))
        all_errors += batch_errors
        total += data[0].shape[0]
        

shift_attack_accuracy = 1 - (all_errors/total)

save_string = "%s___%s"%(MODEL_FILE_NAME, shift_attack_accuracy.item())
with open('shift_accuracies.txt','a') as F:
    print(save_string, file = F)
