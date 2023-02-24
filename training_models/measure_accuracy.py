from __future__ import print_function, division

import os
import torch
import sys
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
import random
import pickle
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

machine_path = os.getcwd()
user_root_dir = '/'.join(machine_path.split('/')[:-2])
sys.path.append('%s/training_scaffold_own/res/'%user_root_dir)
from models.models import get_model
from loader.loader import get_loader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type = str, required = True)
parser.add_argument('--model_file_name', type = str, required = True)
parser.add_argument('--batch_size', type = int, default = 100)
parser.add_argument('--normalize', action = 'store_true')
args = parser.parse_args()

DATASET_NAME = args.dataset_name
BATCH_SIZE = args.batch_size
MODEL_FILE_NAME = args.model_file_name
NORMALIZE = args.normalize

transforms_without_crop = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
image_transform = {}
image_transform['train'] = transforms_without_crop
image_transform['test'] = transforms_without_crop

NUM_CLASSES = 11

loader_new = get_loader('multi_attribute_loader_file_list_shapenet')
if 'smadan' in user_root_dir:
    file_list_root = '%s/dataset_lists_openmind'%user_root_dir
elif 'spandan' in user_root_dir:
    file_list_root = "%s/dataset_lists_fasrc"%user_root_dir


att_path = '%s/differentiable_graphics_ml/training_models/shapenet_id_to_class_num.p'%user_root_dir
shuffles = {'train':True,'val':True,'test':False}


################ GET FROM USER CONFIG - TODO #####################
file_lists = {}
dsets = {}
dset_loaders = {}
dset_sizes = {}
for phase in ['test']:
    file_lists[phase] = "%s/%s_list_%s.txt"%(file_list_root,phase,DATASET_NAME)
    dsets[phase] = loader_new(file_lists[phase],att_path, image_transform[phase])
    dset_loaders[phase] = torch.utils.data.DataLoader(dsets[phase], batch_size=BATCH_SIZE, shuffle = shuffles[phase], num_workers=0,drop_last=True)
    dset_sizes[phase] = len(dsets[phase])
    
    
model_path = '../training_models/saved_models/%s'%MODEL_FILE_NAME
loaded_model = torch.load(model_path)
loaded_model.cuda();
if isinstance(loaded_model, torch.nn.DataParallel):
    loaded_model = loaded_model.module
model_name = model_path.split('/')[-1].split('.p')[0]


all_corrects = 0
total = 0
incorrect_predicted_images = []
all_paths = []
for phase in ['test']:
    for data in tqdm(dset_loaders[phase]):
        inputs, labels, image_paths = data
        inputs = inputs.float().cuda()
        labels = labels.long().cuda()
        if NORMALIZE:
            im_means = torch.mean(inputs.view(BATCH_SIZE, -1),dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            im_stds = torch.std(inputs.view(BATCH_SIZE, -1),dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            inputs = (inputs - im_means)/im_stds

        outputs = loaded_model(inputs)
        preds = torch.argmax(outputs,dim=1)
        corrects = torch.sum(preds == labels).item()
        all_corrects += corrects
        total += len(preds)
        all_paths.extend(image_paths)
        incorrect_predicted_images.extend([image_paths[i] for i in torch.where(preds!=labels)[0]])

acc = all_corrects/total
print_string = "%s___%s___%s"%(DATASET_NAME, MODEL_FILE_NAME, acc)
with open('prediction_results.txt','a') as F:
    print(print_string, file = F)