#!/usr/bin/env python
from __future__ import print_function, division
# coding: utf-8

# In[1]:


import torch
import sys

# from torchvision import datasets, models, transforms

from PIL import Image
import torchvision

sys.path.append('/om5/user/smadan/training_scaffold_own/res/')
from models.models import get_model
from loader.loader import get_loader
import random
import pickle
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type = int, default = 200)
parser.add_argument('--model_arch', type = str, default = 'simple_cnn')
parser.add_argument('--batch_size', type = int, default = 100)
parser.add_argument('--num_classes', type = int, default = 55)
parser.add_argument('--base_lr', type = float, default = 0.001)
parser.add_argument('--cuda_device', type = int, default = 0)
parser.add_argument('--use_gpu', type = bool, default = True)
parser.add_argument('--save_path', type = str, required = True)
parser.add_argument('--log_file', type = str, required = True)
parser.add_argument('--dataset_name', type = str, required = True)

args = parser.parse_args()

config = dict(vars(args))
print(config)
