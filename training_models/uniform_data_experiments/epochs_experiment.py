import pickle
import utils
import numpy as np
import matplotlib.pyplot as plt

CURRENT_MODEL = None
N_dim = 100


x1_min = -10
x1_max = 10

x2_min = 20
x2_max = 40

test_min = -1
test_max= 1

limits = [x1_min, x1_max, x2_min, x2_max]
test_limits = [test_min, test_max]

num_samples = 50

all_info = []

for iteration in range(10):
    print('*'*50)
    print('Iteration %s'%iteration)
    print('*'*50)
    
    trained_models = {}
    attack_output = {}
    for n_epochs in [25,100,400,1600]:
        print('Working with with %s dimensions'%N_dim)
        dsize = 100000
        dset = utils.make_dataset(dsize, N_dim, limits)
        trained_models[N_dim] = utils.train_model(N_dim, dset, 'dset_size_%s'%dsize, n_epochs)
        model_to_attack = trained_models[N_dim].cpu()
        attack_output[n_epochs] = utils.cma_experiment(model_to_attack, N_dim, test_limits, limits, num_samples)
        print("Attack: %s/%s"%(len(attack_output[n_epochs].in_dist_advs), num_samples))
    iter_info = [trained_models, attack_output]
    all_info.append(iter_info)
    
from utils import MLP
from utils import CMA_info

with open('../epoch_experiment_10_repeats.p','wb') as F:
    pickle.dump(all_info, F)