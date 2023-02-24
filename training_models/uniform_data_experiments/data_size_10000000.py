import pickle
import utils
import numpy as np
import matplotlib.pyplot as plt
from utils import MLP
from utils import CMA_info

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

all_info = []

for iteration in range(5):
    print('*'*50, flush = True)
    print('Iteration %s'%iteration)
    print('*'*50, flush = True)
    
    trained_models = {}
    attack_output = {}
    for dsize in [10000000]:
        print('Working with with %s'%dsize, flush = True)
        dset = utils.make_dataset(dsize, N_dim, limits)
        if dsize > 32000:
            trained_models[dsize] = utils.batched_train_model(N_dim, dset, 'dset_size_%s'%dsize)
        else:
            trained_models[dsize] = utils.train_model(N_dim, dset, 'dset_size_%s'%dsize)
        model_to_attack = trained_models[dsize].cpu()
        attack_output[dsize] = utils.cma_experiment(model_to_attack, N_dim, test_limits, limits, 50)
    iter_info = [trained_models, attack_output]
    all_info.append(iter_info)
    
    
with open('../data_size_experiment_5_repeats_10000000.p','wb') as F:
    pickle.dump(all_info, F)
