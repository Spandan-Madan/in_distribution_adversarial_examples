import pickle
import utils
import numpy as np
import matplotlib.pyplot as plt
import utils
from utils import MLP
from utils import CMA_info

CURRENT_MODEL = None
N_dim = 5
dsize = 100000

x1_min = -10
x1_max = 10

x2_min = 20
x2_max = 40

test_min = -1
test_max= 1

limits = [x1_min, x1_max, x2_min, x2_max]
test_limits = [test_min, test_max]

num_samples = 50

for repeat_iteration in range(500):
    trained_models = {}
    attack_output = {}
    ds = utils.make_dataset(dsize, N_dim, limits)
    trained_models[N_dim] = utils.train_model(N_dim, ds, 'dset_size_%s_find_failures'%dsize, disable_progress = True)
    model_to_attack = trained_models[N_dim].cpu()
    attack_output[N_dim] = utils.cma_experiment(model_to_attack, N_dim, test_limits, limits, 10, disable_progress = True)
    attack_rate = len(attack_output[N_dim].in_dist_advs)/len(attack_output[N_dim].starts)
    print(attack_rate)
    if attack_rate < 0.4:
        iter_info = [trained_models, attack_output]
        with open('failed_iter_10.p','wb') as F:
            pickle.dump(iter_info, F)
        print("Found, stopping!")
        break
