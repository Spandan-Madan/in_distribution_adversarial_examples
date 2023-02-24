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


with open('../data_size_experiment_5_repeats_10000000.p','rb') as F:
    all_info = pickle.load(F)

attack_rates = []
for dset in [10000000]:
    for i in range(5):
        total_points = all_info[i][1][dset].starts
        total_failures = all_info[i][1][dset].in_dist_advs
        attack_rate = len(total_failures)/len(total_points)
        attack_rates.append(attack_rate)


dist_means = []
dist_stds = []

for dset in [10000000]:
    overall_distances = []
    for i in range(5):
        all_distances = all_info[i][1][dset].distances
        dist_mean = np.mean(all_distances)
        dist_std = np.std(all_distances)
        overall_distances.extend(all_distances)
        dist_means.append(dist_mean)
        dist_stds.append(dist_std)
    overall_mean = np.mean(overall_distances)
    overall_std = np.std(overall_distances)

to_store = [attack_rates, dist_means, dist_stds, overall_mean, overall_std]

with open('data_10000000_stats.p','wb') as F:
    pickle.dump(to_store, F)
