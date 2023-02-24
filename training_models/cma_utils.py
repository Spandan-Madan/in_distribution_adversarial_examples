from tqdm.notebook import tqdm
import torch
import cma
import numpy as np
import random

CATEGORY_NUM = 0

    
def sample_uniform(batch_size, N_dim, r1, r2):
    random_matrix = (r1 - r2) * torch.rand([batch_size, N_dim]) + r2
    return random_matrix

class CMA_info():
    def __init__(self, model_name, num_samples):
        super(CMA_info, self).__init__()
        self.model_name = model_name
        self.num_samples = num_samples
        
        self.distances = []
        self.in_dist_advs = []
        self.advs = []
        self.starts = []
    
    def summary(self):
        print('****************** CMA Summary *******************')
        print('Trained on %s points:'%self.model_name.split('_')[-1])
        print('Adversarials: %s/%s'%(len(self.advs), self.num_samples))
        print('In-distribution: %s/%s'%(len(self.in_dist_advs), len(self.advs)))
        
        avg_dist = np.mean(self.distances)
        print('Average L2 distance: %s'%(avg_dist))
        print('*'*50)
        

        
def extract_stats(all_info):
    all_rates = []
    all_dists = []
    for iter_info in all_info:
        if iter_info[1] == 'NA':
            continue
        else:
            attack_rate = len(iter_info[1].in_dist_advs)/float(len(iter_info[1].starts))
            if attack_rate > 0:
                mean_iter_dist = np.mean(iter_info[1].distances)
                all_rates.append(attack_rate)
                all_dists.append(mean_iter_dist)
    
    mean_rate = np.mean(all_rates)
    mean_dist = np.mean(all_dists)
    return mean_rate, mean_dist
        
def cma_objective(x_input):
    torch_x = torch.from_numpy(x_input).unsqueeze(0).float()
    output = CURRENT_MODEL(torch_x)
    pred_prob = output[0][CATEGORY_NUM].item()
    prediction = torch.argmax(output[0]).item()
    return pred_prob, prediction

def cma_experiment(attacked_model, N_dim, test_min, test_max, x1_min, x1_max, num_samples = 50):
    cma_search_output = {}
    global CURRENT_MODEL
    CURRENT_MODEL = attacked_model
    print(CURRENT_MODEL.name)
    cma_output = CMA_info(CURRENT_MODEL.name, num_samples)        
    
    for i in tqdm(range(num_samples)):
        initial_pred = 1
        while initial_pred == 1:
            start_pos = sample_uniform(1, N_dim, test_min, test_max)
            output = CURRENT_MODEL(start_pos)
            initial_pred = torch.argmax(output[0]).item()
        cma_output.starts.append(start_pos)
        start_pos = start_pos[0]
        
        es = cma.CMAEvolutionStrategy(start_pos, 0.00005)
        es.optimize(cma_objective, verb_disp = False, iterations=1500, correct_prediction = CATEGORY_NUM)
        adv_offspring_ids = np.where(np.array(es.predictions) != 0)
        
        if len(adv_offspring_ids[0]) > 0:
            random_adv_offspring_id = random.choice(list(adv_offspring_ids[0]))
            random_adv_offspring = es.prediction_settings[random_adv_offspring_id]
            cma_output.advs.append(random_adv_offspring)        

            max_val = np.max(random_adv_offspring)
            if max_val < x1_max:
                cma_output.in_dist_advs.append(random_adv_offspring)
                distance = np.linalg.norm(start_pos - random_adv_offspring)
                cma_output.distances.append(distance)
    return cma_output