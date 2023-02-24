import pickle
import utils
import pickle
import utils
import numpy as np
import matplotlib.pyplot as plt
from utils import MLP
from utils import CMA_info
import os

import torch
import random
import torch.nn as nn

import torch.distributions as D

import numpy as np
import cma
from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt

import torch.nn.functional as F

def sample_uniform(batch_size, N_dim, r1, r2):
    random_matrix = (r1 - r2) * torch.rand([batch_size, N_dim]) + r2
    return random_matrix

class MLP(nn.Module):
    def __init__(self, data_dimensions,name,seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        else:
            print("Torch seed:%s"%torch.seed())
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(data_dimensions, int(data_dimensions)),
            nn.Tanh(),
#             nn.Linear(int(data_dimensions*5), int(data_dimensions/5)),
#             nn.ReLU(),
#             nn.Linear(int(data_dimensions), int(data_dimensions/5)),
#             nn.ReLU(),
#             nn.Linear(int(data_dimensions/5), int(data_dimensions/5)),
#             nn.ReLU(),
            nn.Linear(int(data_dimensions), 2)
        )
        self.name = name
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.dataset = []

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

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

x1_min = -10
x1_max = 10

x2_min = 20
x2_max = 40

test_min = -1
test_max= 1

def make_dataset(dataset_size):
    sample_1 = sample_uniform(dataset_size, N_dim, x1_min, x1_max)
    sample_2 = sample_uniform(dataset_size, N_dim, x2_min, x2_max)

    labels_1 = torch.zeros(len(sample_1))
    labels_2 = torch.ones(len(sample_2))

    X = torch.vstack([sample_1, sample_2])
    Y = torch.hstack([labels_1, labels_2])
    ids = list(range(len(X)))

    random.shuffle(ids)
    train_ids = ids[:int(0.8*len(X))]
    test_ids = ids[int(0.8*len(X)):]

    # len(X)

    X_train = X[train_ids]
    X_test = X[test_ids]

    Y_train = Y[train_ids]
    Y_test = Y[test_ids]
    return X_train, X_test, Y_train, Y_test

def chunked_make_dataset(dataset_size, num_chunks):
    chunk_dataset_size = int(dataset_size/num_chunks)
    for i in range(num_chunks):
        print('sampling chunk...')
        chunk_dataset = make_dataset(chunk_dataset_size)
        print('writing chunk...')
        with open('chunked_datasets/chunk_dataset_%s.p'%i,'wb') as F:
            pickle.dump(chunk_dataset, F)
            
def train_model(N_dim, dataset, name, seed = None):
    X_train, X_test, Y_train, Y_test = dataset
    X_train = X_train.cuda()
    X_test = X_test.cuda()
    Y_train = Y_train.cuda()
    Y_test = Y_test.cuda()

    if seed is not None:
        model = MLP(N_dim, name, seed).cuda()
    else:
        model = MLP(N_dim, name).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    model.dataset = dataset

    for epoch in tqdm(range(1000)):
        outputs = model(X_train)
        loss = loss_fn(outputs, Y_train.long())
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(model(X_test), dim=1)
        accuracy = torch.sum(predictions == Y_test)/len(Y_test)

        train_predictions = torch.argmax(model(X_train), dim=1)
        train_accuracy = torch.sum(train_predictions == Y_train)/len(Y_train)

        if epoch == 999:
            print("Epoch:%s, Train Acc:%s"%(epoch, train_accuracy))
            print("Epoch:%s, Test Acc:%s"%(epoch, accuracy))

        model.test_accuracy = accuracy
        model.train_accuracy = train_accuracy

    return model

def batched_train_model(N_dim, dataset, name, seed = None):
    X_train, X_test, Y_train, Y_test = dataset
    if seed is not None:
        model = MLP(N_dim, name, seed).cuda()
    else:
        model = MLP(N_dim, name).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    model.dataset = dataset

    BATCH_SIZE = 64000
    print('batched training with size %s'%BATCH_SIZE)
    # if dataset[0].shape[0] > 16000:
    #     num_epochs = 100
    for epoch in tqdm(range(100)):
        num_batches = int(X_train.shape[0]/(BATCH_SIZE))

        batch_accuracies = []
        batch_train_accuracies = []
        for batch_num in range(num_batches):
            start_pt = BATCH_SIZE * batch_num
            end_pt = start_pt + BATCH_SIZE
            train_X = X_train[start_pt:end_pt]
            train_Y = Y_train[start_pt:end_pt]
            test_X = X_test
            test_Y = Y_test

            train_X = train_X.cuda()
            train_Y = train_Y.cuda()
            test_X = test_X.cuda()
            test_Y = test_Y.cuda()

            outputs = model(train_X)
            loss = loss_fn(outputs, train_Y.long())
            loss.backward()
            optimizer.step()

            batch_predictions = torch.argmax(model(test_X), dim=1)
            batch_accuracy = torch.sum(batch_predictions == test_Y)/len(test_Y)
            batch_accuracies.append(batch_accuracy.item())

            batch_train_predictions = torch.argmax(model(train_X), dim=1)
            batch_train_accuracy = torch.sum(batch_train_predictions == train_Y)/len(train_Y)
            batch_train_accuracies.append(batch_train_accuracy.item())

        accuracy = np.mean(batch_accuracies)
        train_accuracy = np.mean(batch_train_accuracies)


        if epoch%100 == 0:
            print("Epoch:%s, Train Acc:%s"%(epoch, train_accuracy))
            print("Epoch:%s, Test Acc:%s"%(epoch, accuracy))

        if epoch == 99:
            print("Epoch:%s, Train Acc:%s"%(epoch, train_accuracy))
            print("Epoch:%s, Test Acc:%s"%(epoch, accuracy))

        model.test_accuracy = accuracy
        model.train_accuracy = train_accuracy

    return model

def batched_train_model_from_disk(N_dim, dataset_chunk_folder, name, seed = None):   
    if seed is not None:
        model = MLP(N_dim, name, seed).cuda()
    else:
        model = MLP(N_dim, name).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    
    BATCH_SIZE = 64000
    chunked_dataset_names = os.listdir(dataset_chunk_folder)
    num_chunked_datasets = len(chunked_dataset_names)
    
    print('Found %s chunks'%num_chunked_datasets)
    
    for epoch in tqdm(range(10)):
        for i in range(num_chunked_datasets):
            print('Loading chunk...')
            with open('%s/%s'%(dataset_chunk_folder, chunked_dataset_names[i]),'rb') as F:
                dataset = pickle.load(F)
                
            X_train, X_test, Y_train, Y_test = dataset
            num_batches = int(X_train.shape[0]/(BATCH_SIZE))
        
            batch_accuracies = []
            batch_train_accuracies = []
            for batch_num in range(num_batches):
                start_pt = BATCH_SIZE * batch_num
                end_pt = start_pt + BATCH_SIZE
                train_X = X_train[start_pt:end_pt]
                train_Y = Y_train[start_pt:end_pt]
                test_X = X_test
                test_Y = Y_test

                train_X = train_X.cuda()
                train_Y = train_Y.cuda()
                test_X = test_X.cuda()
                test_Y = test_Y.cuda()

                outputs = model(train_X)
                loss = loss_fn(outputs, train_Y.long())
                loss.backward()
                optimizer.step()

                batch_predictions = torch.argmax(model(test_X), dim=1)
                batch_accuracy = torch.sum(batch_predictions == test_Y)/len(test_Y)
                batch_accuracies.append(batch_accuracy.item())

                batch_train_predictions = torch.argmax(model(train_X), dim=1)
                batch_train_accuracy = torch.sum(batch_train_predictions == train_Y)/len(train_Y)
                batch_train_accuracies.append(batch_train_accuracy.item())        

        accuracy = np.mean(batch_accuracies)
        train_accuracy = np.mean(batch_train_accuracies)

        
        if epoch%1 == 0:
            print("Epoch:%s, Train Acc:%s"%(epoch, train_accuracy))
            print("Epoch:%s, Test Acc:%s"%(epoch, accuracy))
        
        if epoch == 1:
            print("Epoch:%s, Train Acc:%s"%(epoch, train_accuracy))
            print("Epoch:%s, Test Acc:%s"%(epoch, accuracy))
        
        model.test_accuracy = accuracy
        model.train_accuracy = train_accuracy
    
    return model

def cma_objective(x_input):
    torch_x = torch.from_numpy(x_input).unsqueeze(0).float()
    output = CURRENT_MODEL(torch_x)
    pred_prob = output[0][CATEGORY_NUM].item()
    prediction = torch.argmax(output[0]).item()
    return pred_prob, prediction

def cma_experiment(attacked_model, num_samples = 50):
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

CATEGORY_NUM = 0
CURRENT_MODEL = None


all_info = []
# N_dim = 500
for N_dim in [20,100,500]:
    for iteration in range(5):
        print('*'*50)
        print('Iteration %s'%iteration)
        print('*'*50)

        trained_models = {}
        attack_output = {}
        for dsize in [10000000]:
            print('Working with with %s'%dsize)
            dset = chunked_make_dataset(dsize, 10)
            trained_models[dsize] = batched_train_model_from_disk(N_dim, 'chunked_datasets/', 'dset_size_%s'%dsize)
            model_to_attack = trained_models[dsize].cpu()
            attack_output[dsize] = cma_experiment(model_to_attack, 50)
        iter_info = [trained_models, attack_output]
        all_info.append(iter_info)

    with open('data_size_experiment_5_repeats_%s_dim_10000000.p'%N_dim,'wb') as F:
        pickle.dump(all_info, F)

