import torch
import torchvision
from torch import nn
import random
from tqdm.notebook import tqdm
import numpy as np

class MLP(nn.Module):
    def __init__(self, data_dimensions,name,seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        else:
            print("Torch seed:%s"%torch.seed())
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(data_dimensions, int(data_dimensions)),
            nn.ReLU(),
            nn.Linear(int(data_dimensions), 2),
            nn.ReLU(),
        )
        self.name = name
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.dataset = []

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
def sample_uniform(batch_size, N_dim, r1, r2):
    random_matrix = (r1 - r2) * torch.rand([batch_size, N_dim]) + r2
    return random_matrix

def make_dataset(dataset_size, N_dim, x1_min, x1_max, x2_min, x2_max):
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

def train_model(N_dim, dataset, name, seed = None, verbose = True):
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
            if verbose:
                print("Epoch:%s, Train Acc:%s"%(epoch, train_accuracy))
                print("Epoch:%s, Test Acc:%s"%(epoch, accuracy))

        model.test_accuracy = accuracy
        model.train_accuracy = train_accuracy
    
    return model


def batched_train_model(N_dim, dataset, name, seed = None, verbose = True):
    X_train, X_test, Y_train, Y_test = dataset
    if seed is not None:
        model = MLP(N_dim, name, seed).cuda()
    else:
        model = MLP(N_dim, name).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    model.dataset = dataset
    
    BATCH_SIZE = 64000
    
    for epoch in tqdm(range(1000)):
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
            if verbose:
                print("Epoch:%s, Train Acc:%s"%(epoch, train_accuracy))
                print("Epoch:%s, Test Acc:%s"%(epoch, accuracy))

        if epoch == 999:
            if verbose:
                print("Epoch:%s, Train Acc:%s"%(epoch, train_accuracy))
                print("Epoch:%s, Test Acc:%s"%(epoch, accuracy))

        model.test_accuracy = accuracy
        model.train_accuracy = train_accuracy
    
    return model

