import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import MyModel


seed = 2025
torch.manual_seed(seed=seed)
train_test_rate = 0.2
lr = 0.05
batchsize = 32
num_epoches = 100
restart = True

device = torch.device('cuda')

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_uniform_(m._parameters[param])


if __name__ == '__main__':

    stocks = pd.read_pickle('data/stock_code.pkl')
    stocks = stocks['stock'].values

    X = []
    Y = []

    pred_X = []
    pred_Y = []

    for stock in tqdm(stocks):
        if os.path.exists('train/' + str(stock) + '.pkl') == False:
            break
        data = pd.read_pickle('train/' + str(stock) + '.pkl')
        X.append(data[:-1])
        pred_X.append([data[-1]])
        data = pd.read_pickle('pred/' + str(stock) + '.pkl')
        Y.append(data[:-1])
        pred_Y.append([data[-1]])
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    pred_x = np.concatenate(pred_X, axis=0)

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=train_test_rate, random_state=seed, shuffle=True)
    assert (X_train.shape[0] == Y_train.shape[0] and X_valid.shape[0] == Y_valid.shape[0]), f'shape error{X_train.shape[0]} {Y_train.shape[0]} and {X_valid.shape[0]} {Y_valid.shape[0]}'
    X_train = torch.Tensor(X_train).to(device)
    X_valid = torch.Tensor(X_valid).to(device)
    Y_train = torch.Tensor(Y_train).to(device)
    Y_valid = torch.Tensor(Y_valid).to(device)

    loss = nn.MSELoss(reduction='sum')
    if restart:
        model = MyModel(6, 128, 12, 60, 1, 0.3)
        model.to(device)
            
        model.apply(xavier_init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        data = MyDataset(X_train, Y_train)
        data_iter = DataLoader(data, batchsize, shuffle=True)

        loss_recoder = []
        for epoch in tqdm(range(num_epoches)):
            for batch in data_iter:
                optimizer.zero_grad()
                X, Y = [x for x in batch]
                #X.shape is batch_size, num_steps(same as Y)
                dec_input = Y[:, :60]
                #expected_Y = Y[:, -60:]
                expected_Y = Y[:, 60]
                Y_hat, state = model(X, dec_input)
                state
                expected_Y = expected_Y.unsqueeze(dim=1)
                l = loss(Y_hat, expected_Y)
                loss_recoder.append(l)
                l.backward()
                optimizer.step()

        torch.save(model, 'trained_model/model.pth')
    else:
        model = torch.load('trained_model/model.pth')

    print("_________start evaluate__________")

    model.eval()
    Y_pred, _ = model(X_valid, Y_valid[:, :60])
    #Y_origin = Y_valid[:, -60:].unsqueeze(dim=1)
    Y_origin = Y_valid[:, 60].unsqueeze(dim=1)
    l = loss(Y_pred, Y_origin)
    print(l)
    pd.to_pickle(Y_pred, 'result/pred.pkl')
    pd.to_pickle(Y_origin, 'result/origin.pkl')


