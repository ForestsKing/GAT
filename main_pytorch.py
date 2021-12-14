import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim

from model.gat_pytorch import GAT
from utils.make_data import make_data
from utils.setseed import set_seed

if __name__ == '__main__':
    set_seed(2021)
    idx_train = np.array(range(140))
    idx_val = np.array(range(140, 640))
    idx_test = np.array(range(1708, 2708))
    epochs = 100
    lr = 0.005

    A, features, labels = make_data()
    model = GAT(int(features.shape[1]), int(max(labels) + 1))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # train
    for epoch in range(epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, A)

        pred = torch.argmax(output, dim=1)
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train = accuracy_score(labels[idx_train], pred[idx_train])

        loss_train.backward()
        optimizer.step()

        model.eval()
        loss_val = criterion(output[idx_val], labels[idx_val])
        acc_val = accuracy_score(labels[idx_val], pred[idx_val])

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val),
              'time: {:.4f}s'.format(time.time() - t))

    # test
    model.eval()
    output = model(features, A)
    pred = torch.argmax(output, dim=1)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy_score(labels[idx_test], pred[idx_test])

    print('loss_test: {:.4f}'.format(loss_test.item()),
          'acc_test: {:.4f}'.format(acc_test))
