import time

import torch
from sklearn.metrics import accuracy_score
from torch import optim, nn
from torch_geometric.datasets import Planetoid

from model.gat_pyg import GAT
from utils.setseed import set_seed

if __name__ == '__main__':
    set_seed(2021)
    epochs = 100
    lr = 0.005

    dataset = Planetoid(root='./data/pyG_cora', name='Cora')
    data = dataset[0]
    model = GAT(in_features=dataset.num_node_features, out_features=dataset.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # train
    for epoch in range(epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(data)

        pred = torch.argmax(output, dim=1)
        loss_train = criterion(output[data.train_mask], data.y[data.train_mask])
        acc_train = accuracy_score(data.y[data.train_mask], pred[data.train_mask])

        loss_train.backward()
        optimizer.step()

        model.eval()
        loss_val = criterion(output[data.val_mask], data.y[data.val_mask])
        acc_val = accuracy_score(data.y[data.val_mask], pred[data.val_mask])

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val),
              'time: {:.4f}s'.format(time.time() - t))

    # test
    model.eval()
    output = model(data)
    pred = torch.argmax(output, dim=1)
    loss_test = criterion(output[data.test_mask], data.y[data.test_mask])
    acc_test = accuracy_score(data.y[data.test_mask], pred[data.test_mask])

    print('loss_test: {:.4f}'.format(loss_test.item()),
          'acc_test: {:.4f}'.format(acc_test))
