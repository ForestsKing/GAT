import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import Normalizer


def make_data():
    # 读取数据
    idx_features_labels = pd.read_csv('./data/cora/cora.content', sep='\t', header=None)
    edges_unordered = pd.read_csv('./data/cora/cora.cites', sep='\t', header=None)
    N = len(idx_features_labels)

    idx2id = dict(zip(idx_features_labels.iloc[:, 0], idx_features_labels.index))
    features = idx_features_labels.iloc[:, 1:-1].values

    scaler = Normalizer()
    features = scaler.fit_transform(features)

    label2id = {label: i for i, label in enumerate(list(set(list(idx_features_labels.iloc[:, -1].values))))}
    labels = list(map(label2id.get, idx_features_labels.iloc[:, -1].values))

    # 构建邻接矩阵
    A = np.zeros((N, N))
    for i, j in zip(edges_unordered.iloc[:, 0], edges_unordered.iloc[:, 1]):
        A[idx2id[i]][idx2id[j]] = 1
        A[idx2id[j]][idx2id[i]] = 1

    # 计算A_hat
    A_wave = A + np.eye(N)
    D_wave = np.eye(N) * (np.sum(A_wave, axis=0) ** (-0.5))
    A_hat = np.matmul(np.matmul(D_wave, A_wave), D_wave)

    return torch.Tensor(A_hat), torch.Tensor(features), torch.LongTensor(labels)
