import torch
import torch.nn.functional as F
from torch import nn


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, n_head=8, dropout=0.6, alpha=0.2, last=False):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_head = n_head
        self.dropout = dropout
        self.alpha = alpha
        self.last = last

        Ws = []
        As = []
        for _ in range(self.n_head):
            W = nn.Linear(in_features, out_features)
            A = nn.Linear(2 * out_features, 1)
            nn.init.xavier_uniform_(W.weight, gain=1.414)
            nn.init.xavier_uniform_(A.weight, gain=1.414)
            Ws.append(W)
            As.append(A)
        self.Ws = torch.nn.ModuleList(Ws)
        self.As = torch.nn.ModuleList(As)

    def forward(self, h, mask):
        outs = []
        for i in range(self.n_head):
            W = self.Ws[i]
            A = self.As[i]

            Wh = W(h)
            Whi = Wh.unsqueeze(1).repeat(1, Wh.size(0), 1)
            Whj = Wh.unsqueeze(0).repeat(Wh.size(0), 1, 1)
            Whi_cat_Whj = torch.cat((Whi, Whj), dim=2)

            e = A(Whi_cat_Whj).squeeze()
            e = F.leaky_relu(e, negative_slope=self.alpha)

            zero_vec = -9e15 * torch.ones_like(e)
            a = torch.where(mask > 0, e, zero_vec)
            a = F.softmax(a, dim=1)
            a = F.dropout(a, self.dropout, training=self.training)

            aWh = torch.mm(a, Wh)

            if self.last:
                out = aWh
            else:
                out = F.elu(aWh)
            outs.append(out)
        if self.last:
            outs = torch.mean(torch.stack(outs, dim=2), dim=2)
        else:
            outs = torch.cat(outs, dim=1)
        return outs


class GAT(nn.Module):
    def __init__(self, in_features, out_features, h_dim=8, n_head=8, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.conv1 = GATLayer(in_features, h_dim, n_head, dropout, alpha, False)
        self.conv2 = GATLayer(h_dim*n_head, out_features, n_head, dropout, alpha, True)

    def forward(self, X, A):
        hidden = self.conv1(X, A)
        out = self.conv2(hidden, A)

        return out
