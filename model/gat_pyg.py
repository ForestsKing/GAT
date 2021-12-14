from torch import nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_features, out_features, h_dim=8, n_head=8, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()

        self.conv1 = GATConv(in_features, h_dim, heads=n_head, dropout=dropout, concat=True, negative_slope=alpha)
        self.conv2 = GATConv(h_dim*n_head, out_features, heads=n_head, dropout=dropout, concat=False, negative_slope=alpha)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        return x
