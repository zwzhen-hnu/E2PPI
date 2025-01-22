import dgl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import function as fn
from dgl.utils import expand_as_pair

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GINE_layer(nn.Module):
    def __init__(self,
                 apply_func=None,
                 init_eps=0,
                 learn_eps=False,edge_dim=None):
        super(GINE_layer, self).__init__()
        self.apply_func = apply_func
        self.edge_transform = None
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))
            
        if edge_dim is not None:
            if isinstance(self.apply_func, torch.nn.Sequential):
                apply_func = self.apply_func[0]
            if hasattr(apply_func, 'in_features'):
                in_channels = apply_func.in_features
            elif hasattr(apply_func, 'in_channels'):
                in_channels = apply_func.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.edge_transform0 = nn.Linear(edge_dim, edge_dim)
            self.edge_transform = nn.Linear(edge_dim, in_channels)

    def message(self, edges):
        if self.edge_transform is not None:
            edge_feat = F.relu(self.edge_transform0(edges.data['he']))
            edge_feat = F.relu(self.edge_transform(edge_feat))
        else:
            edge_feat = edges.data['he']
            
        return {'m': F.relu(edges.src['hn'] + edge_feat)}

    def forward(self, graph, node_feat, edge_feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(node_feat, graph)
            graph.srcdata['hn'] = feat_src
            graph.edata['he'] = edge_feat
            graph.update_all(self.message, fn.sum('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst
        

class GINE(torch.nn.Module):
    def __init__(self, param,edge_dim = None):
        super(GINE, self).__init__()

        self.num_layers = param['ppi_num_layers']
        self.layers = nn.ModuleList()

        self.layers.append(GINE_layer(nn.Sequential(nn.Linear(param['prot_hidden_dim'] * 2, param['ppi_hidden_dim']),
                                                 nn.ReLU(),
                                                 nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']),
                                                 nn.ReLU(),
                                                 nn.BatchNorm1d(param['ppi_hidden_dim'])),
                                   learn_eps=True,edge_dim=edge_dim))

        for i in range(self.num_layers - 1):
            self.layers.append(GINE_layer(nn.Sequential(nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']),
                                                     nn.ReLU(),
                                                     nn.BatchNorm1d(param['ppi_hidden_dim'])),
                                       learn_eps=True,edge_dim=edge_dim))

        self.linear = nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim'])
        self.fc = nn.Linear(param['ppi_hidden_dim'], param['output_dim'])

    def forward(self, g, x,e, ppi):

        for l, layer in enumerate(self.layers):
            x = layer(g, x, e)

        x = F.dropout(F.relu(self.linear(x)), p=0.5, training=self.training)
        node_id = np.array(ppi)
        x1 = x[node_id[:,0]]
        x2 = x[node_id[:,1]]

        x = self.fc(torch.mul(x1, x2))

        return x