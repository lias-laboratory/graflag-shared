"""
Patched DygDataset that supports loading all data (train+test combined).
This is used by train_graflag.py to get anomaly scores for the entire dataset.
"""

import torch
import numpy as np
import pickle


class DygDatasetAll(torch.utils.data.Dataset):
    """
    Modified version of DygDataset that loads ALL data (no train/test split).
    
    Usage:
        dataset_all = DygDatasetAll(config)
        loader_all = torch.utils.data.DataLoader(dataset_all, batch_size=128, ...)
    """
    
    def __init__(self, config):
        self.config = config

        dataset_name = '{}/{}.pkl'.format(config.dir_data, config.data_set)

        with open(dataset_name, 'rb') as file:
            data = pickle.load(file)
        
        # Get ALL data (no split)
        self.input_nodes_feature, self.input_edges_feature, self.input_edges_pad, \
        self.labels, self.Tmats, self.adjs, self.eadjs, self.mask_edge = self.get_all_data(data)
    
    def get_all_data(self, data):
        """
        Process data without train/test split - returns everything.
        """
        node_features = data['nodefeatures']
        edge_features = data['edgefeatures']
        labels = data['labels']
        Tmats = data['Tmats']
        adjs = data['adjs']
        eadjs = data['eadjs']

        flattened_node = np.concatenate([arr for arr in node_features])
        unique_node = np.unique(flattened_node)
        num_nodes = int(max(unique_node.size, np.max(unique_node) + 1))

        flattened_edge = np.concatenate([arr for arr in edge_features])
        unique_edge = np.unique(flattened_edge)
        num_edges = unique_edge.size

        # Generate random features for nodes and edges
        Nfeatures = np.random.uniform(low=0.0, high=1.0, size=(num_nodes, self.config.input_dim))
        Efeatures = np.random.uniform(low=0.0, high=1.0, size=(num_edges, self.config.input_dim))

        max_mask_edge = max(len(arr) for arr in edge_features)
        masks_edge = [len(edge) for edge in edge_features]

        NS = len(edge_features)
        mask_edge = np.ones((NS, max_mask_edge))
        for i in range(NS):
            mask_edge[i, :masks_edge[i]] = 0

        hidden = Efeatures.shape[1]
        input_edges_pad = np.zeros((NS, max_mask_edge, hidden))
        for i, indices in enumerate(edge_features):
            indices = indices.astype(int)
            input_edges_pad[i, :len(indices), :] = Efeatures[indices]

        input_edges_feature = [torch.tensor(input_edges_pad[i, :masks_edge[i], :]) for i in range(NS)]

        max_mask_node = max(len(arr) for arr in node_features)
        masks_node = [len(node) for node in node_features]

        NS = len(node_features)
        hidden = Nfeatures.shape[1]
        input_nodes_pad = np.zeros((NS, max_mask_node, hidden))
        for i, indices in enumerate(node_features):
            indices = indices.astype(int)
            input_nodes_pad[i, :len(indices), :] = Nfeatures[indices]

        input_nodes_feature = [torch.tensor(input_nodes_pad[i, :masks_node[i], :]) for i in range(NS)]
        
        # NO SPLIT - return all data
        input_edges_pad = torch.tensor(input_edges_pad)
        labels = torch.tensor(labels)
        mask_edge = torch.tensor(mask_edge)
        
        return input_nodes_feature, input_edges_feature, input_edges_pad, labels, Tmats, adjs, eadjs, mask_edge

    def __getitem__(self, item):
        sinput_nodes_feature = self.input_nodes_feature[item]
        sinput_edges_feature = self.input_edges_feature[item]
        sinput_edges_pad = self.input_edges_pad[item]
        slabels = self.labels[item]
        sTmats = self.Tmats[item]
        sadjs = self.adjs[item]
        seadjs = self.eadjs[item]
        smask_edge = self.mask_edge[item]

        return {
            'input_nodes_feature': sinput_nodes_feature,
            'input_edges_feature': sinput_edges_feature,
            'input_edges_pad': sinput_edges_pad,
            'labels': slabels,
            'Tmats': sTmats,
            'adjs': sadjs,
            'eadjs': seadjs,
            'mask_edge': smask_edge
        }

    def __len__(self):
        return len(self.labels)
