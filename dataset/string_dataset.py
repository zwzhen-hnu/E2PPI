from torch.utils.data import Dataset
import numpy as np
import os
import csv
import pickle
import torch
import dgl
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_graph(prot_r_edge_path, prot_k_edge_path, prot_node_path, outdir):
    if "dock" in prot_r_edge_path:
        outdir = outdir + "dock_SHS27k_protein_graphs.pkl"
    else:
        outdir = outdir + "SHS27k_protein_graphs.pkl"

    if os.path.exists(outdir):
        with open(outdir, "rb") as tf:
            prot_graph_list = pickle.load(tf)
    else:
        prot_r_edge = np.load(prot_r_edge_path, allow_pickle=True)
        prot_k_edge = np.load(prot_k_edge_path, allow_pickle=True)
        prot_node = torch.load(prot_node_path)
        prot_graph_list = []
        for i in range(len(prot_r_edge)):
            prot_seq = []
            for j in range(prot_node[i].shape[0] - 1):
                prot_seq.append((j, j + 1))
                prot_seq.append((j + 1, j))

            # prot_g = dgl.graph(prot_edge[i]).to(device)
            prot_g = dgl.heterograph({('amino_acid', 'SEQ', 'amino_acid'): prot_seq,
                                      ('amino_acid', 'STR_KNN', 'amino_acid'): prot_k_edge[i],
                                      ('amino_acid', 'STR_DIS', 'amino_acid'): prot_r_edge[i]}).to(device)
            prot_g.ndata['x'] = torch.FloatTensor(prot_node[i]).to(device)

            prot_graph_list.append(prot_g)

        with open(outdir, "wb") as tf:
            pickle.dump(prot_graph_list, tf)

    return prot_graph_list

class string_dataset(Dataset):
    def __init__(self, processed_path, split_mode = None, mode = None):
        if split_mode != None:
            assert split_mode in ['random','bfs','dfs']
            assert mode !=  None

        if mode != None:
            assert mode in ["train", "val", "test","BS","ES","NS"]

        self.ppi_index = []

        #docked_structure

        self.docked_graph_list = load_graph(processed_path + "protein.rball.edges.dock.npy",
                                             processed_path + "protein.knn.edges.dock.npy",
                                             processed_path + "protein.nodes.dock.pt", processed_path)

        #dictonary
        self.dictionary = []  
        with open(processed_path + "/protein.SHS27k.sequences.dictionary.csv") as f:
            reader = csv.reader(f)
            for row in reader:
                self.dictionary.append(row[0])

        #ppi
        self.ppi_list = []
        with open(processed_path + "SHS27k_ppi.pkl", "rb") as f:
            self.ppi_list = pickle.load(f)

        #ppi_label
        self.labels = []
        with open(processed_path + "SHS27k_ppi_label.pkl", "rb") as f:
            self.labels = pickle.load(f)

        if split_mode != None:
            with open(processed_path + "{}.json".format(split_mode), 'r') as f:
                ppi_split_dict = json.load(f)
                if mode == "BS" or mode == "ES" or mode == "NS":
                    prot_seen = []
                    for ppi in ppi_split_dict['train_index']:
                        prot_seen.append(self.ppi_list[ppi][0])
                        prot_seen.append(self.ppi_list[ppi][1])
                    prot_seen = set(prot_seen)
                    # prot_seen = set(list(self.ppi_list[ppi_split_dict['train_index']].reshape(-1)))

                    BS_list = []
                    ES_list = []
                    NS_list = []
                    # ppis = ppi_split_dict['test_index']+ppi_split_dict['val_index']
                    # print(len(ppis))

                    for index in ppi_split_dict['test_index']:
                        if self.ppi_list[index][0] in prot_seen and self.ppi_list[index][1] in prot_seen:
                            BS_list.append(index)
                        elif self.ppi_list[index][0] not in prot_seen and self.ppi_list[index][1] not in prot_seen:
                            NS_list.append(index)
                        else:
                            ES_list.append(index)

                    if mode == "BS":
                        self.ppi_index = BS_list
                    elif mode == "ES":
                        self.ppi_index = ES_list
                    else:
                        self.ppi_index = NS_list
                    
                else :
                    self.ppi_index = ppi_split_dict[mode + '_index']
                f.close()
        else:
            for i in range(len(self.ppi_list)):
                self.ppi_index.append(i)


    def __len__(self):
        return len(self.ppi_index)


    def __getitem__(self, idx):
        data_dict = {}
        data_dict["labels"] = self.labels[self.ppi_index[idx]]
        data_dict["ppi"] = self.ppi_list[self.ppi_index[idx]]

        #docked_structure
        data_dict["docked_graph"] = self.docked_graph_list[self.ppi_index[idx]]

        return data_dict



class ProteinDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, processed_path):

        self.prot_graph_list = load_graph(processed_path + "protein.rball.edges.SHS27k.npy",
                                            processed_path + "protein.knn.edges.SHS27k.npy",
                                            processed_path + "protein.nodes.SHS27k.pt", processed_path)

    def __len__(self):
        return len(self.prot_graph_list)

    def __getitem__(self, idx):

        return self.prot_graph_list[idx]
    
def collate1(samples):
    graphs = []
    for graph in samples:
        graphs.append(graph)
    return dgl.batch_hetero(graphs)

def collate2(data):
    collated_data = {}
    collated_data["labels"] = []
    collated_data["ppi"] = []
    collated_data["docked_graph"] = []
    
    for data_dict in data:
        collated_data["labels"].append(data_dict["labels"])
        collated_data["ppi"].append(data_dict["ppi"])
        collated_data["docked_graph"].append(data_dict["docked_graph"])
        
    collated_data["labels"] = torch.FloatTensor(np.array(collated_data["labels"]))
    collated_data["docked_graph"] = dgl.batch_hetero(collated_data["docked_graph"])
    return collated_data
