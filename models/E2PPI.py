import torch
from torch import nn
import dgl
from tqdm import tqdm
import pickle
import os

from .codebook import CodeBook
from .GINE import GINE

class E2PPI(nn.Module):
    def __init__(self,param,node_loader,edge_loader,ppi_list):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_model = CodeBook(param)
        self.vae_model.load_state_dict(
            torch.load("./saved_models/vae_model.ckpt",
                       map_location=self.device))
        self.vae_model = self.vae_model.to(self.device)

        self.prot_embed_list = []
        self.edge_embed_list = []
        self.ppi_g = dgl.to_bidirected(dgl.graph(ppi_list))
        self.ppi_g = self.ppi_g.to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        #single protein representation
        for batch in tqdm(node_loader):
            batch = batch.to(self.device)
            emb = self.vae_model.Protein_Encoder.forward(self.vae_model.vq_layer,batch).to(self.device)
            self.prot_embed_list.append(emb)
        #interact protein representation
        for batch in tqdm(edge_loader):
            batch["docked_graph"] = batch["docked_graph"].to(self.device)
            emb = self.vae_model.Protein_Encoder.forward(self.vae_model.vq_layer,batch["docked_graph"]).to(self.device)
            self.edge_embed_list.append(emb)
            
        self.prot_embed_list = torch.cat(self.prot_embed_list, dim=0)
        self.edge_embed_list = torch.cat(self.edge_embed_list, dim=0)

        #match edge and edge_embed
        self.new_edge_embed_list = []
        indexs = []
        if os.path.exists('./edge_index.pkl'):
             with open("./edge_index.pkl", "rb") as tf:
                indexs = pickle.load(tf)
        else:
            x = self.ppi_g.edges()
            for i in tqdm(range(len(ppi_list) * 2)):
                ppi = [x[0][i], x[1][i]]
                try:
                    index = ppi_list.index(ppi)
                except :
                    ppi = [x[1][i], x[0][i]]
                    index = ppi_list.index(ppi)

                indexs.append(index)

            with open('./edge_index.pkl', 'wb') as f:
                pickle.dump(indexs, f)
            
        for index in indexs:
            try :
                self.new_edge_embed_list.append(torch.unsqueeze(self.edge_embed_list[index],dim = 0))
            except :
                print(index)
        
          
        self.edge_embed_list = torch.cat(self.new_edge_embed_list, dim=0)
        self.ppi_encoder = GINE(param,param["prot_hidden_dim"]*2).to(self.device)

    def forward(self, ppi, labels):
        output = self.ppi_encoder(self.ppi_g, self.prot_embed_list, self.edge_embed_list, ppi)
        loss = self.loss_fn(output, labels)

        return output,loss


