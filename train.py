param = {
      "dataset": "SHS27k",
      "split_mode": "random",
      "input_dim": 7,
      "output_dim": 7,
      "ppi_hidden_dim": 4096,
      "prot_hidden_dim": 256, 
      "ppi_num_layers": 3,
      "prot_num_layers": 4,
      "learning_rate": 1e-4,
      "weight_decay": 1e-4,
      "max_epoch": 500,
      "batch_size": 32,
      "dropout_ratio": 0.0,
      "commitment_cost": 0.25,
      "num_embeddings": 1024, 
      "mask_ratio": 0.15,
      "sce_scale": 1.5,
      "mask_loss": 1,
      "seed": 114514,
}

import torch
import copy
from utils.evaluation import evaluat_metrics,evaluator
from dataset.string_dataset import string_dataset,ProteinDatasetDGL,collate1,collate2
import random
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(param["seed"])

processed_dir = "/root/autodl-tmp/processed_data/"
#dataset
graph_data = ProteinDatasetDGL(processed_dir)
edge_data = string_dataset(processed_dir)

train_dataset = string_dataset(processed_dir, split_mode = param["split_mode"], mode = 'train')
val_dataset = string_dataset(processed_dir, split_mode = param["split_mode"], mode = 'val')
test_dataset = string_dataset(processed_dir, split_mode = param["split_mode"], mode = 'test')

from torch.utils.data import DataLoader
from utils.my_log import getLogger
logger = getLogger("E2PPI")

graph_loader = DataLoader(graph_data, batch_size=param["batch_size"], shuffle=False, collate_fn=collate1)
edge_loader = DataLoader(edge_data, batch_size=param["batch_size"], shuffle=False, collate_fn=collate2)

train_loader = DataLoader(train_dataset, batch_size=param["batch_size"], shuffle=False, pin_memory=True, collate_fn=collate2)
val_loader = DataLoader(val_dataset, batch_size=param["batch_size"], shuffle=False, pin_memory=True, collate_fn=collate2)
test_loader = DataLoader(test_dataset, batch_size=param["batch_size"], shuffle=False, pin_memory=True, collate_fn=collate2)

from models.E2PPI import E2PPI
model = E2PPI(param,graph_loader,edge_loader, edge_data.ppi_list)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=param["learning_rate"],
    weight_decay=param["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        
es = 0
val_best = 0
test_val = 0
test_best = 0
best_epoch = 0

for epoch in range(param["max_epoch"]):
    f1_sum = 0.0
    loss_sum = 0.0
    model.train()
    for data_dict in train_loader:
        data_dict["labels"] = data_dict["labels"].to(device)
        output, loss = model(data_dict["ppi"],data_dict["labels"])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        f1_score = evaluat_metrics(output.detach().cpu(), data_dict["labels"].detach().cpu())
        f1_sum += f1_score
        
    scheduler.step(loss_sum / len(train_loader))
        
    val_f1_score = evaluator(model, val_loader)
    test_f1_score = evaluator(model, test_loader)
    
    logger.info("Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}".format(epoch, loss_sum / len(train_loader), f1_sum / len(train_loader), val_f1_score, test_f1_score, val_best, test_val, test_best, best_epoch))

    if test_f1_score > test_best:
        test_best = test_f1_score

    if val_f1_score >= val_best:
        val_best = val_f1_score
        test_val = test_f1_score
        state = copy.deepcopy(model.state_dict())
        es = 0
        best_epoch = epoch
    else:
        es += 1
        
    if es == 100:
        print("Early stopping!")
        break
        
torch.save(state, "./saved_models/E2PPI_state.pth")
model.load_state_dict(state)

test_f1_score = evaluator(model, test_loader)
print("results:",test_f1_score)
        