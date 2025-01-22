import torch

# from sklearn.metrics import precision_recall_curve, auc


device = torch.device("cuda")
def evaluat_metrics(output, label):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    pre_y = (torch.sigmoid(output) > 0.5).numpy()
    truth_y = label.numpy()
    N, C = pre_y.shape

    for i in range(N):
        for j in range(C):
            if pre_y[i][j] == truth_y[i][j]:
                if truth_y[i][j] == 1:
                    TP += 1
                else:
                    TN += 1
            elif truth_y[i][j] == 1:
                FN += 1
            elif truth_y[i][j] == 0:
                FP += 1

        # Accuracy = (TP + TN) / (N*C + 1e-10)
        Precision = TP / (TP + FP + 1e-10)
        Recall = TP / (TP + FN + 1e-10)
        F1_score = 2 * Precision * Recall / (Precision + Recall + 1e-10)

    # aupr_entry_1 = truth_y
    # aupr_entry_2 = pre_y
    # aupr = np.zeros(7)
    # for i in range(7):
    #     precision, recall, _ = precision_recall_curve(aupr_entry_1[:, i], aupr_entry_2[:, i])
    #     aupr[i] = auc(recall, precision)
    

    return F1_score

def evaluator(model, loader):

    eval_output_list = []
    eval_labels_list = []

    model.eval()

    with torch.no_grad():
        for data_dict in loader:
            data_dict["labels"] = data_dict["labels"].to(device)
            output, loss = model(data_dict["ppi"],data_dict["labels"])
            eval_output_list.append(output.detach().cpu())
            eval_labels_list.append(data_dict["labels"].detach().cpu())

        f1_score = evaluat_metrics(torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0))


    return f1_score
