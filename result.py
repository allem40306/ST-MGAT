import numpy as np
import util
import torch

def evaluate(pred, target):
    mape = util.masked_mape(pred, target, 0.0).item()
    rmse = util.masked_rmse(pred, target, 0.0).item()
    return mape, rmse

def evaluate_all(pred, target):
    mape = util.masked_mape(pred, target, 0.0).item()
    rmse = util.masked_rmse(pred, target, 0.0).item()
    mae = util.masked_mae(pred, target, 0.0).item()
    return mape, rmse, mae

with np.load("experiment/PEMS04_02/result.npz") as data:
    pred = torch.tensor(data["pred"])
    target = torch.tensor(data["target"])
    print(target.shape)
    for j in range(3):
        for i in range(12):
            pred_t = pred[:, :i + 1, :, j: j + 1]
            real_target = target[:, :i + 1, :, j: j + 1]
            evaluation = evaluate_all(pred_t, real_target)
            log = 'test for horizon {:d}, {:.4f} {:.4f} {:.4f}'
            print(log.format(i + 1, evaluation[2], evaluation[0], evaluation[1]))