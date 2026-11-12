import numpy as np
import torch
from sklearn import metrics


def cal_auc(label, pred, pos_label=1, return_fpr_tpr=False, save_fpr_tpr=False):
    if type(label) == torch.Tensor:
        label = label.detach().cpu().numpy()
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy()
    fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=pos_label, drop_intermediate=False)
    auc_score = metrics.auc(fpr, tpr)
    if save_fpr_tpr:
        if auc_score > 0.5:
            np.save("./ROC_reinter/{:.0f}".format(auc_score * 10000),
                    np.concatenate([np.expand_dims(fpr, axis=1), np.expand_dims(tpr, axis=1)], axis=1))
    if return_fpr_tpr:
        return fpr, tpr, auc_score
    return auc_score


class Network_Logger(object):
    def __init__(self, model):
        self.model = model
        self.model_grad_dict = {}
        self.model_weight_dict = {}
        self.model_weightSize_dict = {}
        for (i, j) in self.model.named_parameters():
            if len(j.shape) > 1:
                self.model_grad_dict[i] = []
                self.model_weight_dict[i] = [j.abs().mean().item()]
                self.model_weightSize_dict[i] = j.shape
