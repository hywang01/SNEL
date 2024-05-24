import os
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import roc_auc_score


def build_evaluator(cfg):
    
    if "cxr_auc" in cfg.trainers.test_evaluator:
        evaluator = CxrEvaluator(cfg)
    else:
        print('No evaluator')

    return evaluator


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class CxrEvaluator(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self.cfg = cfg
        self._y_true = []
        self._y_pred = []
        assert self.cfg.datasets.num_classes == len(self.cfg.datasets.finding_names)
        
    def call_label_pred(self):
        eval_data_info = dict()
        eval_data_info['number of labels'] = len(self._y_true)
        eval_data_info['number of predictions'] = len(self._y_pred)
        
        return eval_data_info
        
    def reset(self):
        self._y_true = []
        self._y_pred = []

    def process(self, mo, gt, activate=True):
        if activate:
            mo = torch.sigmoid(mo)
        else:
            None

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(mo.data.cpu().numpy().tolist())

    def evaluate(self, save_preds=False):
        results = OrderedDict()
        save_path = None
        if save_preds:
            save_root = './preds'
            save_name = '_'.join([self.cfg.datasets.name, self.cfg.models.model_name])
            save_path = os.path.join(save_root, save_name)
        auc_list = compute_auc(self._y_true, self._y_pred, save_path=save_path) # TODO
        class_names = sorted(self.cfg.datasets.finding_names)
        for i in range(len(class_names)):
            results[class_names[i]] = auc_list[i]
        results['average_auc'] = np.array(auc_list).mean()
        
        info = []
        info += ['=> per-class result']
        info += ['number of labels: ', len(self._y_true)]
        info += ['number of predictions: ', len(self._y_pred)]
        for i in range(len(class_names)):
            info += [class_names[i] + ': ', '{:.4f}'.format(auc_list[i])]
        info += ['* average auc: {:.4f}'.format(results['average_auc'])]

        return results


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def compute_auc(gt, pred, save_path=None):
    AUROCs = []
    gt_np = np.array(gt)
    pred_np = np.array(pred)
    
    if save_path:
        pd.DataFrame(data=gt_np).to_csv(save_path+'_label.csv')
        pd.DataFrame(data=pred_np).to_csv(save_path+'_pred.csv')
    
    for i in range(gt_np.shape[1]):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError:
            AUROCs.append(0)

    return AUROCs

def compute_average_auc(pred, gt):
    AUROCs = []
    gt_np = np.array(gt)
    pred_np = np.array(pred)
    for i in range(gt_np.shape[1]):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError:
            AUROCs.append(0)
            
    average_auc = sum(AUROCs)/len(AUROCs)
    
    return average_auc