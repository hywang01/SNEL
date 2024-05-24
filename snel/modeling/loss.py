import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_loss(loss_name, cfg):
    
    if "bce" == loss_name:
        loss = bce_loss
    elif 'ajs' == loss_name:
        weight_target = cfg.models.loss_ajs_weight_target
        n_ensemble = cfg.models.n_ensemble
        initial_weights = [(1-weight_target)/n_ensemble for i in range(n_ensemble)]
        loss = AJS_loss(num_classes=cfg.datasets.num_classes,
                        weight_target=weight_target,
                        weights=initial_weights)
    else:
        raise NotImplementedError('No implemented loss function')

    return loss

def bce_loss(output, label):

    return F.binary_cross_entropy_with_logits(output, label)


def custom_kl_div(prediction, target):
    output_pos = target * (target.clamp(min=1e-7).log() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, axis=1)
    return output.mean()

    
class AJS_loss(torch.nn.Module):
    def __init__(self, num_classes, weight_target, weights):
        super(AJS_loss, self).__init__()
        self.num_classes = num_classes
        self.weigt_target = weight_target
        self.weights = [weight_target] + [float(w) for w in weights]
        
        scaled = True
        if scaled:
            self.scale = -1.0 / ((1.0-self.weights[0]) * np.log((1.0-self.weights[0])))
        else:
            self.scale = 1.0
        assert abs(1.0 - sum(self.weights)) < 0.001
    
    def forward(self, pred, labels):
        preds = list()
        if type(pred) == list:
            for i, p in enumerate(pred):
                preds.append(torch.sigmoid(p)) 
        else:
            preds.append(torch.sigmoid(pred))

        #labels = F.one_hot(labels, self.num_classes).float() 
        distribs = [labels] + preds
        assert len(self.weights) == len(distribs)

        mean_distrib = sum([w*d for w,d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()
        
        jsw = sum([w*custom_kl_div(mean_distrib_log, d) for w,d in zip(self.weights, distribs)])
        
        return self.scale * jsw
