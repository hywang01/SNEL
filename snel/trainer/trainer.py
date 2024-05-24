import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid

from snel.modeling import build_model_training
from snel.modeling import compute_average_auc
from snel.modeling.optimizer import build_optimizer
from snel.modeling.lr_scheduler import build_lr_scheduler
from snel.utils import (
    inf_loop, MetricTracker, 
    load_pretrained_weights, count_num_param)

from .baseline_trainer import BaselineTrainer
from .snel_trainer import SNELTrainer


def build_trainer(cfg):
    config = cfg
    
    if cfg.trainers.trainer_name == "baseline":
        trainer = BaselineTrainer(config)
    elif cfg.trainers.trainer_name == "snel":
        trainer = SNELTrainer(config)
    else:
        raise NotImplementedError('No implemented trainer')

    return trainer
