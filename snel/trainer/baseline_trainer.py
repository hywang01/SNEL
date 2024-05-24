import numpy as np
import torch
from torchvision.utils import make_grid
from .base_trainer import TrainerBase
from snel.modeling import compute_average_auc


class BaselineTrainer(TrainerBase):
    """Baseline model.
 
    """
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss(output, label)
        self.model_backward_and_update(loss)
        
        loss_summary = {
            "loss": loss.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        image = batch["img"]
        label = batch["lab"]

        image = image.to(self.device)
        label = label.to(self.device)

        return image, label
