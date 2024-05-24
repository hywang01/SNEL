import numpy as np
import torch
import torch.nn.functional as F
from .base_trainer import TrainerBase

from shapley import PermutationSampler
    

class SNELTrainer(TrainerBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.diversity_metric = cfg.models.diversity_metric
        self.n_ensemble = cfg.models.n_ensemble
        self.lambda_v = cfg.models.lambda_v
        self.shapley_q = cfg.models.shapley_q 

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        x_feature = self.model.return_feature()
        
        loss_ajs = 0
        probs_y = []
        output_logits = []
        
        ensemble_weights = np.zeros((label.shape[0], self.n_ensemble+1))
        
        for i in range(self.n_ensemble):
            logit = self.model.classifier(x_feature) 
            output_logits.append(logit)
            prob_y = torch.sigmoid(logit)
            
            if self.diversity_metric == 'var':
                prob_y_vec = torch.masked_select(input=prob_y, mask=label.bool())          
                probs_y.append(prob_y_vec.unsqueeze(0))
            elif self.diversity_metric == 'shapley':
                probs_y.append(prob_y)
            else:
                loss_diversity = 0.0
        
        batch_loss = self.loss(output_logits, label)
                
        loss_ajs += batch_loss
        loss_ajs /= self.n_ensemble
        
        # measure diversity of ensembles     
        if self.diversity_metric == 'erm':
            loss_diversity = 0.0
            
        elif self.diversity_metric == 'var':
            probs_y = torch.cat(probs_y, dim=0)
            X = torch.sqrt(torch.log(2/(1+probs_y)) + probs_y * torch.log(2*probs_y/(1+probs_y)) + 1e-6)
            loss_diversity = (X.pow(2).mean(dim=0) - X.mean(dim=0).pow(2)).mean()
            
        elif self.diversity_metric == 'shapley':
            ensemble_weights = self.compute_weights(probs_y, label)
            ensemble_shapley_entropy = self.compute_shapley(ensemble_weights, 
                                                            output_value='entropy')
            loss_diversity = ensemble_shapley_entropy
        else:
            raise NotImplementedError
        
        loss = loss_ajs - self.lambda_v * loss_diversity
        
        self.model_backward_and_update(loss)
        
        loss_summary = {
            "loss": loss.item(),
            'loss_ajs': loss_ajs.item(),
            'loss_diversity': loss_diversity #TODO
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

    def compute_shapley(self, voting_weights, output_value='entropy'):

        if isinstance(voting_weights, torch.Tensor):
            voting_weights = voting_weights.cpu().detach().numpy()
        
        solver = PermutationSampler()
        solver.solve_game(voting_weights, self.shapley_q)
        
        if output_value == 'shapley':
            shapley_values = solver.get_average_shapley()
            shapley_output = shapley_values
        elif output_value == 'entropy':
            shapley_entropy = solver.get_shapley_entropy()
            shapley_output = shapley_entropy
        else:
            raise NotImplementedError
        
        return shapley_output
    
    def compute_weights(self, probs_y, label):
        label_invert = 1 - label
        batch_weight = np.zeros((label.shape[0]*label.shape[1], len(probs_y)))
        
        for i, prob_m in enumerate(probs_y):
            weight_matrix = torch.abs(prob_m - label_invert)
            weight_matrix = weight_matrix / self.n_ensemble
            weight_matrix = weight_matrix.cpu().detach().numpy()
            batch_weight[:, i] = weight_matrix.flatten()
        
        return batch_weight
